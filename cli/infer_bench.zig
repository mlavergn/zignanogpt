const std = @import("std");
const log = std.log.scoped(.zignanogpt_cli_infer_bench);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// What `infer-bench` measures with.
pub const InferBenchSettings = struct {
    source: mod.CheckpointKind = .base,
    tag: ?[]const u8 = null,
    step: ?usize = null,
    prompt_tokens: usize = 2048,
    decode_tokens: usize = 256,
    batch_sizes: []const u8 = "1,8,32,128",
    temperature: f32 = 0,
    threads: usize = 0,
    /// The device's peak memory bandwidth (bytes/s) and compute (FLOP/s); null
    /// leaves MBU / MFU unreported (nanochat looks them up by NVIDIA GPU name).
    peak_bandwidth: ?f64 = null,
    peak_flops: ?f64 = null,
};

/// One timed generation: time to first token and every decode step.
const Timing = struct {
    ttft: f64,
    steps: []f64,
};

/// `zignanogpt infer-bench`: nanochat's `infer_bench.py`. Prefill throughput
/// at batch 1, then a sweep over decode batch sizes: time to first token,
/// median time per output token, tokens/s, and (given the device's peaks)
/// bandwidth and compute utilization. The last line is the whole run as JSON.
pub const InferBench = struct {
    pub const usage =
        \\usage: zignanogpt infer-bench [-i <source>] [-g <tag>] [-s <step>] [--prompt-tokens <n>] [--decode-tokens <n>]
        \\                              [--batch-sizes <list>] [-t <temperature>] [--peak-bandwidth <bytes/s>]
        \\                              [--peak-flops <flop/s>] [--threads <n>]
        \\  -i, --source         base, sft or rl (default: base)
        \\  --prompt-tokens      prefill length (default 2048; clamped so prompt + decode fits)
        \\  --decode-tokens      tokens generated per row (default 256)
        \\  --batch-sizes        comma-separated decode batch sizes (default 1,8,32,128)
        \\  --peak-bandwidth     the device's memory bandwidth in bytes/s, for MBU (e.g. 546e9)
        \\  --peak-flops         the device's f32 FLOP/s, for MFU
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: the card, the table and the JSON line.
    ///
    /// Return: the exit code; loading and generation errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (args.flag("help")) {
            try out.writeAll(usage);
            return 0;
        }
        const settings = parse(args) catch |err| {
            log.debug("infer-bench usage error [{t}]", .{err});
            try out.writeAll(usage);
            return 2;
        };
        const allocator = init.gpa;
        var batches: std.ArrayList(usize) = .empty;
        defer batches.deinit(allocator);
        var it = std.mem.splitScalar(u8, settings.batch_sizes, ',');
        while (it.next()) |text| {
            const b = std.fmt.parseInt(usize, std.mem.trim(u8, text, " "), 10) catch 0;
            if (b == 0) {
                try out.print("--batch-sizes: '{s}' is not a positive number\n", .{text});
                return 2;
            }
            try batches.append(allocator, b);
        }

        var backend = try mod.Backend.init(allocator, init.io, .{ .threads = settings.threads });
        defer backend.deinit();
        const storage = mod.Storage.init(allocator, init.io);
        var config = try mod.Config.load(allocator, init.environ_map, storage);
        defer config.deinit();
        const loaded = try allocator.create(mod.LoadedModel);
        defer allocator.destroy(loaded);
        try loaded.init(allocator, &backend, storage, config.base_dir, .{ .kind = settings.source, .tag = settings.tag, .step = settings.step });
        defer loaded.deinit();
        const model_config = loaded.model.config;

        // Clamp the prompt so prompt + decode fits in the training context.
        const max_prompt = model_config.sequence_len -| settings.decode_tokens;
        const prompt_len = @min(settings.prompt_tokens, max_prompt);
        if (prompt_len < settings.prompt_tokens) try out.print("note: clamping prompt to {d} tokens so prompt+decode fits sequence_len={d}\n", .{ prompt_len, model_config.sequence_len });
        if (prompt_len == 0) {
            try out.writeAll("--decode-tokens leaves no room for a prompt\n");
            return 2;
        }
        const prompt = try buildPrompt(allocator, &loaded.tokenizer, prompt_len);
        defer allocator.free(prompt);

        // Static card: the inference cost the architecture implies.
        const num_params = model_config.numParams();
        const w_bytes = num_params * @sizeOf(f32);
        const kv_store = model_config.kvBytesPerToken();
        const context_mid = prompt_len + settings.decode_tokens / 2;
        const kv_read = model_config.kvReadBytes(context_mid);
        const decode_flops = model_config.decodeFlops(context_mid);
        const bar = "=" ** 100;
        try out.print("{s}\nModel: {t} {s} (step {d}) | depth {d}, dim {d}, heads {d}, kv heads {d} (GQA)\n", .{ bar, settings.source, loaded.tag, loaded.step, model_config.n_layer, model_config.n_embd, model_config.n_head, model_config.n_kv_head });
        try out.print("Device: {s} backend | peak bandwidth {s} | peak compute {s}\n{s}\n", .{ mod.Backend.name, try peakText(allocator, settings.peak_bandwidth, 1e12, "TB/s"), try peakText(allocator, settings.peak_flops, 1e12, "TFLOPS"), "-" ** 100 });
        try out.print("Parameters: {d} (f32) | weight bytes as stored: {d} MiB\n", .{ num_params, w_bytes / (1 << 20) });
        try out.print("KV cache: {d} bytes/token stored | {d} bytes read/step at context {d} (window pattern {s})\n", .{ kv_store, kv_read, context_mid, model_config.window_pattern });
        const ceiling: ?f64 = if (settings.peak_bandwidth) |bw| bw / @as(f64, @floatFromInt(w_bytes + kv_read)) else null;
        if (ceiling) |c| try out.print("Theoretical decode ceiling at batch 1: {d:.0} tok/s\n", .{c});
        try out.print("{s}\n", .{bar});
        try out.flush();

        var json: std.Io.Writer.Allocating = .init(allocator);
        defer json.deinit();
        var jw: std.json.Stringify = .{ .writer = &json.writer };
        try jw.beginObject();
        try field(&jw, "source", @tagName(settings.source));
        try field(&jw, "step", loaded.step);
        try field(&jw, "model_config", model_config);
        try field(&jw, "device", mod.Backend.name);
        try field(&jw, "peak_bandwidth_bytes_per_sec", settings.peak_bandwidth);
        try field(&jw, "num_params", num_params);
        try field(&jw, "weight_bytes", w_bytes);
        try field(&jw, "kv_bytes_per_token", kv_store);
        try field(&jw, "kv_read_bytes_per_step", kv_read);
        try field(&jw, "context_mid", context_mid);
        try field(&jw, "peak_flops_per_sec", settings.peak_flops);
        try field(&jw, "decode_flops_per_token", decode_flops);
        try field(&jw, "ceiling_bs1_tok_per_sec", ceiling);
        try field(&jw, "prompt_tokens", prompt_len);
        try field(&jw, "decode_tokens", settings.decode_tokens);
        try field(&jw, "temperature", settings.temperature);

        // Prefill: batch 1, one decode step, so the time to first token is the prefill.
        const engine = mod.Engine.init(&loaded.model, &loaded.tokenizer);
        _ = try timeGeneration(init.io, allocator, engine, prompt, 1, 2, settings.temperature); // warmup
        const prefill = try timeGeneration(init.io, allocator, engine, prompt, 1, 2, settings.temperature);
        allocator.free(prefill.steps);
        const prefill_tok_per_sec = @as(f64, @floatFromInt(prompt_len)) / prefill.ttft;
        const prefill_mfu: ?f64 = if (settings.peak_flops) |pf| 100 * @as(f64, @floatFromInt(model_config.prefillFlops(prompt_len))) / prefill.ttft / pf else null;
        try out.print("Prefill (batch 1, {d} tokens): {d:.0} tok/s | MFU {s}\n", .{ prompt_len, prefill_tok_per_sec, try percent(allocator, prefill_mfu) });
        try jw.objectField("prefill");
        try jw.beginObject();
        try field(&jw, "tok_per_sec", prefill_tok_per_sec);
        try field(&jw, "mfu_percent", prefill_mfu);
        try field(&jw, "time_sec", prefill.ttft);
        try jw.endObject();

        // The sweep: decode reads every weight and the KV cache each step (MBU
        // binds at small batch); FLOPs grow with the batch (MFU at large batch).
        const header = "  batch   TTFT ms   TPOT ms      tok/s   MBU %   MFU %  steps";
        try out.print("{s}\n{s}\n", .{ header, "-" ** header.len });
        try out.flush();
        try jw.objectField("sweep");
        try jw.beginArray();
        for (batches.items) |batch| {
            _ = try timeGeneration(init.io, allocator, engine, prompt, batch, 8, settings.temperature); // warmup
            const t = try timeGeneration(init.io, allocator, engine, prompt, batch, settings.decode_tokens, settings.temperature);
            defer allocator.free(t.steps);
            if (t.steps.len == 0) {
                try out.print("{d:>7}  every row ended at its first token; skipping\n", .{batch});
                continue;
            }
            std.mem.sort(f64, t.steps, {}, std.sort.asc(f64));
            const tpot = t.steps[t.steps.len / 2];
            var total: f64 = 0;
            for (t.steps) |s| total += s;
            const tok_per_sec = @as(f64, @floatFromInt(batch * t.steps.len)) / total;
            const mbu: ?f64 = if (settings.peak_bandwidth) |bw| 100 * (@as(f64, @floatFromInt(w_bytes + batch * kv_read)) / tpot) / bw else null;
            const mfu: ?f64 = if (settings.peak_flops) |pf| 100 * (@as(f64, @floatFromInt(batch * decode_flops)) / tpot) / pf else null;
            const early = t.steps.len != settings.decode_tokens -| 1;
            try out.print("{d:>7} {d:>9.1} {d:>9.2} {d:>10.0} {s:>7} {s:>7} {d:>6}", .{ batch, t.ttft * 1e3, tpot * 1e3, tok_per_sec, try percent(allocator, mbu), try percent(allocator, mfu), t.steps.len });
            if (early) try out.print(" (early stop @ {d})", .{t.steps.len});
            try out.writeByte('\n');
            try out.flush();
            try jw.beginObject();
            try field(&jw, "batch_size", batch);
            try field(&jw, "ttft_sec", t.ttft);
            try field(&jw, "tpot_sec", tpot);
            try field(&jw, "tok_per_sec", tok_per_sec);
            try field(&jw, "mbu_percent", mbu);
            try field(&jw, "mfu_percent", mfu);
            try field(&jw, "decode_steps", t.steps.len);
            try jw.endObject();
        }
        try jw.endArray();
        try jw.endObject();
        // The last line of stdout is the machine-readable version of the whole run.
        try out.print("{s}\n{s}\n", .{ "-" ** header.len, json.written() });
        try out.flush();
        return 0;
    }

    /// Parses the options.
    ///
    /// Parameters:
    /// - `args`: the command's options; every one must be consumed.
    ///
    /// Return: the settings; `error.InvalidValue` or an unknown option.
    pub fn parse(args: *cli.Args) !InferBenchSettings {
        var s: InferBenchSettings = .{};
        if (try either(args, "source", "i")) |text| {
            s.source = std.meta.stringToEnum(mod.CheckpointKind, text) orelse {
                log.warn("--source: '{s}' is not base, sft or rl", .{text});
                return error.InvalidValue;
            };
        }
        s.tag = try either(args, "model-tag", "g");
        if (try either(args, "step", "s")) |text| s.step = std.fmt.parseInt(usize, text, 10) catch return invalid("step", text);
        s.prompt_tokens = try args.int(usize, "prompt-tokens", s.prompt_tokens);
        s.decode_tokens = try args.int(usize, "decode-tokens", s.decode_tokens);
        if (try args.string("batch-sizes")) |text| s.batch_sizes = text;
        if (try either(args, "temperature", "t")) |text| {
            const t = std.fmt.parseFloat(f32, text) catch return invalid("temperature", text);
            if (!(t >= 0)) return invalid("temperature", text);
            s.temperature = t;
        }
        s.threads = try args.int(usize, "threads", 0);
        if (try args.string("peak-bandwidth")) |text| s.peak_bandwidth = try positive("peak-bandwidth", text);
        if (try args.string("peak-flops")) |text| s.peak_flops = try positive("peak-flops", text);
        if (s.decode_tokens < 2) return invalid("decode-tokens", "below 2");
        try args.finish();
        return s;
    }

    // -------------------------------------------------------------------------
    // Private helpers

    /// Times one generation of `max_tokens` per row: the first token (prefill,
    /// the copy to `batch` rows, the first sample), then each decode step.
    fn timeGeneration(io: std.Io, allocator: std.mem.Allocator, engine: mod.Engine, prompt: []const u32, batch: usize, max_tokens: usize, temperature: f32) !Timing {
        // `generate` runs the prefill and the copy to `batch` rows; the first
        // `next` samples the first token. Each `next` returns the column's
        // tokens on the host, so the work is done when it returns.
        var start = std.Io.Clock.awake.now(io);
        var gen = try engine.generate(allocator, prompt, .{ .num_samples = batch, .max_tokens = max_tokens, .temperature = temperature, .seed = 42 });
        defer gen.deinit();
        var steps: std.ArrayList(f64) = .empty;
        errdefer steps.deinit(allocator);
        _ = try gen.next();
        const ttft = seconds(start, io);
        while (true) {
            start = std.Io.Clock.awake.now(io);
            if (try gen.next() == null) break;
            try steps.append(allocator, seconds(start, io));
        }
        return .{ .ttft = ttft, .steps = try steps.toOwnedSlice(allocator) };
    }

    fn seconds(start: std.Io.Timestamp, io: std.Io) f64 {
        return @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / 1e9;
    }

    /// A natural-language prompt of exactly `count` tokens, BOS first (random
    /// ids would do for speed, but real text keeps greedy decoding from degenerating).
    fn buildPrompt(allocator: std.mem.Allocator, tokenizer: *const mod.Tokenizer, count: usize) ![]u32 {
        const paragraph = "The history of science is the study of the development of science, " ++
            "including both the natural and social sciences. Science is a body of " ++
            "empirical, theoretical, and practical knowledge about the natural world. ";
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(allocator);
        for (0..count / 10 + 1) |_| try text.appendSlice(allocator, paragraph);
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(allocator);
        try tokens.append(allocator, try tokenizer.bos());
        try tokenizer.encodeAppend(allocator, &tokens, text.items);
        if (tokens.items.len < count) return error.PromptTooShort;
        tokens.shrinkRetainingCapacity(count);
        return tokens.toOwnedSlice(allocator);
    }

    fn field(jw: *std.json.Stringify, name: []const u8, value: anytype) !void {
        try jw.objectField(name);
        try jw.write(value);
    }

    fn peakText(allocator: std.mem.Allocator, value: ?f64, unit: f64, label: []const u8) ![]const u8 {
        const v = value orelse return "unknown (--peak-*)";
        return std.fmt.allocPrint(allocator, "{d:.2} {s}", .{ v / unit, label });
    }

    fn percent(allocator: std.mem.Allocator, value: ?f64) ![]const u8 {
        const v = value orelse return "-";
        return std.fmt.allocPrint(allocator, "{d:.1}", .{v});
    }

    fn either(args: *cli.Args, long: []const u8, short: []const u8) !?[]const u8 {
        return try args.string(long) orelse try args.string(short);
    }

    fn positive(name: []const u8, text: []const u8) !f64 {
        const v = std.fmt.parseFloat(f64, text) catch return invalid(name, text);
        if (!(v > 0)) return invalid(name, text);
        return v;
    }

    fn invalid(name: []const u8, text: []const u8) error{InvalidValue} {
        log.warn("--{s}: '{s}' is not valid", .{ name, text });
        return error.InvalidValue;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "infer-bench parses infer_bench.py's flags and rejects bad values" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "-i", "sft", "--prompt-tokens", "64", "--decode-tokens", "16", "--batch-sizes", "1,4", "--peak-bandwidth", "546e9" });
    defer args.deinit();
    const s = try InferBench.parse(&args);
    try std.testing.expectEqual(mod.CheckpointKind.sft, s.source);
    try std.testing.expectEqual(@as(usize, 64), s.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 16), s.decode_tokens);
    try std.testing.expectEqualStrings("1,4", s.batch_sizes);
    try std.testing.expectEqual(@as(?f64, 546e9), s.peak_bandwidth);
    try std.testing.expectEqual(@as(?f64, null), s.peak_flops);

    var bad = try cli.Args.init(std.testing.allocator, &.{ "--peak-flops", "-3" });
    defer bad.deinit();
    try std.testing.expectError(error.InvalidValue, InferBench.parse(&bad));
}
