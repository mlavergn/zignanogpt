const std = @import("std");
const log = std.log.scoped(.zignanogpt_base_eval);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt eval`: nanochat's `base_eval.py` for a base model: samples,
/// train/val bits per byte, and the CORE metric (eval bundle downloaded on
/// first use; results also written to `<base>/base_eval/base_model_<step>.csv`).
pub const BaseEval = struct {
    pub const usage =
        \\usage: zignanogpt eval [--eval core,bpb,sample] [--model-tag <tag>] [--step <n>] [--max-per-task <n>]
        \\                       [--device-batch-size <n>] [--split-tokens <n>] [--dataset <name>] [--threads <n>]
        \\  --eval               any of core, bpb, sample (default: all three)
        \\  --max-per-task       CORE examples per task (default: all; runcpu.sh: 16)
        \\  --device-batch-size  bpb batch (default 32; runcpu.sh: 1)
        \\  --split-tokens       bpb tokens per split (default 20971520; runcpu.sh: 16384)
        \\  --dataset            the shards bpb reads (default climbmix)
        \\
    ;

    const prompts = [_][]const u8{
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    };

    const Options = struct {
        eval: []const u8 = "core,bpb,sample",
        model_tag: ?[]const u8 = null,
        step: ?usize = null,
        max_per_task: usize = 0,
        device_batch_size: usize = 32,
        split_tokens: usize = 40 * 524288,
        dataset: []const u8 = mod.Dataset.default_name,
        threads: usize = 0,
    };

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress and results.
    ///
    /// Return: the exit code; loading and evaluation errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (args.flag("help")) {
            try out.writeAll(usage);
            return 0;
        }
        var o: Options = .{};
        args.fill(Options, &o) catch |err| {
            log.debug("eval usage error [{t}]", .{err});
            try out.writeAll(usage);
            return 2;
        };
        try args.finish();
        var modes = struct { core: bool = false, bpb: bool = false, sample: bool = false }{};
        var it = std.mem.splitScalar(u8, o.eval, ',');
        while (it.next()) |raw| {
            const m = std.mem.trim(u8, raw, " ");
            if (std.mem.eql(u8, m, "core")) modes.core = true else if (std.mem.eql(u8, m, "bpb")) modes.bpb = true else if (std.mem.eql(u8, m, "sample")) modes.sample = true else {
                try out.print("invalid eval mode: {s} (core, bpb, sample)\n", .{m});
                return 2;
            }
        }

        const allocator = init.gpa;
        const storage = mod.Storage.init(allocator, init.io);
        var config = try mod.Config.load(allocator, init.environ_map, storage);
        defer config.deinit();
        var backend = try mod.Backend.init(allocator, init.io, .{ .threads = o.threads });
        defer backend.deinit();
        const loaded = try allocator.create(mod.LoadedModel);
        defer allocator.destroy(loaded);
        try loaded.init(allocator, &backend, storage, config.base_dir, .{ .kind = .base, .tag = o.model_tag, .step = o.step });
        defer loaded.deinit();
        try out.print("Evaluating model: base_model (step {d})\n", .{loaded.step});
        try out.flush();

        if (modes.sample) try sample(allocator, loaded, out, observer);
        if (modes.bpb) try bpb(allocator, init.io, &config, loaded, o, out, observer);
        if (modes.core) try core(allocator, init.io, &config, loaded, o, out, observer);
        return 0;
    }

    fn banner(out: *std.Io.Writer, title: []const u8) !void {
        try out.print("\n{s}\n{s}\n{s}\n", .{ &@as([80]u8, @splat('=')), title, &@as([80]u8, @splat('=')) });
    }

    /// Greedy completions of the prompts, then 8 unconditioned samples at temperature 1.
    fn sample(allocator: std.mem.Allocator, loaded: *mod.LoadedModel, out: *std.Io.Writer, observer: ?mod.TrainObserver) !void {
        try banner(out, "Model Samples");
        const tok = &loaded.tokenizer;
        const engine = mod.Engine.init(&loaded.model, tok);
        try out.writeAll("\nConditioned samples:\n");
        for (prompts, 0..) |p, i| {
            mod.TrainObserver.progress(observer, "samples", i, prompts.len);
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(allocator);
            try ids.append(allocator, try tok.bos());
            try tok.encodeAppend(allocator, &ids, p);
            var batch = try engine.generateBatch(allocator, ids.items, .{ .max_tokens = 16, .temperature = 0 });
            defer batch.deinit();
            const text = try tok.decode(allocator, batch.results[0]);
            defer allocator.free(text);
            try out.print("{s}\n{s}\n", .{ &@as([80]u8, @splat('-')), text });
            try out.flush();
        }
        try out.writeAll("\nUnconditioned samples:\n");
        var batch = try engine.generateBatch(allocator, &.{try tok.bos()}, .{ .num_samples = 8, .max_tokens = 128, .temperature = 1.0 });
        defer batch.deinit();
        for (batch.results) |r| {
            const text = try tok.decode(allocator, r);
            defer allocator.free(text);
            try out.print("{s}\n{s}\n", .{ &@as([80]u8, @splat('-')), text });
        }
        try out.flush();
    }

    /// Bits per byte on the train and val shards.
    fn bpb(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, loaded: *mod.LoadedModel, o: Options, out: *std.Io.Writer, observer: ?mod.TrainObserver) !void {
        try banner(out, "BPB Evaluation");
        const seq = loaded.model.config.sequence_len;
        const b = o.device_batch_size;
        const per_step = b * seq;
        var split_tokens = o.split_tokens;
        if (split_tokens % per_step != 0) {
            split_tokens = split_tokens / per_step * per_step;
            try out.print("Adjusted split_tokens to {d} (must be divisible by {d})\n", .{ split_tokens, per_step });
        }
        const steps = split_tokens / per_step;
        var dataset = try mod.Dataset.init(allocator, io, config, o.dataset);
        defer dataset.deinit();
        const shards = try dataset.list(allocator);
        defer {
            for (shards) |p| allocator.free(p);
            allocator.free(shards);
        }
        if (shards.len < 2) {
            try out.writeAll("bpb needs at least one train shard and the val shard; run `zignanogpt download`\n");
            return;
        }
        const token_bytes = try loaded.tokenizer.tokenByteCounts(allocator);
        defer allocator.free(token_bytes);
        const be = loaded.model.backend;
        var acts = try mod.GptActivations.init(allocator, be, loaded.model.config, b, seq);
        defer acts.deinit();
        const inputs = try allocator.alloc(i32, per_step);
        defer allocator.free(inputs);
        const targets = try allocator.alloc(i32, per_step);
        defer allocator.free(targets);
        const losses = try allocator.alloc(f32, per_step);
        defer allocator.free(losses);
        const idx = try be.alloc(.i32, &.{ b, seq });
        defer be.free(idx);
        const target_ids = try be.alloc(.i32, &.{per_step});
        defer be.free(target_ids);
        const row_losses = try be.alloc(.f32, &.{per_step});
        defer be.free(row_losses);
        const logits = try acts.logits.reshape(&.{ per_step, loaded.model.config.vocab_size });
        for ([_][]const u8{ "train", "val" }) |split| {
            const files: []const []const u8 = if (std.mem.eql(u8, split, "train")) shards[0 .. shards.len - 1] else shards[shards.len - 1 ..];
            var stream = try mod.DocumentStream.init(allocator, io, .{ .parquet = files }, 128, null);
            defer stream.deinit();
            var loader = try mod.DataLoader.init(allocator, &loaded.tokenizer, &stream, b, seq, 1000);
            defer loader.deinit();
            var nats: f64 = 0;
            var bytes: u64 = 0;
            const label = if (std.mem.eql(u8, split, "train")) "bpb train" else "bpb val";
            for (0..steps) |i| {
                mod.TrainObserver.progress(observer, label, i, steps);
                _ = try loader.next(inputs, targets);
                try be.upload(idx, i32, inputs);
                try be.upload(target_ids, i32, targets);
                try loaded.model.forward(&acts, idx);
                try be.crossEntropyRows(row_losses, logits, target_ids);
                try be.download(row_losses, f32, losses);
                for (losses, targets) |l, t| {
                    if (t < 0) continue;
                    const n = token_bytes[@intCast(t)];
                    if (n > 0) nats += l;
                    bytes += @intCast(n);
                }
            }
            const value = if (bytes == 0) std.math.inf(f64) else nats / (@log(2.0) * @as(f64, @floatFromInt(bytes)));
            try out.print("{s} bpb: {d:.6}\n", .{ split, value });
            try out.flush();
        }
    }

    /// The CORE metric over the eval bundle's tasks, also written as CSV.
    fn core(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, loaded: *mod.LoadedModel, o: Options, out: *std.Io.Writer, observer: ?mod.TrainObserver) !void {
        try banner(out, "CORE Evaluation");
        try out.flush();
        const storage = mod.Storage.init(allocator, io);
        var bundle = try mod.EvalBundle.open(allocator, io, config, out);
        defer bundle.deinit();
        var eval = mod.CoreEval.init(allocator, &loaded.model, &loaded.tokenizer);
        eval.observer = observer;
        var csv: std.ArrayList(u8) = .empty;
        defer csv.deinit(allocator);
        try csv.print(allocator, "{s:<35}, {s:<10}, {s:<10}\n", .{ "Task", "Accuracy", "Centered" });
        var sum: f64 = 0;
        for (bundle.tasks) |task| {
            const start = std.Io.Clock.awake.now(io);
            try out.print("Evaluating: {s} ({d}-shot, type: {t})... ", .{ task.label, task.num_fewshot, task.task_type });
            try out.flush();
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const examples = try bundle.readExamples(arena.allocator(), storage, task);
            const accuracy = try eval.evaluateTask(task, examples, if (o.max_per_task == 0) null else o.max_per_task);
            const base = 0.01 * task.random_baseline;
            const centered = (accuracy - base) / (1.0 - base);
            sum += centered;
            const secs = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / 1e9;
            try out.print("accuracy: {d:.4} | centered: {d:.4} | time: {d:.2}s\n", .{ accuracy, centered, secs });
            try out.flush();
            try csv.print(allocator, "{s:<35}, {d:<10.6}, {d:<10.6}\n", .{ task.label, accuracy, centered });
        }
        const metric = sum / @as(f64, @floatFromInt(@max(bundle.tasks.len, 1)));
        try csv.print(allocator, "{s:<35}, {s:<10}, {d:<10.6}\n", .{ "CORE", "", metric });
        const path = try allocator.print("{s}/base_eval/base_model_{d:0>6}.csv", .{ config.base_dir, loaded.step });
        defer allocator.free(path);
        try storage.write(path, csv.items);
        try out.print("\nResults written to: {s}\nCORE metric: {d:.4}\n", .{ path, metric });
        try out.flush();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "eval reads base_eval.py's flags" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "--eval", "core,bpb", "--max-per-task", "16", "--device-batch-size=1", "--split-tokens", "16384" });
    defer args.deinit();
    var o: BaseEval.Options = .{};
    try args.fill(BaseEval.Options, &o);
    try args.finish();
    try std.testing.expectEqualStrings("core,bpb", o.eval);
    try std.testing.expectEqual(@as(usize, 16), o.max_per_task);
    try std.testing.expectEqual(@as(usize, 1), o.device_batch_size);
    try std.testing.expectEqual(@as(usize, 16384), o.split_tokens);
}
