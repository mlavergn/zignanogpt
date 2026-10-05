const std = @import("std");
const log = std.log.scoped(.zignanogpt_chat_eval);
const mod = @import("module.zig");

/// The tasks `chat_eval.py` runs by default, in order (HumanEval is out of
/// scope: it executes generated Python).
pub const chat_eval_tasks = [_]mod.TaskKind{ .arc_easy, .arc_challenge, .mmlu, .gsm8k };

/// One task's result.
pub const EvalResult = struct {
    kind: mod.TaskKind,
    passed: usize,
    total: usize,

    pub fn accuracy(self: EvalResult) f64 {
        if (self.total == 0) return 0;
        return @as(f64, @floatFromInt(self.passed)) / @as(f64, @floatFromInt(self.total));
    }
};

/// `chat_eval.py`'s sampling settings for generative tasks.
pub const GenerativeOptions = struct {
    num_samples: usize = 1,
    max_new_tokens: usize = 512,
    temperature: f32 = 0,
    top_k: ?usize = 50,
};

/// nanochat's chat evaluation (`scripts/chat_eval.py`). Categorical tasks
/// score the logits of the answer letters at the last prompt position (one
/// cached prefill per problem: the same logits as the padded batch Python
/// runs). Generative tasks sample completions with `Engine` and pass when any
/// sample is correct.
pub const ChatEval = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    model: *mod.Gpt,
    tokenizer: *const mod.Tokenizer,
    /// Receives `\r` progress lines, or null.
    progress: ?*std.Io.Writer = null,
    /// Checked between problems (training's ChatCORE): a stop request ends the
    /// task with `error.Stopped`.
    observer: ?mod.TrainObserver = null,

    pub fn init(allocator: std.mem.Allocator, model: *mod.Gpt, tokenizer: *const mod.Tokenizer) Self {
        return .{ .allocator = allocator, .model = model, .tokenizer = tokenizer };
    }

    /// Runs one task the way `run_chat_eval` does.
    ///
    /// Parameters:
    /// - `self`: the evaluator.
    /// - `task`: the task (its test split).
    /// - `options`: generative sampling settings.
    /// - `max_problems`: evaluate the first N only, or null for all.
    ///
    /// Return: passes and total; model and task errors.
    pub fn run(self: *Self, task: *const mod.Task, options: GenerativeOptions, max_problems: ?usize) !EvalResult {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const n = if (max_problems) |m| @min(m, task.len()) else task.len();
        return switch (task.evalType()) {
            .categorical => self.categorical(task, n),
            .generative => self.generative(task, n, options),
        };
    }

    /// `run_categorical_eval`: the argmax over the answer letters' logits.
    fn categorical(self: *Self, task: *const mod.Task, n: usize) !EvalResult {
        var result = EvalResult{ .kind = task.kind, .passed = 0, .total = 0 };
        const vocab = self.model.config.vocab_size;
        const logits = try self.allocator.alloc(f32, vocab);
        defer self.allocator.free(logits);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        for (0..n) |i| {
            if (mod.TrainObserver.stopRequested(self.observer)) return error.Stopped;
            _ = arena.reset(.retain_capacity);
            const a = arena.allocator();
            const conv = try task.conversation(a, i);
            const prompt = try self.tokenizer.renderForCompletion(a, conv);
            try self.lastLogits(prompt, logits);
            const letters = try task.letters(a, i);
            var best: ?usize = null;
            var best_logit: f32 = 0;
            for (letters, 0..) |letter, li| {
                const ids = try self.tokenizer.encode(a, letter);
                if (ids.len != 1) {
                    log.warn("answer letter '{s}' is not a single token", .{letter});
                    return error.LetterNotOneToken;
                }
                // torch's argmax keeps the first maximum
                if (best == null or logits[ids[0]] > best_logit) {
                    best = li;
                    best_logit = logits[ids[0]];
                }
            }
            if (try task.evaluate(a, i, letters[best.?])) result.passed += 1;
            result.total += 1;
            try self.report(result);
        }
        return result;
    }

    /// `run_generative_eval`: sample, decode, pass when any sample is correct.
    fn generative(self: *Self, task: *const mod.Task, n: usize, options: GenerativeOptions) !EvalResult {
        var result = EvalResult{ .kind = task.kind, .passed = 0, .total = 0 };
        const engine = mod.Engine.init(self.model, self.tokenizer);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        for (0..n) |i| {
            if (mod.TrainObserver.stopRequested(self.observer)) return error.Stopped;
            _ = arena.reset(.retain_capacity);
            const a = arena.allocator();
            const conv = try task.conversation(a, i);
            const prompt = try self.tokenizer.renderForCompletion(a, conv);
            var batch = try engine.generateBatch(self.allocator, prompt, .{
                .num_samples = options.num_samples,
                .max_tokens = options.max_new_tokens,
                .temperature = options.temperature,
                .top_k = options.top_k,
            });
            defer batch.deinit();
            var passed = false;
            for (batch.results) |tokens| {
                const completion = try self.tokenizer.decode(a, tokens[prompt.len..]);
                if (try task.evaluate(a, i, completion)) passed = true;
            }
            if (passed) result.passed += 1;
            result.total += 1;
            try self.report(result);
        }
        return result;
    }

    /// The logits after the last prompt token.
    fn lastLogits(self: *Self, prompt: []const u32, out: []f32) !void {
        const be = self.model.backend;
        var cache = try mod.KvCache.init(self.allocator, be, self.model.config, 1, prompt.len);
        defer cache.deinit();
        var bufs = try mod.InferenceBuffers.init(self.allocator, be, self.model.config, 1, prompt.len);
        defer bufs.deinit();
        const idx = try be.alloc(.i32, &.{ 1, prompt.len });
        defer be.free(idx);
        const ids = try self.allocator.alloc(i32, prompt.len);
        defer self.allocator.free(ids);
        for (prompt, ids) |t, *d| d.* = @intCast(t);
        try be.upload(idx, i32, ids);
        try self.model.forwardStep(&cache, &bufs, idx);
        try be.download(bufs.logits, f32, out);
    }

    fn report(self: *Self, r: EvalResult) !void {
        const w = self.progress orelse return;
        try w.print("\r\x1b[K{s} | {d}/{d} ({d:.2}%)", .{ r.kind.name(), r.passed, r.total, 100 * r.accuracy() });
        try w.flush();
    }

    /// ChatCORE: the mean accuracy centered on each task's random baseline
    /// (0.25 for 4-way multiple choice, 0 for generative), so 0 is chance and 1 perfect.
    ///
    /// Parameters:
    /// - `results`: per-task results.
    ///
    /// Return: the centered mean.
    pub fn chatCore(results: []const EvalResult) f64 {
        if (results.len == 0) return 0;
        var sum: f64 = 0;
        for (results) |r| {
            const baseline: f64 = switch (r.kind) {
                .mmlu, .arc_easy, .arc_challenge => 0.25,
                .gsm8k, .smoltalk => 0,
            };
            sum += (r.accuracy() - baseline) / (1 - baseline);
        }
        return sum / @as(f64, @floatFromInt(results.len));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "chat eval matches chat_eval.py on the task slices" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const meta_bytes = try storage.read(root ++ "/nanochat_base/base_checkpoints/d2/meta_000005.json");
    defer allocator.free(meta_bytes);
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), meta_bytes, .{});
    var model = try mod.Gpt.init(allocator, &backend, try mod.GptConfig.fromJson(arena.allocator(), meta.object.get("model_config").?));
    defer model.deinit();
    var state = try mod.TorchImport.loadStateDict(allocator, std.testing.io, root ++ "/nanochat_base/base_checkpoints/d2/model_000005.pt");
    defer state.deinit();
    try mod.TorchImport.loadWeights(&backend, &state, &model.weights);

    const bytes = try storage.read(root ++ "/chat_eval.json");
    defer allocator.free(bytes);
    const Expected = struct { MMLU: f64, @"ARC-Easy": f64, GSM8K: f64, completions: []const []const u8 };
    var want = try std.json.parseFromSlice(Expected, allocator, bytes, .{});
    defer want.deinit();

    const config = mod.Config{ .allocator = allocator, .base_dir = root ++ "/task_base", .nanochat_dir = "/nonexistent-nanochat", .data_url = "http://unused" };
    var mmlu = try mod.Task.open(allocator, std.testing.io, &config, .mmlu, "test", null);
    defer mmlu.deinit();
    var arc = try mod.Task.open(allocator, std.testing.io, &config, .arc_easy, "test", null);
    defer arc.deinit();
    var gsm = try mod.Task.open(allocator, std.testing.io, &config, .gsm8k, "test", null);
    defer gsm.deinit();

    var eval = ChatEval.init(allocator, &model, &tok);
    const opts = GenerativeOptions{ .max_new_tokens = 20 };
    try std.testing.expectEqual(want.value.MMLU, (try eval.run(&mmlu, opts, null)).accuracy());
    try std.testing.expectEqual(want.value.@"ARC-Easy", (try eval.run(&arc, opts, 20)).accuracy());
    const g = try eval.run(&gsm, opts, 6);
    try std.testing.expectEqual(want.value.GSM8K, g.accuracy());
    try std.testing.expectEqual(@as(usize, 6), g.total);

    // The greedy completions themselves (Python decodes invalid UTF-8 as U+FFFD).
    const engine = mod.Engine.init(&model, &tok);
    for (want.value.completions, 0..) |w, i| {
        const a = arena.allocator();
        const prompt = try tok.renderForCompletion(a, try gsm.conversation(a, i));
        var batch = try engine.generateBatch(allocator, prompt, .{ .max_tokens = 20, .temperature = 0, .top_k = 50 });
        defer batch.deinit();
        const text = try tok.decode(a, batch.results[0][prompt.len..]);
        const replaced = try std.fmt.allocPrint(a, "{f}", .{std.unicode.fmtUtf8(text)});
        try std.testing.expectEqualStrings(w, replaced);
    }
    try std.testing.expectApproxEqAbs(@as(f64, (0.3 - 0.25) / 0.75), ChatEval.chatCore(&.{.{ .kind = .mmlu, .passed = 3, .total = 10 }}), 1e-12);
}
