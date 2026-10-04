const std = @import("std");
const log = std.log.scoped(.zignanogpt_core_eval);
const mod = @import("module.zig");

/// One example's model inputs: BOS-prefixed token sequences and, per
/// sequence, the span `[start, end)` whose tokens are scored.
pub const CorePrepared = struct {
    tokens: []const []const u32,
    starts: []const usize,
    ends: []const usize,
};

/// nanochat's CORE evaluation (`nanochat/core_eval.py`, DCLM's in-context
/// tasks). Prompts are rendered as its Jinja templates render them; multiple
/// choice and schema examples pick the option with the lowest mean loss over
/// its scored span, language modeling examples need every span token to be
/// the argmax prediction.
pub const CoreEval = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    model: *mod.Gpt,
    tokenizer: *const mod.Tokenizer,

    pub fn init(allocator: std.mem.Allocator, model: *mod.Gpt, tokenizer: *const mod.Tokenizer) Self {
        return .{ .allocator = allocator, .model = model, .tokenizer = tokenizer };
    }

    /// `evaluate_task` after `evaluate_core`'s shuffle (`random.Random(1337)`)
    /// and `max_per_task` cut.
    ///
    /// Parameters:
    /// - `self`: the evaluator.
    /// - `task`: the task.
    /// - `examples`: its examples in file order.
    /// - `max_per_task`: keep the first N after shuffling, or null for all.
    ///
    /// Return: the accuracy; model and format errors.
    pub fn evaluateTask(self: *Self, task: mod.CoreTask, examples: []const std.json.Value, max_per_task: ?usize) !f64 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const data = try self.allocator.dupe(std.json.Value, examples);
        defer self.allocator.free(data);
        var rng = mod.PythonRandom.init(1337);
        rng.shuffle(std.json.Value, data);
        const n = if (max_per_task) |m| @min(m, data.len) else data.len;
        if (n == 0) return 0;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var correct: usize = 0;
        for (0..n) |i| {
            _ = arena.reset(.retain_capacity);
            if (try self.evaluateExample(arena.allocator(), task, data[0..n], i)) correct += 1;
        }
        return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(n));
    }

    /// `evaluate_example`: renders, tokenizes and scores one example.
    ///
    /// Parameters:
    /// - `self`: the evaluator.
    /// - `arena`: scratch.
    /// - `task`: the task.
    /// - `data`: the (shuffled, cut) examples; few-shot examples come from them.
    /// - `idx`: the example.
    ///
    /// Return: whether the model got it right; model and format errors.
    pub fn evaluateExample(self: *Self, arena: std.mem.Allocator, task: mod.CoreTask, data: []const std.json.Value, idx: usize) !bool {
        const prepared = try self.prepare(arena, task, data, idx);
        const b = prepared.tokens.len;
        var t: usize = 0;
        for (prepared.tokens) |seq| t = @max(t, seq.len);
        const be = self.model.backend;
        const config = self.model.config;
        var acts = try mod.GptActivations.init(self.allocator, be, config, b, t);
        defer acts.deinit();
        const bos = try self.tokenizer.bos();
        // Padded with BOS on the right; targets are the inputs rolled left by one.
        const ids = try arena.alloc(i32, b * t);
        const targets = try arena.alloc(i32, b * t);
        for (prepared.tokens, 0..) |seq, r| {
            const row = ids[r * t ..][0..t];
            for (row, 0..) |*d, i| d.* = @intCast(if (i < seq.len) seq[i] else bos);
            for (targets[r * t ..][0..t], 0..) |*d, i| d.* = row[(i + 1) % t];
        }
        const idx_t = try be.alloc(.i32, &.{ b, t });
        defer be.free(idx_t);
        const target_t = try be.alloc(.i32, &.{b * t});
        defer be.free(target_t);
        const losses_t = try be.alloc(.f32, &.{b * t});
        defer be.free(losses_t);
        try be.upload(idx_t, i32, ids);
        try be.upload(target_t, i32, targets);
        try self.model.forward(&acts, idx_t);
        const logits = try acts.logits.reshape(&.{ b * t, config.vocab_size });

        switch (task.task_type) {
            .language_modeling => {
                // Every continuation token must be the argmax of the position before it.
                const s = prepared.starts[0];
                const e = prepared.ends[0];
                const rows = try arena.alloc(f32, (e - s) * config.vocab_size);
                try be.download(try logits.rows(s - 1, e - s), f32, rows);
                for (0..e - s) |k| {
                    const predicted = std.mem.indexOfMax(f32, rows[k * config.vocab_size ..][0..config.vocab_size]);
                    if (predicted != prepared.tokens[0][s + k]) return false;
                }
                return true;
            },
            .multiple_choice, .schema => {
                try be.crossEntropyRows(losses_t, logits, target_t);
                const losses = try arena.alloc(f32, b * t);
                try be.download(losses_t, f32, losses);
                var best: usize = 0;
                var best_loss: f64 = std.math.inf(f64);
                for (prepared.starts, prepared.ends, 0..) |s, e, r| {
                    var sum: f64 = 0;
                    for (losses[r * t + s - 1 .. r * t + e - 1]) |l| sum += l;
                    const mean = sum / @as(f64, @floatFromInt(e - s));
                    if (mean < best_loss) {
                        best_loss = mean;
                        best = r;
                    }
                }
                const gold = try intField(data[idx], "gold");
                return best == gold;
            },
        }
    }

    /// Renders and tokenizes an example (`render_prompts_*` + `batch_sequences_*`).
    ///
    /// Parameters:
    /// - `self`: the evaluator.
    /// - `arena`: holds the result.
    /// - `task`: the task.
    /// - `data`: the examples few-shot examples are drawn from.
    /// - `idx`: the example.
    ///
    /// Return: the sequences and spans; `error.InvalidCoreExample`.
    pub fn prepare(self: *Self, arena: std.mem.Allocator, task: mod.CoreTask, data: []const std.json.Value, idx: usize) !CorePrepared {
        const prompts = try renderPrompts(arena, task, data, idx);
        const bos = try self.tokenizer.bos();
        const tokens = try arena.alloc([]const u32, prompts.len);
        for (prompts, tokens) |p, *seq| {
            var ids: std.ArrayList(u32) = .empty;
            try ids.append(arena, bos);
            try self.tokenizer.encodeAppend(arena, &ids, p);
            seq.* = ids.items;
        }
        switch (task.task_type) {
            .multiple_choice => {
                // Same context, different continuations: score after the common prefix.
                const start = commonLength(tokens, .left);
                const starts = try arena.alloc(usize, tokens.len);
                const ends = try arena.alloc(usize, tokens.len);
                for (tokens, starts, ends) |seq, *s, *e| {
                    s.* = start;
                    e.* = seq.len;
                }
                return .{ .tokens = tokens, .starts = starts, .ends = ends };
            },
            .schema => {
                // Different contexts, same continuation: score the common suffix.
                const suffix = commonLength(tokens, .right);
                const starts = try arena.alloc(usize, tokens.len);
                const ends = try arena.alloc(usize, tokens.len);
                for (tokens, starts, ends) |seq, *s, *e| {
                    e.* = seq.len;
                    s.* = seq.len - suffix;
                }
                return .{ .tokens = tokens, .starts = starts, .ends = ends };
            },
            .language_modeling => {
                const without = tokens[0];
                const with = tokens[1];
                if (without.len >= with.len or !std.mem.eql(u32, without, with[0..without.len])) {
                    log.warn("{s} example {d}: the prompt is not a token prefix of prompt + continuation", .{ task.label, idx });
                    return error.InvalidCoreExample;
                }
                return .{ .tokens = tokens[1..], .starts = try arena.dupe(usize, &.{without.len}), .ends = try arena.dupe(usize, &.{with.len}) };
            },
        }
    }

    /// The prompts of an example: one per choice (multiple choice), per
    /// context option (schema), or without and with the continuation (LM).
    pub fn renderPrompts(arena: std.mem.Allocator, task: mod.CoreTask, data: []const std.json.Value, idx: usize) ![]const []const u8 {
        // Few-shot examples: random.Random(1234 + idx).sample of the others.
        var shots: std.ArrayList(std.json.Value) = .empty;
        if (task.num_fewshot > 0) {
            var rng = mod.PythonRandom.init(1234 + idx);
            const picks = try rng.sample(arena, data.len - 1, task.num_fewshot);
            for (picks) |p| try shots.append(arena, data[if (p < idx) p else p + 1]);
        }
        const d = task.continuation_delimiter;
        const item = data[idx];
        var prefix: std.ArrayList(u8) = .empty;
        for (shots.items) |ex| {
            switch (task.task_type) {
                .multiple_choice => {
                    const choices = try arrayField(ex, "choices");
                    try prefix.appendSlice(arena, try pyStr(arena, try field(ex, "query")));
                    try prefix.appendSlice(arena, d);
                    try prefix.appendSlice(arena, try pyStr(arena, try at(choices, try intField(ex, "gold"))));
                },
                .schema => {
                    const options = try arrayField(ex, "context_options");
                    try prefix.appendSlice(arena, try pyStr(arena, try at(options, try intField(ex, "gold"))));
                    try prefix.appendSlice(arena, d);
                    try prefix.appendSlice(arena, try pyStr(arena, try field(ex, "continuation")));
                },
                .language_modeling => {
                    try prefix.appendSlice(arena, pyStrip(try pyStr(arena, try field(ex, "context"))));
                    try prefix.appendSlice(arena, d);
                    try prefix.appendSlice(arena, try pyStr(arena, try field(ex, "continuation")));
                },
            }
            try prefix.appendSlice(arena, "\n\n");
        }
        var prompts: std.ArrayList([]const u8) = .empty;
        switch (task.task_type) {
            .multiple_choice => for ((try arrayField(item, "choices")).items) |choice| {
                try prompts.append(arena, try std.mem.concat(arena, u8, &.{ prefix.items, try pyStr(arena, try field(item, "query")), d, try pyStr(arena, choice) }));
            },
            .schema => for ((try arrayField(item, "context_options")).items) |option| {
                try prompts.append(arena, try std.mem.concat(arena, u8, &.{ prefix.items, try pyStr(arena, option), d, try pyStr(arena, try field(item, "continuation")) }));
            },
            .language_modeling => {
                const context = pyStrip(try pyStr(arena, try field(item, "context")));
                const without = try std.mem.concat(arena, u8, &.{ prefix.items, context, d });
                try prompts.append(arena, pyStrip(without));
                try prompts.append(arena, try std.mem.concat(arena, u8, &.{ without, try pyStr(arena, try field(item, "continuation")) }));
            },
        }
        return prompts.items;
    }

    /// `find_common_length`: the shared prefix (`left`) or suffix (`right`) length.
    fn commonLength(seqs: []const []const u32, direction: enum { left, right }) usize {
        var min_len: usize = std.math.maxInt(usize);
        for (seqs) |s| min_len = @min(min_len, s.len);
        for (0..min_len) |i| {
            const pos0 = if (direction == .left) i else seqs[0].len - 1 - i;
            const token = seqs[0][pos0];
            for (seqs) |s| {
                const pos = if (direction == .left) i else s.len - 1 - i;
                if (s[pos] != token) return i;
            }
        }
        return min_len;
    }

    /// Python's `str.strip()`: drops leading and trailing `str.isspace` characters.
    pub fn pyStrip(text: []const u8) []const u8 {
        var start: usize = 0;
        while (start < text.len) {
            const n = std.unicode.utf8ByteSequenceLength(text[start]) catch break;
            if (start + n > text.len) break;
            const cp = std.unicode.utf8Decode(text[start..][0..n]) catch break;
            if (!isSpace(cp)) break;
            start += n;
        }
        var end = text.len;
        while (end > start) {
            var begin = end - 1;
            while (begin > start and text[begin] & 0xC0 == 0x80) begin -= 1;
            const cp = std.unicode.utf8Decode(text[begin..end]) catch break;
            if (!isSpace(cp)) break;
            end = begin;
        }
        return text[start..end];
    }

    /// `str.isspace` for one code point (Unicode whitespace plus the ASCII separators).
    fn isSpace(cp: u21) bool {
        return switch (cp) {
            0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
            else => false,
        };
    }

    /// Jinja's `{{ value }}`: `str(value)` for the strings and ints the data holds.
    fn pyStr(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
        return switch (value) {
            .string => |s| s,
            .number_string => |s| s,
            .integer => |i| std.fmt.allocPrint(arena, "{d}", .{i}),
            else => {
                log.warn("unsupported CORE field value: {t}", .{value});
                return error.InvalidCoreExample;
            },
        };
    }

    fn field(value: std.json.Value, name: []const u8) !std.json.Value {
        if (value != .object) return error.InvalidCoreExample;
        return value.object.get(name) orelse {
            log.warn("CORE example without '{s}'", .{name});
            return error.InvalidCoreExample;
        };
    }

    fn arrayField(value: std.json.Value, name: []const u8) !std.json.Array {
        const v = try field(value, name);
        if (v != .array) return error.InvalidCoreExample;
        return v.array;
    }

    fn intField(value: std.json.Value, name: []const u8) !usize {
        const v = try field(value, name);
        if (v != .integer or v.integer < 0) return error.InvalidCoreExample;
        return @intCast(v.integer);
    }

    fn at(array: std.json.Array, i: usize) !std.json.Value {
        if (i >= array.items.len) return error.InvalidCoreExample;
        return array.items[i];
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "core eval reproduces nanochat's prompts, spans and scores" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    try mod.EvalBundle.extract(allocator, std.testing.io, root ++ "/eval_bundle.zip", base);
    const config = mod.Config{ .allocator = allocator, .base_dir = base, .nanochat_dir = "/nonexistent-nanochat", .data_url = "http://unused" };
    var bundle = try mod.EvalBundle.open(allocator, std.testing.io, &config, null);
    defer bundle.deinit();

    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const meta_bytes = try storage.read(root ++ "/nanochat_base/base_checkpoints/d2/meta_000005.json");
    defer allocator.free(meta_bytes);
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, meta_bytes, .{});
    var model = try mod.Gpt.init(allocator, &backend, try mod.GptConfig.fromJson(a, meta.object.get("model_config").?));
    defer model.deinit();
    var state = try mod.TorchImport.loadStateDict(allocator, std.testing.io, root ++ "/nanochat_base/base_checkpoints/d2/model_000005.pt");
    defer state.deinit();
    try mod.TorchImport.loadWeights(&backend, &state, &model.weights);

    const bytes = try storage.read(root ++ "/core.json");
    defer allocator.free(bytes);
    const Row = struct { prompts: []const []const u8, tokens: []const []const u32, starts: []const usize, ends: []const usize, correct: bool };
    const Expected = struct {
        results: std.json.ArrayHashMap(f64),
        centered: std.json.ArrayHashMap(f64),
        core: f64,
        examples: std.json.ArrayHashMap([]const Row),
        max_per_task: usize,
    };
    var want = try std.json.parseFromSlice(Expected, allocator, bytes, .{});
    defer want.deinit();

    var eval = CoreEval.init(allocator, &model, &tok);
    var centered_sum: f64 = 0;
    for (bundle.tasks) |task| {
        const examples = try bundle.readExamples(a, storage, task);

        // Prompts, tokens and spans, example by example.
        const data = try a.dupe(std.json.Value, examples);
        var rng = mod.PythonRandom.init(1337);
        rng.shuffle(std.json.Value, data);
        const cut = data[0..@min(want.value.max_per_task, data.len)];
        for (want.value.examples.map.get(task.label).?, 0..) |row, i| {
            const prompts = try CoreEval.renderPrompts(a, task, cut, i);
            try std.testing.expectEqual(row.prompts.len, prompts.len);
            for (row.prompts, prompts) |w, g| try std.testing.expectEqualStrings(w, g);
            const p = try eval.prepare(a, task, cut, i);
            try std.testing.expectEqual(row.tokens.len, p.tokens.len);
            for (row.tokens, p.tokens) |w, g| try std.testing.expectEqualSlices(u32, w, g);
            try std.testing.expectEqualSlices(usize, row.starts, p.starts);
            try std.testing.expectEqualSlices(usize, row.ends, p.ends);
            const got = try eval.evaluateExample(a, task, cut, i);
            if (got != row.correct) {
                std.debug.print("{s} example {d}: python {}, zig {}\n", .{ task.label, i, row.correct, got });
                return error.TestUnexpectedResult;
            }
        }
        const accuracy = try eval.evaluateTask(task, examples, want.value.max_per_task);
        try std.testing.expectApproxEqAbs(want.value.results.map.get(task.label).?, accuracy, 1e-6);
        const centered = (accuracy - 0.01 * task.random_baseline) / (1.0 - 0.01 * task.random_baseline);
        try std.testing.expectApproxEqAbs(want.value.centered.map.get(task.label).?, centered, 1e-6);
        centered_sum += centered;
    }
    try std.testing.expectApproxEqAbs(want.value.core, centered_sum / @as(f64, @floatFromInt(bundle.tasks.len)), 1e-6);
    try std.testing.expectEqualStrings("a b", CoreEval.pyStrip("\u{a0} a b\u{3000}\n"));
}
