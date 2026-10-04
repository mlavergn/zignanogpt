const std = @import("std");
const log = std.log.scoped(.zignanogpt_engine);
const mod = @import("module.zig");

/// nanochat's `Engine.generate` arguments.
pub const GenerateOptions = struct {
    num_samples: usize = 1,
    /// Tokens per row at most; null runs until the rows end or the cache
    /// (`sequence_len` positions) fills.
    max_tokens: ?usize = null,
    /// 0 is greedy.
    temperature: f32 = 1.0,
    /// Sample among the k most likely tokens; null or 0 for all.
    top_k: ?usize = null,
    seed: u64 = 42,
};

/// `generateBatch`'s result: each row's prompt plus generated tokens (without
/// the terminating token), and per token 1 when sampled, 0 when forced or prompt.
pub const GeneratedBatch = struct {
    allocator: std.mem.Allocator,
    results: [][]u32,
    masks: [][]u8,

    pub fn deinit(self: *GeneratedBatch) void {
        for (self.results) |r| self.allocator.free(r);
        for (self.masks) |m| self.allocator.free(m);
        self.allocator.free(self.results);
        self.allocator.free(self.masks);
    }
};

/// nanochat's inference engine (`nanochat/engine.py`): KV-cached generation
/// with batched samples and the calculator tool.
pub const Engine = struct {
    const Self = @This();

    model: *mod.Gpt,
    tokenizer: *const mod.Tokenizer,

    pub fn init(model: *mod.Gpt, tokenizer: *const mod.Tokenizer) Self {
        return .{ .model = model, .tokenizer = tokenizer };
    }

    /// Starts a streaming generation; call `next` for each token column.
    ///
    /// Parameters:
    /// - `self`: the engine.
    /// - `allocator`: owns the generation.
    /// - `prompt`: the token ids.
    /// - `options`: samples, limits and sampling.
    ///
    /// Return: the generation (`deinit` it); prefill errors.
    pub fn generate(self: Self, allocator: std.mem.Allocator, prompt: []const u32, options: GenerateOptions) !mod.Generation {
        return mod.Generation.init(allocator, self.model, self.tokenizer, prompt, options);
    }

    /// Runs a generation to the end (nanochat's `generate_batch`).
    ///
    /// Parameters:
    /// - `self`: the engine.
    /// - `allocator`: owns the result.
    /// - `prompt`: the token ids.
    /// - `options`: samples, limits and sampling.
    ///
    /// Return: every row's tokens and masks; generation errors.
    pub fn generateBatch(self: Self, allocator: std.mem.Allocator, prompt: []const u32, options: GenerateOptions) !GeneratedBatch {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var gen = try self.generate(allocator, prompt, options);
        defer gen.deinit();
        const b = options.num_samples;
        const results = try allocator.alloc(std.ArrayList(u32), b);
        defer allocator.free(results);
        const masks = try allocator.alloc(std.ArrayList(u8), b);
        defer allocator.free(masks);
        for (results, masks) |*r, *m| {
            r.* = .empty;
            m.* = .empty;
        }
        defer for (results, masks) |*r, *m| {
            r.deinit(allocator);
            m.deinit(allocator);
        };
        for (results, masks) |*r, *m| {
            try r.appendSlice(allocator, prompt);
            try m.appendNTimes(allocator, 0, prompt.len);
        }
        const done = try allocator.alloc(bool, b);
        defer allocator.free(done);
        @memset(done, false);
        while (try gen.next()) |column| {
            for (column.tokens, column.masks, results, masks, done) |token, mask, *r, *m, *d| {
                if (d.*) continue;
                if (token == gen.tools.assistant_end or token == gen.tools.bos) {
                    d.* = true;
                } else {
                    try r.append(allocator, token);
                    try m.append(allocator, mask);
                }
            }
            if (std.mem.allEqual(bool, done, true)) break;
        }
        var out = GeneratedBatch{ .allocator = allocator, .results = try allocator.alloc([]u32, b), .masks = undefined };
        errdefer allocator.free(out.results);
        out.masks = try allocator.alloc([]u8, b);
        errdefer allocator.free(out.masks);
        for (results, masks, out.results, out.masks) |*r, *m, *dr, *dm| {
            dr.* = try r.toOwnedSlice(allocator);
            dm.* = try m.toOwnedSlice(allocator);
        }
        return out;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

/// The d2 model and tokenizer of the import fixture.
const Fixture = struct {
    backend: mod.Backend,
    tokenizer: mod.Tokenizer,
    model: mod.Gpt,
    arena: std.heap.ArenaAllocator,

    fn init(self: *Fixture, allocator: std.mem.Allocator) !void {
        const root = mod.build_options.source_root ++ "/testdata/nanochat_base";
        const storage = mod.Storage.init(allocator, std.testing.io);
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        self.backend = try mod.Backend.init(allocator, std.testing.io, .{});
        errdefer self.backend.deinit();
        self.tokenizer = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/tokenizer/tokenizer.pkl");
        errdefer self.tokenizer.deinit();
        const meta_bytes = try storage.read(root ++ "/base_checkpoints/d2/meta_000005.json");
        defer allocator.free(meta_bytes);
        const meta = try std.json.parseFromSliceLeaky(std.json.Value, self.arena.allocator(), meta_bytes, .{});
        const config = try mod.GptConfig.fromJson(self.arena.allocator(), meta.object.get("model_config").?);
        self.model = try mod.Gpt.init(allocator, &self.backend, config);
        errdefer self.model.deinit();
        var state = try mod.TorchImport.loadStateDict(allocator, std.testing.io, root ++ "/base_checkpoints/d2/model_000005.pt");
        defer state.deinit();
        try mod.TorchImport.loadWeights(&self.backend, &state, &self.model.weights);
    }

    fn deinit(self: *Fixture) void {
        self.model.deinit();
        self.tokenizer.deinit();
        self.backend.deinit();
        self.arena.deinit();
    }
};

test "engine greedy generations match nanochat's engine" {
    const allocator = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.init(allocator);
    defer fx.deinit();
    const engine = Engine.init(&fx.model, &fx.tokenizer);

    const storage = mod.Storage.init(allocator, std.testing.io);
    const bytes = try storage.read(mod.build_options.source_root ++ "/testdata/engine.json");
    defer allocator.free(bytes);
    const Expected = struct {
        calculator: []const [2]?[]const u8,
        generations: []const struct { prompt: []const u8, tokens: []const u32, results: []const []const u32, masks: []const []const u8 },
    };
    var parsed = try std.json.parseFromSlice(Expected, allocator, bytes, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.generations.len > 0);
    for (parsed.value.generations) |g| {
        const prompt = try fx.tokenizer.encode(allocator, g.prompt);
        defer allocator.free(prompt);
        try std.testing.expectEqualSlices(u32, g.tokens[1..], prompt);
        var batch = try engine.generateBatch(allocator, g.tokens, .{ .num_samples = g.results.len, .max_tokens = 24, .temperature = 0 });
        defer batch.deinit();
        for (g.results, g.masks, batch.results, batch.masks) |want, want_mask, got, got_mask| {
            try std.testing.expectEqualSlices(u32, want, got);
            try std.testing.expectEqualSlices(u8, want_mask, got_mask);
        }
        // Gpt.greedy (the trainer's samples) decodes the same tokens.
        const greedy = try fx.model.greedy(allocator, g.tokens, 24, null);
        defer allocator.free(greedy);
        const generated = g.results[0][g.tokens.len..];
        try std.testing.expectEqualSlices(u32, generated, greedy[0..generated.len]);
    }
}

test "engine sampling is seeded, honors top-k and stops at max tokens" {
    const allocator = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.init(allocator);
    defer fx.deinit();
    const engine = Engine.init(&fx.model, &fx.tokenizer);
    const prompt = [_]u32{ 1024, 72, 281, 367 };

    var a = try engine.generateBatch(allocator, &prompt, .{ .num_samples = 3, .max_tokens = 10, .temperature = 1, .top_k = 5, .seed = 7 });
    defer a.deinit();
    var b = try engine.generateBatch(allocator, &prompt, .{ .num_samples = 3, .max_tokens = 10, .temperature = 1, .top_k = 5, .seed = 7 });
    defer b.deinit();
    for (a.results, b.results) |x, y| {
        try std.testing.expectEqualSlices(u32, x, y);
        try std.testing.expect(x.len <= prompt.len + 10);
    }
    // top_k = 1 is greedy.
    var greedy = try engine.generateBatch(allocator, &prompt, .{ .max_tokens = 10, .temperature = 0 });
    defer greedy.deinit();
    var top1 = try engine.generateBatch(allocator, &prompt, .{ .max_tokens = 10, .temperature = 0.8, .top_k = 1 });
    defer top1.deinit();
    try std.testing.expectEqualSlices(u32, greedy.results[0], top1.results[0]);

    // Without max_tokens, generation stops when the cache (sequence_len) fills.
    var gen = try engine.generate(allocator, &prompt, .{ .temperature = 0 });
    defer gen.deinit();
    var steps: usize = 0;
    while (try gen.next()) |_| steps += 1;
    try std.testing.expect(steps <= fx.model.config.sequence_len - prompt.len + 1);
}

test "engine calculator matches python's use_calculator" {
    const allocator = std.testing.allocator;
    const storage = mod.Storage.init(allocator, std.testing.io);
    const bytes = try storage.read(mod.build_options.source_root ++ "/testdata/engine.json");
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(struct { calculator: []const [2]?[]const u8 }, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.calculator) |case| {
        const got = try mod.Calculator.evaluate(allocator, case[0].?);
        defer if (got) |g| allocator.free(g);
        const want = case[1];
        if ((want == null) != (got == null) or (want != null and !std.mem.eql(u8, want.?, got.?))) {
            std.debug.print("calculator {s}: python {?s}, zig {?s}\n", .{ case[0].?, want, got });
            return error.TestUnexpectedResult;
        }
    }
}
