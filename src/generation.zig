const std = @import("std");
const log = std.log.scoped(.zignanogpt_generation);
const mod = @import("module.zig");

/// The special tokens the generation loop reacts to.
pub const ToolTokens = struct {
    python_start: u32,
    python_end: u32,
    output_start: u32,
    output_end: u32,
    assistant_end: u32,
    bos: u32,

    pub fn init(tokenizer: *const mod.Tokenizer) !ToolTokens {
        return .{
            .python_start = try tokenizer.special("<|python_start|>"),
            .python_end = try tokenizer.special("<|python_end|>"),
            .output_start = try tokenizer.special("<|output_start|>"),
            .output_end = try tokenizer.special("<|output_end|>"),
            .assistant_end = try tokenizer.special("<|assistant_end|>"),
            .bos = try tokenizer.bos(),
        };
    }
};

/// One row's state (nanochat's `RowState`): tokens waiting to be forced in,
/// the calculator expression being written, and whether the row has ended.
pub const RowState = struct {
    const Self = @This();

    forced: std.ArrayList(u32) = .empty,
    forced_head: usize = 0,
    in_python: bool = false,
    expr: std.ArrayList(u32) = .empty,
    completed: bool = false,

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.forced.deinit(allocator);
        self.expr.deinit(allocator);
    }

    /// The next forced token, if any.
    pub fn takeForced(self: *Self) ?u32 {
        if (self.forced_head == self.forced.items.len) return null;
        const t = self.forced.items[self.forced_head];
        self.forced_head += 1;
        if (self.forced_head == self.forced.items.len) {
            self.forced.clearRetainingCapacity();
            self.forced_head = 0;
        }
        return t;
    }

    /// Updates the row for its next token: ends it on `<|assistant_end|>` or
    /// `<|bos|>`, and runs the calculator when a python block closes, queueing
    /// `<|output_start|> str(result) <|output_end|>`.
    ///
    /// Parameters:
    /// - `self`: the row.
    /// - `allocator`: the row's lists and scratch.
    /// - `tokenizer`: decodes the expression, encodes the result.
    /// - `tools`: the special tokens.
    /// - `token`: the row's next token.
    ///
    /// Return: nothing; allocation errors.
    pub fn accept(self: *Self, allocator: std.mem.Allocator, tokenizer: *const mod.Tokenizer, tools: ToolTokens, token: u32) !void {
        if (token == tools.assistant_end or token == tools.bos) self.completed = true;
        if (token == tools.python_start) {
            self.in_python = true;
            self.expr.clearRetainingCapacity();
        } else if (token == tools.python_end and self.in_python) {
            self.in_python = false;
            defer self.expr.clearRetainingCapacity();
            if (self.expr.items.len == 0) return;
            const expr = try tokenizer.decode(allocator, self.expr.items);
            defer allocator.free(expr);
            const result = try mod.Calculator.evaluate(allocator, expr) orelse return;
            defer allocator.free(result);
            const ids = try tokenizer.encode(allocator, result);
            defer allocator.free(ids);
            try self.forced.append(allocator, tools.output_start);
            try self.forced.appendSlice(allocator, ids);
            try self.forced.append(allocator, tools.output_end);
        } else if (self.in_python) {
            try self.expr.append(allocator, token);
        }
    }
};

/// One step's output: the next token of every row, and whether each was
/// sampled (1) or forced by the calculator (0).
pub const Column = struct {
    tokens: []const u32,
    masks: []const u8,
};

/// A running generation (nanochat's `Engine.generate`): a batch-1 prefill of
/// the prompt copied into `num_samples` rows, then one token per row per
/// `next` call until every row ends, `max_tokens` is reached or the cache is full.
pub const Generation = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    model: *mod.Gpt,
    tokenizer: *const mod.Tokenizer,
    tools: ToolTokens,
    options: mod.GenerateOptions,
    rng: mod.Random,
    cache: mod.KvCache,
    bufs: mod.InferenceBuffers,
    /// `[B, 1]` ids of the last column, fed on the next call.
    idx: mod.Tensor,
    /// `[B, V]` logits on the host.
    logits: []f32,
    rows: []RowState,
    column: []u32,
    masks: []u8,
    ids: []i32,
    /// Sampling scratch: `[V]` indices and weights.
    order: []u32,
    weights: []f32,
    generated: usize = 0,
    /// Whether `column` still has to be fed through the model.
    pending: bool = false,

    /// Prefills the prompt and prepares the rows.
    ///
    /// Parameters:
    /// - `allocator`: owns everything.
    /// - `model`: the model.
    /// - `tokenizer`: for the calculator.
    /// - `prompt`: the token ids, non-empty.
    /// - `options`: samples, limits and sampling.
    ///
    /// Return: the generation; `error.EmptyPrompt`, `error.SequenceTooLong`, backend errors.
    pub fn init(allocator: std.mem.Allocator, model: *mod.Gpt, tokenizer: *const mod.Tokenizer, prompt: []const u32, options: mod.GenerateOptions) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (prompt.len == 0) return error.EmptyPrompt;
        if (options.num_samples == 0) return error.InvalidOptions;
        if (!(options.temperature >= 0)) return error.InvalidOptions;
        const config = model.config;
        const be = model.backend;
        const b = options.num_samples;
        const vocab = config.vocab_size;
        if (prompt.len > config.rotarySeqLen()) return error.SequenceTooLong;
        const want = if (options.max_tokens) |m| prompt.len + m else @max(config.sequence_len, prompt.len);
        const capacity = @min(want, config.rotarySeqLen());

        // 1) Batch-1 prefill.
        const logits_one = try allocator.alloc(f32, vocab);
        defer allocator.free(logits_one);
        var prefill = try mod.KvCache.init(allocator, be, config, 1, prompt.len);
        defer prefill.deinit();
        {
            var bufs = try mod.InferenceBuffers.init(allocator, be, config, 1, prompt.len);
            defer bufs.deinit();
            const idx = try be.alloc(.i32, &.{ 1, prompt.len });
            defer be.free(idx);
            const ids = try allocator.alloc(i32, prompt.len);
            defer allocator.free(ids);
            for (prompt, ids) |t, *d| d.* = @intCast(t);
            try be.upload(idx, i32, ids);
            try model.forwardStep(&prefill, &bufs, idx);
            try be.download(bufs.logits, f32, logits_one);
        }

        // 2) One cache row per sample.
        var self: Self = undefined;
        self.allocator = allocator;
        self.model = model;
        self.tokenizer = tokenizer;
        self.tools = try ToolTokens.init(tokenizer);
        self.options = options;
        self.rng = mod.Random.init(options.seed);
        self.generated = 0;
        self.pending = false;
        self.cache = try mod.KvCache.init(allocator, be, config, b, capacity);
        errdefer self.cache.deinit();
        try self.cache.copyFrom(&prefill);
        self.bufs = try mod.InferenceBuffers.init(allocator, be, config, b, 1);
        errdefer self.bufs.deinit();
        self.idx = try be.alloc(.i32, &.{ b, 1 });
        errdefer be.free(self.idx);
        self.logits = try allocator.alloc(f32, b * vocab);
        errdefer allocator.free(self.logits);
        for (0..b) |r| @memcpy(self.logits[r * vocab ..][0..vocab], logits_one);

        // 3) Row states and scratch.
        self.rows = try allocator.alloc(RowState, b);
        errdefer allocator.free(self.rows);
        for (self.rows) |*r| r.* = .{};
        self.column = try allocator.alloc(u32, b);
        errdefer allocator.free(self.column);
        self.masks = try allocator.alloc(u8, b);
        errdefer allocator.free(self.masks);
        self.ids = try allocator.alloc(i32, b);
        errdefer allocator.free(self.ids);
        self.order = try allocator.alloc(u32, vocab);
        errdefer allocator.free(self.order);
        self.weights = try allocator.alloc(f32, vocab);
        return self;
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const a = self.allocator;
        a.free(self.weights);
        a.free(self.order);
        a.free(self.ids);
        a.free(self.masks);
        a.free(self.column);
        for (self.rows) |*r| r.deinit(a);
        a.free(self.rows);
        a.free(self.logits);
        self.model.backend.free(self.idx);
        self.bufs.deinit();
        self.cache.deinit();
    }

    /// The next token of every row (rows that ended keep producing tokens,
    /// as in nanochat; callers ignore them).
    ///
    /// Parameters:
    /// - `self`: the generation.
    ///
    /// Return: the column (valid until the next call), or null when done; backend errors.
    pub fn next(self: *Self) !?Column {
        if (self.options.max_tokens) |m| {
            if (self.generated >= m) return null;
        }
        if (self.allCompleted()) return null;
        if (self.pending) {
            if (self.cache.remaining() == 0) {
                log.debug("generation stopped: KV cache full at {d}", .{self.cache.pos});
                return null;
            }
            for (self.column, self.ids) |t, *d| d.* = @intCast(t);
            try self.model.backend.upload(self.idx, i32, self.ids);
            try self.model.forwardStep(&self.cache, &self.bufs, self.idx);
            try self.model.backend.download(self.bufs.logits, f32, self.logits);
            self.pending = false;
        }
        const vocab = self.model.config.vocab_size;
        for (self.rows, self.column, self.masks, 0..) |*row, *token, *mask, r| {
            // Python samples every row, then discards the draw for forced rows.
            const sampled = self.sample(self.logits[r * vocab ..][0..vocab]);
            if (row.takeForced()) |forced| {
                token.* = forced;
                mask.* = 0;
            } else {
                token.* = sampled;
                mask.* = 1;
            }
            try row.accept(self.allocator, self.tokenizer, self.tools, token.*);
        }
        self.generated += 1;
        self.pending = true;
        return .{ .tokens = self.column, .masks = self.masks };
    }

    fn allCompleted(self: *const Self) bool {
        for (self.rows) |r| {
            if (!r.completed) return false;
        }
        return true;
    }

    /// nanochat's `sample_next_token`: argmax at temperature 0, otherwise a
    /// draw from `softmax(logits / temperature)`, restricted to the top k.
    /// Draws use this port's generator, not PyTorch's.
    fn sample(self: *Self, logits: []const f32) u32 {
        const temperature = self.options.temperature;
        if (temperature == 0) return @intCast(std.mem.findMax(f32, logits));
        var n = logits.len;
        for (self.order, 0..) |*o, i| o.* = @intCast(i);
        if (self.options.top_k) |k| {
            if (k > 0 and k < logits.len) {
                std.mem.sortUnstable(u32, self.order, logits, struct {
                    fn greater(l: []const f32, a: u32, b: u32) bool {
                        return l[a] > l[b];
                    }
                }.greater);
                n = k;
            }
        }
        var peak: f32 = -std.math.inf(f32);
        for (self.order[0..n]) |i| peak = @max(peak, logits[i]);
        for (self.order[0..n], self.weights[0..n]) |i, *w| w.* = @exp((logits[i] - peak) / temperature);
        return self.order[self.rng.categorical(self.weights[0..n])];
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "row state forces the calculator's result after a python block" {
    const allocator = std.testing.allocator;
    const storage = mod.Storage.init(allocator, std.testing.io);
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, mod.build_options.source_root ++ "/testdata/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    const tools = try ToolTokens.init(&tok);
    var row: RowState = .{};
    defer row.deinit(allocator);
    const expr = try tok.encode(allocator, "12*3");
    defer allocator.free(expr);
    try row.accept(allocator, &tok, tools, tools.python_start);
    for (expr) |t| try row.accept(allocator, &tok, tools, t);
    try std.testing.expect(row.takeForced() == null);
    try row.accept(allocator, &tok, tools, tools.python_end);
    const result = try tok.encode(allocator, "36");
    defer allocator.free(result);
    try std.testing.expectEqual(tools.output_start, row.takeForced().?);
    for (result) |t| try std.testing.expectEqual(t, row.takeForced().?);
    try std.testing.expectEqual(tools.output_end, row.takeForced().?);
    try std.testing.expect(row.takeForced() == null);
    try std.testing.expect(!row.completed);

    // An invalid expression forces nothing; assistant_end ends the row.
    const bad = try tok.encode(allocator, "2**8");
    defer allocator.free(bad);
    try row.accept(allocator, &tok, tools, tools.python_start);
    for (bad) |t| try row.accept(allocator, &tok, tools, t);
    try row.accept(allocator, &tok, tools, tools.python_end);
    try std.testing.expect(row.takeForced() == null);
    try row.accept(allocator, &tok, tools, tools.assistant_end);
    try std.testing.expect(row.completed);
}
