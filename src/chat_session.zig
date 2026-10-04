const std = @import("std");
const log = std.log.scoped(.zignanogpt_chat_session);
const mod = @import("module.zig");

/// `chat_cli.py`'s settings.
pub const ChatOptions = struct {
    temperature: f32 = 0.6,
    top_k: ?usize = 50,
    /// Tokens per reply at most.
    max_tokens: usize = 256,
    seed: u64 = 42,
};

/// A conversation with a model (nanochat's `chat_cli.py` state machine): the
/// tokens so far (`<|bos|>`, then user and assistant turns), extended by each
/// reply, whose text streams to a writer as it is generated.
pub const ChatSession = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    engine: mod.Engine,
    options: ChatOptions,
    tokens: std.ArrayList(u32) = .empty,
    bos: u32,
    user_start: u32,
    user_end: u32,
    assistant_start: u32,
    assistant_end: u32,

    /// Starts an empty conversation.
    ///
    /// Parameters:
    /// - `allocator`: owns the conversation.
    /// - `model`: the model (outlives the session).
    /// - `tokenizer`: its tokenizer (outlives the session).
    /// - `options`: sampling settings.
    ///
    /// Return: the session; `error.UnknownSpecialToken`, allocation errors.
    pub fn init(allocator: std.mem.Allocator, model: *mod.Gpt, tokenizer: *const mod.Tokenizer, options: ChatOptions) !Self {
        var self = Self{
            .allocator = allocator,
            .engine = mod.Engine.init(model, tokenizer),
            .options = options,
            .bos = try tokenizer.bos(),
            .user_start = try tokenizer.special("<|user_start|>"),
            .user_end = try tokenizer.special("<|user_end|>"),
            .assistant_start = try tokenizer.special("<|assistant_start|>"),
            .assistant_end = try tokenizer.special("<|assistant_end|>"),
        };
        try self.tokens.append(allocator, self.bos);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.tokens.deinit(self.allocator);
    }

    /// Forgets the conversation (`clear`).
    pub fn clear(self: *Self) void {
        self.tokens.shrinkRetainingCapacity(1);
    }

    /// Adds a user turn and generates the assistant's reply, writing each
    /// token's text to `out` (flushed per token) as it comes. The reply,
    /// ended by `<|assistant_end|>` even when cut short, joins the conversation.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `user`: the user's message.
    /// - `out`: receives the reply's text.
    /// - `stop`: set to end the reply early, or null.
    ///
    /// Return: nothing; `error.SequenceTooLong` when the conversation outgrew
    /// the model (it is left as before), generation and writer errors.
    pub fn reply(self: *Self, user: []const u8, out: *std.Io.Writer, stop: ?*const std.atomic.Value(bool)) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const tokenizer = self.engine.tokenizer;
        const before = self.tokens.items.len;
        errdefer self.tokens.shrinkRetainingCapacity(before);
        try self.tokens.append(self.allocator, self.user_start);
        try tokenizer.encodeAppend(self.allocator, &self.tokens, user);
        try self.tokens.append(self.allocator, self.user_end);
        try self.tokens.append(self.allocator, self.assistant_start);

        var gen = try self.engine.generate(self.allocator, self.tokens.items, .{
            .max_tokens = self.options.max_tokens,
            .temperature = self.options.temperature,
            .top_k = self.options.top_k,
            .seed = self.options.seed,
        });
        defer gen.deinit();
        var last: ?u32 = null;
        while (try gen.next()) |column| {
            const token = column.tokens[0];
            try self.tokens.append(self.allocator, token);
            last = token;
            if (token != self.assistant_end) {
                try out.writeAll(tokenizer.tokenBytes(token));
                try out.flush();
            }
            if (stop) |s| {
                if (s.load(.acquire)) break;
            }
        }
        if (last != self.assistant_end) try self.tokens.append(self.allocator, self.assistant_end);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "chat session keeps the conversation in nanochat's token layout" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata/nanochat_base";
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const meta_bytes = try storage.read(root ++ "/base_checkpoints/d2/meta_000005.json");
    defer allocator.free(meta_bytes);
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), meta_bytes, .{});
    var model = try mod.Gpt.init(allocator, &backend, try mod.GptConfig.fromJson(arena.allocator(), meta.object.get("model_config").?));
    defer model.deinit();
    var state = try mod.TorchImport.loadStateDict(allocator, std.testing.io, root ++ "/base_checkpoints/d2/model_000005.pt");
    defer state.deinit();
    try mod.TorchImport.loadWeights(&backend, &state, &model.weights);

    var session = try ChatSession.init(allocator, &model, &tok, .{ .temperature = 0, .max_tokens = 5 });
    defer session.deinit();
    var text: std.Io.Writer.Allocating = .init(allocator);
    defer text.deinit();
    try session.reply("Hi", &text.writer, null);
    const t = session.tokens.items;
    // bos, user_start, "Hi"..., user_end, assistant_start, 5 tokens, assistant_end
    try std.testing.expectEqual(session.bos, t[0]);
    try std.testing.expectEqual(session.user_start, t[1]);
    try std.testing.expectEqual(session.assistant_end, t[t.len - 1]);
    const hi = try tok.encode(allocator, "Hi");
    defer allocator.free(hi);
    try std.testing.expectEqual(1 + 1 + hi.len + 2 + 5 + 1, t.len);
    // The streamed text is the generated tokens' bytes.
    const generated = t[t.len - 6 .. t.len - 1];
    const decoded = try tok.decode(allocator, generated);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(decoded, text.written());

    // A second turn extends the conversation; clear drops back to bos.
    try session.reply("Again", &text.writer, null);
    try std.testing.expect(session.tokens.items.len > t.len);
    session.clear();
    try std.testing.expectEqual(@as(usize, 1), session.tokens.items.len);
}
