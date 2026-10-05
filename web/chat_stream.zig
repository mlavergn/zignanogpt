const std = @import("std");
const log = std.log.scoped(.zignanogpt_chat_stream);
const web = @import("module.zig");
const mod = web.nanogpt;

/// One message of a chat request.
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

/// `POST /api/chat`'s body: the whole conversation (stateless, like nanochat's
/// web UI), ending with the user's turn, and the sampling settings.
pub const ChatRequest = struct {
    messages: []const ChatMessage,
    temperature: f32 = 0.6,
    top_k: ?usize = 50,
    max_tokens: usize = 512,
};

/// Generates an assistant reply as server-sent events: `data: {"token":"..."}`
/// per piece of text (whole UTF-8 characters only), then `data: {"done":true}`,
/// or `data: {"error":"..."}`.
pub const ChatStream = struct {
    /// Upper bound on a request's messages.
    pub const max_messages = 512;

    /// Renders the conversation as nanochat does (`<|bos|>`, user and
    /// assistant turns, `<|assistant_start|>`) and streams the reply.
    ///
    /// Parameters:
    /// - `allocator`: scratch.
    /// - `model`: the model.
    /// - `tokenizer`: its tokenizer.
    /// - `body`: the JSON request.
    /// - `seed`: the sampling seed.
    /// - `out`: receives the events (flushed per event).
    ///
    /// Return: nothing; write errors (request and model errors become an error event).
    pub fn run(allocator: std.mem.Allocator, model: *mod.Gpt, tokenizer: *const mod.Tokenizer, body: []const u8, seed: u64, out: *std.Io.Writer) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        stream(allocator, model, tokenizer, body, seed, out) catch |err| switch (err) {
            error.WriteFailed => return err,
            else => {
                log.warn("chat request failed [{t}]", .{err});
                try event(out, .{ .@"error" = @errorName(err) });
            },
        };
    }

    fn stream(allocator: std.mem.Allocator, model: *mod.Gpt, tokenizer: *const mod.Tokenizer, body: []const u8, seed: u64, out: *std.Io.Writer) !void {
        var parsed = try std.json.parseFromSlice(ChatRequest, allocator, body, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const req = parsed.value;
        if (req.messages.len == 0 or req.messages.len > max_messages) return error.InvalidChatRequest;
        if (!(req.temperature >= 0) or req.temperature > 2) return error.InvalidChatRequest;

        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(allocator);
        try tokens.append(allocator, try tokenizer.bos());
        for (req.messages, 0..) |m, i| {
            const user = i % 2 == 0;
            if (!std.mem.eql(u8, m.role, if (user) "user" else "assistant")) return error.InvalidChatRequest;
            try tokens.append(allocator, try tokenizer.special(if (user) "<|user_start|>" else "<|assistant_start|>"));
            try tokenizer.encodeAppend(allocator, &tokens, m.content);
            try tokens.append(allocator, try tokenizer.special(if (user) "<|user_end|>" else "<|assistant_end|>"));
        }
        if (req.messages.len % 2 == 0) return error.InvalidChatRequest; // must end with the user
        try tokens.append(allocator, try tokenizer.special("<|assistant_start|>"));

        const engine = mod.Engine.init(model, tokenizer);
        var gen = try engine.generate(allocator, tokens.items, .{
            .max_tokens = @min(req.max_tokens, 4096),
            .temperature = req.temperature,
            .top_k = if (req.top_k) |k| (if (k == 0) null else k) else null,
            .seed = seed,
        });
        defer gen.deinit();
        const assistant_end = try tokenizer.special("<|assistant_end|>");
        const bos = try tokenizer.bos();
        // Bytes of a character split across tokens wait for the rest.
        var pending: std.ArrayList(u8) = .empty;
        defer pending.deinit(allocator);
        while (try gen.next()) |column| {
            const token = column.tokens[0];
            if (token == assistant_end or token == bos) break;
            try pending.appendSlice(allocator, tokenizer.tokenBytes(token));
            const complete = completeUtf8(pending.items);
            if (complete == 0) continue;
            // Invalid bytes become U+FFFD so the event stays valid JSON.
            const text = try std.fmt.allocPrint(allocator, "{f}", .{std.unicode.fmtUtf8(pending.items[0..complete])});
            defer allocator.free(text);
            try event(out, .{ .token = text });
            const rest = pending.items.len - complete;
            std.mem.copyForwards(u8, pending.items[0..rest], pending.items[complete..]);
            pending.shrinkRetainingCapacity(rest);
        }
        if (pending.items.len > 0) try event(out, .{ .token = "\u{FFFD}" });
        try event(out, .{ .done = true });
    }

    /// The length of the longest prefix that ends on a UTF-8 character
    /// boundary (invalid bytes count as complete).
    fn completeUtf8(bytes: []const u8) usize {
        var i: usize = 0;
        while (i < bytes.len) {
            const n = std.unicode.utf8ByteSequenceLength(bytes[i]) catch {
                i += 1;
                continue;
            };
            if (i + n > bytes.len) return i;
            i += n;
        }
        return i;
    }

    /// One `data: <json>` event.
    fn event(out: *std.Io.Writer, value: anytype) !void {
        try out.writeAll("data: ");
        try std.json.Stringify.value(value, .{}, out);
        try out.writeAll("\n\n");
        try out.flush();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "chat stream sends the greedy reply as server-sent events" {
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

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try ChatStream.run(allocator, &model, &tok, "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"temperature\":0,\"max_tokens\":6}", 1, &out.writer);
    // The events' text is the greedy reply ChatSession produces.
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var done = false;
    var events = std.mem.splitSequence(u8, out.written(), "\n\n");
    while (events.next()) |e| {
        if (e.len == 0) continue;
        try std.testing.expect(std.mem.startsWith(u8, e, "data: "));
        const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), e[6..], .{});
        if (v.object.get("token")) |t| try text.appendSlice(allocator, t.string);
        if (v.object.get("done") != null) done = true;
    }
    try std.testing.expect(done);
    var session = try mod.ChatSession.init(allocator, &model, &tok, .{ .temperature = 0, .max_tokens = 6 });
    defer session.deinit();
    var want: std.Io.Writer.Allocating = .init(allocator);
    defer want.deinit();
    try session.reply("Hi", &want.writer, null);
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.unicode.fmtUtf8(want.written())}), text.items);

    // A malformed request becomes an error event.
    var bad: std.Io.Writer.Allocating = .init(allocator);
    defer bad.deinit();
    try ChatStream.run(allocator, &model, &tok, "{\"messages\":[{\"role\":\"assistant\",\"content\":\"x\"}]}", 1, &bad.writer);
    try std.testing.expect(std.mem.indexOf(u8, bad.written(), "\"error\"") != null);
}
