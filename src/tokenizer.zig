const std = @import("std");
const log = std.log.scoped(.zignanogpt_tokenizer);
const mod = @import("module.zig");

/// A rendered conversation: token ids, and a mask that is 1 where the
/// assistant is trained to produce the token.
pub const Rendered = struct {
    ids: std.ArrayList(u32) = .empty,
    mask: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Rendered, allocator: std.mem.Allocator) void {
        self.ids.deinit(allocator);
        self.mask.deinit(allocator);
    }
};

/// A byte-level BPE tokenizer with tiktoken's encoding (nanochat's
/// `RustBPETokenizer`): the split pattern cuts text into pieces, and each
/// piece is merged by repeatedly joining the adjacent pair whose bytes have
/// the lowest rank. Token id = rank; special tokens follow the ranks.
pub const Tokenizer = struct {
    const Self = @This();

    /// nanochat's special tokens, appended after the mergeable vocabulary in this order.
    pub const special_tokens = [_][]const u8{
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>",
    };

    /// nanochat's split pattern bounds digit runs at 2 (GPT-4's cl100k uses 3).
    pub const nanochat_max_digits = 2;

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    /// Bytes of every token by id; special tokens hold their text.
    vocab: []const []const u8,
    /// Mergeable token bytes -> rank.
    ranks: std.StringHashMapUnmanaged(u32),
    /// Count of mergeable tokens (the first special token's id).
    num_mergeable: usize,
    num_special: usize,
    max_digits: usize,

    /// Builds a tokenizer from its mergeable tokens.
    ///
    /// Parameters:
    /// - `allocator`: owns the tables.
    /// - `tokens`: token bytes in rank order; copied.
    /// - `specials`: special token names, ids following the ranks.
    /// - `max_digits`: the split pattern's digit-run bound.
    ///
    /// Return: the tokenizer; allocation errors, `error.DuplicateToken`.
    pub fn init(allocator: std.mem.Allocator, tokens: []const []const u8, specials: []const []const u8, max_digits: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = undefined;
        self.allocator = allocator;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        self.num_mergeable = tokens.len;
        self.num_special = specials.len;
        self.max_digits = max_digits;
        const vocab = try arena.alloc([]const u8, tokens.len + specials.len);
        self.ranks = .empty;
        try self.ranks.ensureTotalCapacity(arena, @intCast(tokens.len));
        for (tokens, 0..) |bytes, rank| {
            vocab[rank] = try arena.dupe(u8, bytes);
            const entry = self.ranks.getOrPutAssumeCapacity(vocab[rank]);
            if (entry.found_existing) {
                log.err("token bytes appear twice (ranks {d} and {d})", .{ entry.value_ptr.*, rank });
                return error.DuplicateToken;
            }
            entry.value_ptr.* = @intCast(rank);
        }
        for (specials, 0..) |name, i| vocab[tokens.len + i] = try arena.dupe(u8, name);
        self.vocab = vocab;
        return self;
    }

    /// Frees the tables.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.arena.deinit();
    }

    pub fn vocabSize(self: *const Self) usize {
        return self.vocab.len;
    }

    /// The id of a special token.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `name`: e.g. `<|bos|>`.
    ///
    /// Return: the id, or `error.UnknownSpecialToken`.
    pub fn special(self: *const Self, name: []const u8) !u32 {
        for (self.vocab[self.num_mergeable..], self.num_mergeable..) |text, id| {
            if (std.mem.eql(u8, text, name)) return @intCast(id);
        }
        log.debug("unknown special token {s}", .{name});
        return error.UnknownSpecialToken;
    }

    pub fn bos(self: *const Self) !u32 {
        return self.special("<|bos|>");
    }

    /// The bytes of a token.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `id`: a token id below `vocabSize`.
    ///
    /// Return: the bytes (a special token's name for specials).
    pub fn tokenBytes(self: *const Self, id: u32) []const u8 {
        return self.vocab[id];
    }

    /// Encodes ordinary text (special-token text is encoded literally), as
    /// tiktoken's `encode_ordinary`.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `allocator`: grows `out`.
    /// - `out`: receives the ids, appended.
    /// - `text`: valid UTF-8.
    ///
    /// Return: nothing; `error.InvalidUtf8`, allocation errors.
    pub fn encodeAppend(self: *const Self, allocator: std.mem.Allocator, out: *std.ArrayList(u32), text: []const u8) !void {
        var pieces = try mod.Pretokenizer.init(text, self.max_digits);
        while (pieces.next()) |piece| {
            if (self.ranks.get(piece)) |rank| {
                try out.append(allocator, rank);
                continue;
            }
            try self.bytePairEncode(allocator, out, piece);
        }
    }

    /// Encodes ordinary text into a new list.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `allocator`: allocates the result.
    /// - `text`: valid UTF-8.
    ///
    /// Return: the ids, owned by the caller.
    pub fn encode(self: *const Self, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        try self.encodeAppend(allocator, &out, text);
        return out.toOwnedSlice(allocator);
    }

    /// Concatenates token bytes (not necessarily valid UTF-8).
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `allocator`: allocates the result.
    /// - `ids`: token ids.
    ///
    /// Return: the bytes, owned by the caller; `error.OutOfBounds` for an unknown id.
    pub fn decode(self: *const Self, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            if (id >= self.vocab.len) return error.OutOfBounds;
            try out.appendSlice(allocator, self.vocab[id]);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Bytes per token for bits-per-byte (`token_bytes.pt`): 0 for special tokens.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `allocator`: allocates the result.
    ///
    /// Return: one count per id, owned by the caller.
    pub fn tokenByteCounts(self: *const Self, allocator: std.mem.Allocator) ![]i32 {
        const out = try allocator.alloc(i32, self.vocab.len);
        for (out, 0..) |*o, id| o.* = if (id >= self.num_mergeable) 0 else @intCast(self.vocab[id].len);
        return out;
    }

    /// nanochat's `render_conversation`: BOS, then each user message between
    /// `<|user_start|>`/`<|user_end|>` (mask 0) and each assistant message
    /// between `<|assistant_start|>`/`<|assistant_end|>` (mask 1 for its tokens
    /// and the end marker; python calls wrapped in `<|python_start|>`/`<|python_end|>`;
    /// python outputs between `<|output_start|>`/`<|output_end|>` with mask 0).
    /// A leading system message is merged into the first user message. The
    /// result is truncated to `max_tokens`.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `allocator`: grows the result.
    /// - `conversation`: alternating user/assistant messages, optional system first.
    /// - `max_tokens`: the cap (nanochat's default is 2048).
    ///
    /// Return: ids and mask; `error.InvalidConversation` on a bad role order.
    pub fn renderConversation(self: *const Self, allocator: std.mem.Allocator, conversation: mod.Conversation, max_tokens: usize) !Rendered {
        var r: Rendered = .{};
        errdefer r.deinit(allocator);
        var messages = conversation.messages;
        var merged_first: ?[]u8 = null;
        defer if (merged_first) |m| allocator.free(m);
        if (messages.len > 0 and messages[0].role == .system) {
            if (messages.len < 2 or messages[1].role != .user) return error.InvalidConversation;
            merged_first = try std.mem.concat(allocator, u8, &.{ messages[0].content.text, "\n\n", messages[1].content.text });
            messages = messages[1..];
        }
        if (messages.len == 0) return error.InvalidConversation;

        try self.add(allocator, &r, try self.bos(), 0);
        for (messages, 0..) |message, i| {
            const want: mod.Role = if (i % 2 == 0) .user else .assistant;
            if (message.role != want) {
                log.debug("message {d} is from {t}, expected {t}", .{ i, message.role, want });
                return error.InvalidConversation;
            }
            switch (message.role) {
                .user => {
                    const text = if (i == 0 and merged_first != null) merged_first.? else switch (message.content) {
                        .text => |t| t,
                        .parts => return error.InvalidConversation,
                    };
                    try self.add(allocator, &r, try self.special("<|user_start|>"), 0);
                    try self.addText(allocator, &r, text, 0);
                    try self.add(allocator, &r, try self.special("<|user_end|>"), 0);
                },
                .assistant => {
                    try self.add(allocator, &r, try self.special("<|assistant_start|>"), 0);
                    switch (message.content) {
                        .text => |t| try self.addText(allocator, &r, t, 1),
                        .parts => |parts| for (parts) |part| switch (part.kind) {
                            .text => try self.addText(allocator, &r, part.text, 1),
                            .python => {
                                try self.add(allocator, &r, try self.special("<|python_start|>"), 1);
                                try self.addText(allocator, &r, part.text, 1);
                                try self.add(allocator, &r, try self.special("<|python_end|>"), 1);
                            },
                            .python_output => {
                                try self.add(allocator, &r, try self.special("<|output_start|>"), 0);
                                try self.addText(allocator, &r, part.text, 0);
                                try self.add(allocator, &r, try self.special("<|output_end|>"), 0);
                            },
                        },
                    }
                    try self.add(allocator, &r, try self.special("<|assistant_end|>"), 1);
                },
                .system => return error.InvalidConversation,
            }
        }
        if (r.ids.items.len > max_tokens) {
            r.ids.shrinkRetainingCapacity(max_tokens);
            r.mask.shrinkRetainingCapacity(max_tokens);
        }
        return r;
    }

    /// nanochat's `render_for_completion`: the conversation without its final
    /// assistant message, then `<|assistant_start|>` to prime a reply.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `allocator`: allocates the result.
    /// - `conversation`: ending with an assistant message.
    ///
    /// Return: the ids, owned by the caller.
    pub fn renderForCompletion(self: *const Self, allocator: std.mem.Allocator, conversation: mod.Conversation) ![]u32 {
        const messages = conversation.messages;
        if (messages.len == 0 or messages[messages.len - 1].role != .assistant) return error.InvalidConversation;
        var r = try self.renderConversation(allocator, .{ .messages = messages[0 .. messages.len - 1] }, 2048);
        defer r.mask.deinit(allocator);
        errdefer r.ids.deinit(allocator);
        try r.ids.append(allocator, try self.special("<|assistant_start|>"));
        return r.ids.toOwnedSlice(allocator);
    }

    /// Writes the mergeable ranks in tiktoken's `.tiktoken` format
    /// (`base64(bytes) rank` per line). Special tokens are not stored.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `writer`: the destination.
    ///
    /// Return: nothing; write errors.
    pub fn writeTiktoken(self: *const Self, writer: *std.Io.Writer) !void {
        for (self.vocab[0..self.num_mergeable], 0..) |bytes, rank| {
            try std.base64.standard.Encoder.encodeWriter(writer, bytes);
            try writer.print(" {d}\n", .{rank});
        }
    }

    /// Parses a `.tiktoken` file's ranks (ranks must be `0..n` without gaps).
    ///
    /// Parameters:
    /// - `allocator`: owns the tokenizer.
    /// - `text`: the file contents.
    /// - `specials`: special token names appended after the ranks.
    /// - `max_digits`: the split pattern's digit-run bound.
    ///
    /// Return: the tokenizer; `error.InvalidTiktoken` on a malformed file.
    pub fn parseTiktoken(allocator: std.mem.Allocator, text: []const u8, specials: []const []const u8, max_digits: usize) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var tokens: std.ArrayList([]const u8) = .empty;
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const space = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidTiktoken;
            const encoded = line[0..space];
            const rank = std.fmt.parseInt(usize, std.mem.trimEnd(u8, line[space + 1 ..], "\r"), 10) catch return error.InvalidTiktoken;
            if (rank != tokens.items.len) {
                log.err("tiktoken ranks must be consecutive: got {d} at line {d}", .{ rank, tokens.items.len });
                return error.InvalidTiktoken;
            }
            const size = std.base64.standard.Decoder.calcSizeForSlice(encoded) catch return error.InvalidTiktoken;
            const bytes = try arena.allocator().alloc(u8, size);
            std.base64.standard.Decoder.decode(bytes, encoded) catch return error.InvalidTiktoken;
            try tokens.append(arena.allocator(), bytes);
        }
        return init(allocator, tokens.items, specials, max_digits);
    }

    /// The file name `save` writes in a tokenizer directory.
    pub const file_name = "tokenizer.tiktoken";

    /// Writes `<dir>/tokenizer.tiktoken`.
    ///
    /// Parameters:
    /// - `self`: the tokenizer.
    /// - `storage`: the file helper.
    /// - `dir`: the tokenizer directory (created if missing).
    ///
    /// Return: nothing; storage errors.
    pub fn save(self: *const Self, storage: mod.Storage, dir: []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var buffer: std.Io.Writer.Allocating = .init(self.allocator);
        defer buffer.deinit();
        try self.writeTiktoken(&buffer.writer);
        const path = try std.fs.path.join(self.allocator, &.{ dir, file_name });
        defer self.allocator.free(path);
        try storage.write(path, buffer.written());
    }

    /// Reads `<dir>/tokenizer.tiktoken` with nanochat's special tokens.
    ///
    /// Parameters:
    /// - `allocator`: owns the tokenizer.
    /// - `storage`: the file helper.
    /// - `dir`: the tokenizer directory.
    ///
    /// Return: the tokenizer; storage or format errors.
    pub fn load(allocator: std.mem.Allocator, storage: mod.Storage, dir: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const path = try std.fs.path.join(allocator, &.{ dir, file_name });
        defer allocator.free(path);
        const text = storage.read(path) catch |err| {
            log.err("no tokenizer at {s}; run `zignanogpt tok-train` first [{t}]", .{ path, err });
            return err;
        };
        defer allocator.free(text);
        return parseTiktoken(allocator, text, &Self.special_tokens, Self.nanochat_max_digits);
    }

    /// tiktoken's `_byte_pair_merge` + `byte_pair_encode` for one piece.
    fn bytePairEncode(self: *const Self, allocator: std.mem.Allocator, out: *std.ArrayList(u32), piece: []const u8) !void {
        const none = std.math.maxInt(u32);
        const Part = struct { start: usize, rank: u32 };
        var stack: [64]Part = undefined;
        const heap = if (piece.len + 1 > stack.len) try allocator.alloc(Part, piece.len + 1) else null;
        defer if (heap) |h| allocator.free(h);
        const backing: []Part = heap orelse &stack;
        var parts = backing[0 .. piece.len + 1];

        for (0..piece.len - 1) |i| parts[i] = .{ .start = i, .rank = self.ranks.get(piece[i .. i + 2]) orelse none };
        parts[piece.len - 1] = .{ .start = piece.len - 1, .rank = none };
        parts[piece.len] = .{ .start = piece.len, .rank = none };

        var len = parts.len;
        while (true) {
            var best: u32 = none;
            var at: usize = 0;
            for (parts[0 .. len - 1], 0..) |p, i| {
                if (p.rank < best) {
                    best = p.rank;
                    at = i;
                }
            }
            if (best == none) break;
            // Merge parts[at] and parts[at + 1]: recompute the ranks around the join.
            if (at > 0) parts[at - 1].rank = self.rankSpan(piece, parts[0..len], at - 1);
            parts[at].rank = self.rankSpan(piece, parts[0..len], at);
            std.mem.copyForwards(Part, parts[at + 1 .. len - 1], parts[at + 2 .. len]);
            len -= 1;
        }
        for (0..len - 1) |i| {
            const bytes = piece[parts[i].start..parts[i + 1].start];
            try out.append(allocator, self.ranks.get(bytes) orelse return error.UnencodableBytes);
        }
    }

    /// Rank of the bytes spanning parts `i .. i + 3` (the merged pair plus its
    /// right neighbor), or none.
    fn rankSpan(self: *const Self, piece: []const u8, parts: anytype, i: usize) u32 {
        if (i + 3 >= parts.len) return std.math.maxInt(u32);
        return self.ranks.get(piece[parts[i].start..parts[i + 3].start]) orelse std.math.maxInt(u32);
    }

    fn add(self: *const Self, allocator: std.mem.Allocator, r: *Rendered, id: u32, mask: u8) !void {
        _ = self;
        try r.ids.append(allocator, id);
        try r.mask.append(allocator, mask);
    }

    fn addText(self: *const Self, allocator: std.mem.Allocator, r: *Rendered, text: []const u8, mask: u8) !void {
        const before = r.ids.items.len;
        try self.encodeAppend(allocator, &r.ids, text);
        try r.mask.appendNTimes(allocator, mask, r.ids.items.len - before);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

/// The fixture file, parsed.
const Fixture = struct {
    parsed: std.json.Parsed(std.json.Value),
    bytes: []u8,

    fn load(allocator: std.mem.Allocator) !Fixture {
        var node = try mod.zigstorage.Node.init(allocator, std.testing.io, .empty, mod.build_options.source_root ++ "/testdata/tokenizer.json");
        defer node.deinit();
        const bytes = try node.read(.all);
        errdefer allocator.free(bytes);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}), .bytes = bytes };
    }

    fn deinit(self: *Fixture, allocator: std.mem.Allocator) void {
        self.parsed.deinit();
        allocator.free(self.bytes);
    }

    fn get(self: *const Fixture, key: []const u8) std.json.Value {
        return self.parsed.value.object.get(key).?;
    }

    /// The fixture's tokenizer, built from its ranks.
    fn tokenizer(self: *const Fixture, allocator: std.mem.Allocator) !mod.Tokenizer {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const ranks = self.get("ranks").array.items;
        const tokens = try arena.allocator().alloc([]const u8, ranks.len);
        for (ranks, tokens) |r, *t| t.* = try jsonBytes(arena.allocator(), r);
        return mod.Tokenizer.init(allocator, tokens, &mod.Tokenizer.special_tokens, mod.Tokenizer.nanochat_max_digits);
    }
};

fn jsonBytes(allocator: std.mem.Allocator, value: std.json.Value) ![]u8 {
    const out = try allocator.alloc(u8, value.array.items.len);
    for (value.array.items, out) |b, *o| o.* = @intCast(b.integer);
    return out;
}

fn jsonIds(allocator: std.mem.Allocator, value: std.json.Value) ![]u32 {
    const out = try allocator.alloc(u32, value.array.items.len);
    for (value.array.items, out) |b, *o| o.* = @intCast(b.integer);
    return out;
}

test "tokenizer encodes exactly like tiktoken on the fixture" {
    const allocator = std.testing.allocator;
    var fixture = try Fixture.load(allocator);
    defer fixture.deinit(allocator);
    var tok = try fixture.tokenizer(allocator);
    defer tok.deinit();

    var specials = fixture.get("specials").object.iterator();
    while (specials.next()) |kv| try std.testing.expectEqual(@as(u32, @intCast(kv.value_ptr.integer)), try tok.special(kv.key_ptr.*));

    for (fixture.get("cases").array.items) |case| {
        const text = case.object.get("text").?.string;
        const want = try jsonIds(allocator, case.object.get("ids").?);
        defer allocator.free(want);
        const got = try tok.encode(allocator, text);
        defer allocator.free(got);
        std.testing.expectEqualSlices(u32, want, got) catch |err| {
            std.debug.print("encoding differs for {f}\n", .{std.json.fmt(text, .{})});
            return err;
        };
        const round_trip = try tok.decode(allocator, got);
        defer allocator.free(round_trip);
        try std.testing.expectEqualStrings(text, round_trip);
    }

    const counts = try tok.tokenByteCounts(allocator);
    defer allocator.free(counts);
    for (fixture.get("token_bytes").array.items, counts) |want, got| try std.testing.expectEqual(@as(i32, @intCast(want.integer)), got);
}

test "tokenizer renders conversations like nanochat" {
    const allocator = std.testing.allocator;
    var fixture = try Fixture.load(allocator);
    defer fixture.deinit(allocator);
    var tok = try fixture.tokenizer(allocator);
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    for (fixture.get("conversations").array.items) |entry| {
        const conv = try mod.Conversation.fromJson(arena.allocator(), entry.object.get("conversation").?);
        var r = try tok.renderConversation(allocator, conv, 2048);
        defer r.deinit(allocator);
        const want_ids = try jsonIds(arena.allocator(), entry.object.get("ids").?);
        try std.testing.expectEqualSlices(u32, want_ids, r.ids.items);
        const mask = entry.object.get("mask").?.array.items;
        for (mask, r.mask.items) |m, got| try std.testing.expectEqual(@as(u8, @intCast(m.integer)), got);
    }
    const conv = try mod.Conversation.fromJson(arena.allocator(), fixture.get("conversations").array.items[1].object.get("conversation").?);
    const completion = try tok.renderForCompletion(allocator, conv);
    defer allocator.free(completion);
    try std.testing.expectEqualSlices(u32, try jsonIds(arena.allocator(), fixture.get("completion")), completion);
}

test "tokenizer round-trips the tiktoken file format" {
    const allocator = std.testing.allocator;
    var tok = try mod.Tokenizer.init(allocator, &.{ "a", "b", "ab", "\xff" }, &mod.Tokenizer.special_tokens, mod.Tokenizer.nanochat_max_digits);
    defer tok.deinit();
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    try tok.writeTiktoken(&buffer.writer);
    try std.testing.expectEqualStrings("YQ== 0\nYg== 1\nYWI= 2\n/w== 3\n", buffer.written());
    var back = try mod.Tokenizer.parseTiktoken(allocator, buffer.written(), &mod.Tokenizer.special_tokens, mod.Tokenizer.nanochat_max_digits);
    defer back.deinit();
    try std.testing.expectEqualStrings("ab", back.tokenBytes(2));
    try std.testing.expectEqual(@as(u32, 4), try back.bos());
}
