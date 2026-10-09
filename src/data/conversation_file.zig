const std = @import("std");
const log = std.log.scoped(.zignanogpt_conversation_file);
const mod = @import("../module.zig");

/// Your own chat data for SFT: a JSONL file, one conversation per line, as a
/// JSON array of `{"role", "content"}` messages (the shape earlier nanochat
/// versions read with `CustomJSON`) or an object with a `messages` array (the
/// shape nanochat's tasks produce). Every line is checked when the file loads;
/// the first bad one fails the load with its line number.
pub const ConversationFile = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    /// Holds the text, the parsed JSON and the message arrays.
    arena: std.heap.ArenaAllocator,
    conversations: []mod.Conversation,

    /// Reads and checks a file.
    ///
    /// Parameters:
    /// - `allocator`: backs the arena.
    /// - `storage`: the file helper.
    /// - `path`: the JSONL file.
    ///
    /// Return: the conversations; `error.InvalidConversation` (warned with the
    /// line), `error.EmptyConversations`, storage errors.
    pub fn load(allocator: std.mem.Allocator, storage: mod.Storage, path: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const text = try storage.read(path);
        defer allocator.free(text);
        return parse(allocator, text, path);
    }

    /// Parses JSONL text (copied).
    ///
    /// Parameters:
    /// - `allocator`: backs the arena.
    /// - `text`: the lines.
    /// - `name`: for messages (the file's path).
    ///
    /// Return: the conversations; `error.InvalidConversation`, `error.EmptyConversations`.
    pub fn parse(allocator: std.mem.Allocator, text: []const u8, name: []const u8) !Self {
        var self: Self = .{ .allocator = allocator, .arena = std.heap.ArenaAllocator.init(allocator), .conversations = &.{} };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        const owned = try a.dupe(u8, text);
        var list: std.ArrayList(mod.Conversation) = .empty;
        var lines = std.mem.splitScalar(u8, owned, '\n');
        var number: usize = 0;
        while (lines.next()) |raw| {
            number += 1;
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0) continue;
            const value = std.json.parseFromSliceLeaky(std.json.Value, a, line, .{}) catch |err| {
                return bad(name, number, @errorName(err));
            };
            const conversation = switch (value) {
                .array => |items| mod.Conversation.fromMessages(a, items.items),
                .object => mod.Conversation.fromJson(a, value),
                else => return bad(name, number, "not a list of messages"),
            } catch return bad(name, number, "a message needs a role (system, user, assistant) and string content");
            if (conversation.problem()) |reason| return bad(name, number, reason);
            try list.append(a, conversation);
        }
        if (list.items.len == 0) {
            log.warn("{s}: no conversations", .{name});
            return error.EmptyConversations;
        }
        self.conversations = list.items;
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    fn bad(name: []const u8, line: usize, reason: []const u8) error{InvalidConversation} {
        log.warn("{s}:{d}: {s}", .{ name, line, reason });
        return error.InvalidConversation;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "conversation files read both shapes, skip blank lines and name the bad line" {
    const allocator = std.testing.allocator;
    const good =
        \\[{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "zignanogpt, a small model."}]
        \\
        \\{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": [{"type": "text", "text": "4"}]}]}
        \\
    ;
    var file = try ConversationFile.parse(allocator, good, "good.jsonl");
    defer file.deinit();
    try std.testing.expectEqual(@as(usize, 2), file.conversations.len);
    try std.testing.expectEqualStrings("zignanogpt, a small model.", file.conversations[0].messages[1].content.text);
    try std.testing.expectEqual(mod.Role.system, file.conversations[1].messages[0].role);

    const first_line = good[0 .. std.mem.findScalar(u8, good, '\n').? + 1];
    const cases = [_][]const u8{
        "[{\"role\": \"user\", \"content\": \"hi\"}]", // no reply
        "[{\"role\": \"assistant\", \"content\": \"hi\"}, {\"role\": \"user\", \"content\": \"?\"}]", // wrong order
        "[{\"role\": \"user\", \"content\": 3}, {\"role\": \"assistant\", \"content\": \"x\"}]", // content not text
        "{\"text\": \"a pretraining row\"}", // no messages
        "not json",
    };
    for (cases) |case| {
        const text = try allocator.print("{s}{s}\n", .{ first_line, case });
        defer allocator.free(text);
        try std.testing.expectError(error.InvalidConversation, ConversationFile.parse(allocator, text, "case.jsonl"));
    }
    try std.testing.expectError(error.EmptyConversations, ConversationFile.parse(allocator, "\n\n", "empty.jsonl"));
}
