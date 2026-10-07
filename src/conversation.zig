const std = @import("std");
const log = std.log.scoped(.zignanogpt_conversation);
const mod = @import("module.zig");

pub const Role = enum { system, user, assistant };

/// A piece of an assistant message: plain text, a python tool call, or the
/// tool's output (which the model is not trained to produce).
pub const MessagePart = struct {
    pub const Kind = enum { text, python, python_output };
    kind: Kind,
    text: []const u8,
};

/// One message. User and system messages are plain text; assistant messages
/// are text or a list of parts.
pub const Message = struct {
    role: Role,
    content: union(enum) {
        text: []const u8,
        parts: []const MessagePart,
    },
};

/// A chat conversation, as nanochat's tasks produce it
/// (`{"messages": [{"role": ..., "content": str | [{"type", "text"}]}]}`).
pub const Conversation = struct {
    const Self = @This();

    messages: []const Message,

    /// Converts the JSON form, borrowing strings from `value`.
    ///
    /// Parameters:
    /// - `arena`: holds the message arrays (free it with `value`'s arena).
    /// - `value`: the parsed `{"messages": [...]}` object.
    ///
    /// Return: the conversation; `error.InvalidConversation` on a malformed shape.
    pub fn fromJson(arena: std.mem.Allocator, value: std.json.Value) !Self {
        const messages_json = (if (value == .object) value.object.get("messages") else null) orelse return invalid("no messages");
        if (messages_json != .array) return invalid("messages is not an array");
        return fromMessages(arena, messages_json.array.items);
    }

    /// Converts a JSON array of `{role, content}` messages, borrowing strings.
    ///
    /// Parameters:
    /// - `arena`: holds the message array.
    /// - `list`: the messages.
    ///
    /// Return: the conversation; `error.InvalidConversation` on a malformed message.
    pub fn fromMessages(arena: std.mem.Allocator, list: []const std.json.Value) !Self {
        const messages = try arena.alloc(Message, list.len);
        for (list, messages) |m, *out| {
            if (m != .object) return invalid("message is not an object");
            const role_json = m.object.get("role") orelse return invalid("message without role");
            const content = m.object.get("content") orelse return invalid("message without content");
            if (role_json != .string) return invalid("role is not a string");
            const role = std.meta.stringToEnum(Role, role_json.string) orelse return invalid("unknown role");
            out.role = role;
            switch (content) {
                .string => |s| out.content = .{ .text = s },
                .array => |items| {
                    const parts = try arena.alloc(MessagePart, items.items.len);
                    for (items.items, parts) |p, *part| {
                        if (p != .object) return invalid("part is not an object");
                        const kind = p.object.get("type") orelse return invalid("part without type");
                        const text = p.object.get("text") orelse return invalid("part without text");
                        if (kind != .string or text != .string) return invalid("part fields are not strings");
                        part.* = .{ .kind = std.meta.stringToEnum(MessagePart.Kind, kind.string) orelse return invalid("unknown part type"), .text = text.string };
                    }
                    out.content = .{ .parts = parts };
                },
                else => return invalid("content is neither a string nor a list"),
            }
        }
        return Self{ .messages = messages };
    }

    /// Why the conversation cannot be trained on, or null when it can: an
    /// optional system message first (followed by a user message), then user
    /// and assistant messages alternating from the user, at least one reply;
    /// only assistant messages may have parts.
    ///
    /// Parameters:
    /// - `self`: the conversation.
    ///
    /// Return: the reason, or null.
    pub fn problem(self: Self) ?[]const u8 {
        var messages = self.messages;
        if (messages.len > 0 and messages[0].role == .system) {
            if (messages[0].content != .text) return "the system message must be text";
            messages = messages[1..];
        }
        if (messages.len < 2) return "needs a user message and an assistant reply";
        for (messages, 0..) |m, i| {
            const want: Role = if (i % 2 == 0) .user else .assistant;
            if (m.role != want) return if (want == .user) "messages must alternate user, assistant, starting with user" else "messages must alternate user, assistant";
            if (m.role == .user and m.content != .text) return "user messages must be text";
        }
        return null;
    }

    fn invalid(reason: []const u8) error{InvalidConversation} {
        log.debug("invalid conversation: {s}", .{reason});
        return error.InvalidConversation;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "conversation parses text and part contents" {
    const allocator = std.testing.allocator;
    const json =
        \\{"messages": [{"role": "user", "content": "hi"},
        \\  {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "python", "text": "1+1"}]}]}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const conv = try mod.Conversation.fromJson(arena.allocator(), parsed.value);
    try std.testing.expectEqual(@as(usize, 2), conv.messages.len);
    try std.testing.expectEqualStrings("hi", conv.messages[0].content.text);
    try std.testing.expectEqual(mod.MessagePart.Kind.python, conv.messages[1].content.parts[1].kind);

    const bad = try std.json.parseFromSlice(std.json.Value, allocator, "{\"messages\": [{\"role\": \"robot\", \"content\": \"x\"}]}", .{});
    defer bad.deinit();
    try std.testing.expectError(error.InvalidConversation, mod.Conversation.fromJson(arena.allocator(), bad.value));
}
