// ---------------------------------------------------------------------------
// The web console barrel: every public type in web/ plus the library.
// ---------------------------------------------------------------------------

const std = @import("std");

pub const nanogpt = @import("zignanogpt");

pub const ChatMessage = @import("chat_stream.zig").ChatMessage;
pub const ChatRequest = @import("chat_stream.zig").ChatRequest;
pub const ChatStream = @import("chat_stream.zig").ChatStream;
pub const Run = @import("runs.zig").Run;
pub const Runs = @import("runs.zig").Runs;
pub const Server = @import("server.zig").Server;
pub const Tail = @import("runs.zig").Tail;

test {
    std.testing.refAllDecls(@This());
}
