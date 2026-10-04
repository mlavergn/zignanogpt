// ---------------------------------------------------------------------------
// The web console barrel: every public type in web/ plus the library.
// The console itself (chat UI + training dashboard) lands in PLAN.md phase 11.
// ---------------------------------------------------------------------------

const std = @import("std");

pub const nanogpt = @import("zignanogpt");

test {
    std.testing.refAllDecls(@This());
}
