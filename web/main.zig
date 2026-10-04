// ---------------------------------------------------------------------------
// zignanogpt-web: the web console (chat UI + training dashboard). A stub
// until PLAN.md phase 11.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.zignanogpt_web);
const web = @import("module.zig");

pub const std_options: std.Options = .{ .log_level = .info };

/// Reports that the console is not built yet.
///
/// Parameters:
/// - `init`: runtime-supplied process state.
///
/// Return: exit code 1.
pub fn main(init: std.process.Init) !u8 {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_writer.interface;
    try out.print("zignanogpt-web {s}: not implemented yet (PLAN.md phase 11)\n", .{web.nanogpt.build_options.version});
    try out.flush();
    return 1;
}
