// ---------------------------------------------------------------------------
// zignanogpt: nanochat in Zig. The entry point only; it dispatches a
// subcommand to the library types re-exported by the barrels.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.zignanogpt_main);
const mod = @import("zignanogpt");
const cli = @import("module.zig");

/// `.info` compiles every function-entry trace out; `.debug` turns them on.
pub const std_options: std.Options = .{ .log_level = .info };

/// Runs the subcommand and flushes its output.
///
/// Parameters:
/// - `init`: runtime-supplied process state.
///
/// Return: the exit code: 0 on success, 1 for a command not implemented yet,
/// 2 for a usage error.
pub fn main(init: std.process.Init) !u8 {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_writer.interface;
    const code = try run(init, out);
    try out.flush();
    return code;
}

/// Parses the subcommand and runs it.
///
/// Parameters:
/// - `init`: runtime-supplied process state.
/// - `out`: where the command writes its output.
///
/// Return: the exit code, as for `main`.
fn run(init: std.process.Init, out: *std.Io.Writer) !u8 {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    var args = init.minimal.args.iterate();
    _ = args.skip(); // program name
    const name = args.next() orelse {
        try cli.Command.writeUsage(out);
        return 2;
    };
    const command = cli.Command.parse(name) orelse {
        try out.print("unknown command: {s}\n\n", .{name});
        try cli.Command.writeUsage(out);
        return 2;
    };

    if (command.phase()) |number| {
        try out.print("{s}: not implemented yet (PLAN.md phase {d})\n", .{ @tagName(command), number });
        return 1;
    }
    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(init.gpa);
    while (args.next()) |arg| try rest.append(init.gpa, arg);
    var options = try cli.Args.init(init.gpa, rest.items);
    defer options.deinit();
    switch (command) {
        .@"tok-train" => return cli.TokTrain.run(init, &options, out),
        .@"tok-eval" => return cli.TokEval.run(init, &options, out),
        .help => try cli.Command.writeUsage(out),
        .version => try out.print("zignanogpt {s} ({t} backend)\n", .{ mod.build_options.version, mod.build_options.backend }),
        .config => {
            var config = try mod.Config.init(init.gpa, init.environ_map);
            defer config.deinit();
            try out.print("base dir:     {s}\nnanochat dir: {s} (read-only)\n", .{ config.base_dir, config.nanochat_dir });
        },
        else => unreachable, // every other command has a phase
    }
    return 0;
}
