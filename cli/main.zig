// ---------------------------------------------------------------------------
// zignanogpt: nanochat in Zig. The entry point only; it dispatches a
// subcommand to the library types re-exported by the barrels.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.zignanogpt_main);
const mod = @import("zignanogpt");
const cli = @import("module.zig");

/// `.info` compiles every function-entry trace out; `.debug` turns them on.
pub const std_options: std.Options = .{ .log_level = .info, .logFn = logFn };

pub const panic = cli.tui.Panic;

/// Logs normally, or into the console's running job while it holds the terminal.
fn logFn(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (!cli.Console.quiet()) return std.log.defaultLog(level, scope, format, args);
    var buf: [1024]u8 = undefined;
    const prefix = "[" ++ comptime level.asText() ++ "] ";
    const text = std.fmt.bufPrint(&buf, prefix ++ format ++ "\n", args) catch return;
    cli.Console.captureLog(text);
}

/// Runs the subcommand and flushes its output.
///
/// Parameters:
/// - `init`: runtime-supplied process state.
///
/// Return: the exit code: 0 on success, 1 for a failure (or a command not
/// implemented yet), 2 for a usage error.
pub fn main(init: std.process.Init) !u8 {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_writer.interface;
    const code = run(init, out) catch |err| {
        try out.flush();
        std.debug.print("error: {t}\n", .{err});
        return 1;
    };
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
        // Nothing asked for: the console on a terminal, the usage otherwise.
        if (try std.Io.File.stdout().isTty(init.io)) {
            try cli.Console.run(init, null);
            return 0;
        }
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
    switch (command) {
        .help => try cli.Command.writeUsage(out),
        .tui => try cli.Console.run(init, null),
        .version => try out.print("zignanogpt {s} ({t} backend)\n", .{ mod.build_options.version, mod.build_options.backend }),
        .config => {
            var config = try mod.Config.load(init.gpa, init.environ_map, mod.Storage.init(init.gpa, init.io));
            defer config.deinit();
            try out.print("base dir:     {s}\nnanochat dir: {s} (read-only)\ndata url:     {s}\n", .{ config.base_dir, config.nanochat_dir, config.data_url });
        },
        else => {
            var rest: std.ArrayList([]const u8) = .empty;
            defer rest.deinit(init.gpa);
            while (args.next()) |arg| try rest.append(init.gpa, arg);
            return cli.Runner.execute(init, command, rest.items, out, null);
        },
    }
    return 0;
}
