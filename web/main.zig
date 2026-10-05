// ---------------------------------------------------------------------------
// zignanogpt-web: the web console. One embedded page with a chat UI
// (streamed replies) and a training dashboard (tails metrics.jsonl).
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.zignanogpt_web);
const web = @import("module.zig");
const mod = web.nanogpt;

pub const std_options: std.Options = .{ .log_level = .info };

const usage =
    \\usage: zignanogpt-web [--host 127.0.0.1] [--port 8000] [-i <source>] [-g <tag>] [-s <step>] [--threads <n>]
    \\  -i, --source     base, sft or rl (default: sft, or base while there is none)
    \\  -g, --model-tag  (default: the largest d<N>)
    \\  -s, --step       (default: the last)
    \\  Without any checkpoint the dashboard still runs; chat reports that no model is loaded.
    \\
;

/// Loads the model and serves the console until interrupted.
///
/// Parameters:
/// - `init`: runtime-supplied process state.
///
/// Return: 0, or 2 for a usage error; setup errors.
pub fn main(init: std.process.Init) !u8 {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    const allocator = init.gpa;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_writer.interface;

    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8000;
    var source: ?mod.CheckpointKind = null;
    var tag: ?[]const u8 = null;
    var step: ?usize = null;
    var threads: usize = 0;
    var args = init.minimal.args.iterate();
    _ = args.skip();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try out.writeAll(usage);
            try out.flush();
            return 0;
        }
        const v = args.next() orelse return usageError(out);
        if (std.mem.eql(u8, arg, "--host")) {
            host = v;
        } else if (std.mem.eql(u8, arg, "--port")) {
            port = std.fmt.parseInt(u16, v, 10) catch return usageError(out);
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--source")) {
            source = std.meta.stringToEnum(mod.CheckpointKind, v) orelse return usageError(out);
        } else if (std.mem.eql(u8, arg, "-g") or std.mem.eql(u8, arg, "--model-tag")) {
            tag = v;
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--step")) {
            step = std.fmt.parseInt(usize, v, 10) catch return usageError(out);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            threads = std.fmt.parseInt(usize, v, 10) catch return usageError(out);
        } else return usageError(out);
    }

    const storage = mod.Storage.init(allocator, init.io);
    var config = try mod.Config.load(allocator, init.environ_map, storage);
    defer config.deinit();
    var backend = try mod.Backend.init(allocator, init.io, .{ .threads = threads });
    defer backend.deinit();

    // The model: the given source, or sft, falling back to base; none is fine.
    const loaded = try allocator.create(mod.LoadedModel);
    defer allocator.destroy(loaded);
    var model: ?*mod.LoadedModel = null;
    const kinds: []const mod.CheckpointKind = if (source) |s| &.{s} else &.{ .sft, .base };
    for (kinds) |kind| {
        loaded.init(allocator, &backend, storage, config.base_dir, .{ .kind = kind, .tag = tag, .step = step }) catch |err| switch (err) {
            error.NoCheckpoint => continue,
            else => return err,
        };
        model = loaded;
        break;
    }
    defer if (model) |m| m.deinit();
    if (model) |m| {
        try out.print("Loaded {s} model {s} step {d}\n", .{ @tagName(m.kind), m.tag, m.step });
    } else try out.writeAll("No checkpoint found: chat is unavailable, the dashboard still works\n");

    const address = try std.Io.net.IpAddress.parse(host, port);
    var listener = try address.listen(init.io, .{ .reuse_address = true });
    defer listener.deinit(init.io);
    try out.print("zignanogpt-web {s}: http://{s}:{d}/ (base dir {s})\n", .{ mod.build_options.version, host, listener.socket.address.getPort(), config.base_dir });
    try out.flush();
    var server = web.Server.init(allocator, init.io, config.base_dir, model);
    try server.serve(&listener);
    return 0;
}

fn usageError(out: *std.Io.Writer) !u8 {
    try out.writeAll(usage);
    try out.flush();
    return 2;
}
