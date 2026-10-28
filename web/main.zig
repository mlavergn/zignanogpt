// ---------------------------------------------------------------------------
// zignanogpt-web: the web console. A zigwui server for the Svelte page in
// views/: a chat UI (streamed replies) and a training dashboard (tails
// metrics.jsonl).
// ---------------------------------------------------------------------------

const std = @import("std");
const builtin = @import("builtin");
const log = std.log.scoped(.zignanogpt_web);
const zigwui = @import("zigwui");
const bundle = @import("assets");
const web = @import("module.zig");
const mod = web.nanogpt;

pub const std_options: std.Options = .{ .log_level = .info };

const usage =
    \\usage: zignanogpt-web [--port 8000] [--no-browser] [-i <source>] [-g <tag>] [-s <step>] [--threads <n>]
    \\  --port           listens on 127.0.0.1 only (reach it from elsewhere through an SSH tunnel)
    \\  --no-browser     do not open the page in a browser
    \\  -i, --source     base, sft or rl (default: sft, or base while there is none)
    \\  -g, --model-tag  (default: the largest d<N>)
    \\  -s, --step       (default: the last)
    \\  Without any checkpoint the dashboard still runs; chat reports that no model is loaded.
    \\
;

/// Shown for `--host`, which the console no longer takes.
const host_gone = "the web console listens on 127.0.0.1 only; --host is gone (use an SSH tunnel to reach it from elsewhere)\n";

/// What `make views` builds, named in the banner when the bundle is missing.
const build_hint = "Build the views (`make views`), then rebuild.";

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

    var port: u16 = 8000;
    var open_browser = true;
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
        if (std.mem.eql(u8, arg, "--no-browser")) {
            open_browser = false;
            continue;
        }
        if (std.mem.eql(u8, arg, "--host")) {
            try out.writeAll(host_gone);
            return usageError(out);
        }
        const v = args.next() orelse return usageError(out);
        if (std.mem.eql(u8, arg, "--port")) {
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

    var assets = try zigwui.AssetStore.init(allocator, init.io, null, &bundle.files);
    defer assets.deinit();
    var handler = web.Handler.init(allocator, init.io, config.base_dir, model, port);
    var server = try zigwui.Server(web.Handler).init(allocator, init.io, &handler, &assets, .{
        .port = port,
        .open_browser = false,
        .name = "zignanogpt-web",
        .build_hint = build_hint,
    });
    defer server.deinit();
    server.bind() catch |err| {
        log.err("Cannot listen on {f} [{t}]", .{ server.address, err });
        return err;
    };

    const url = try server.urlOwned();
    defer allocator.free(url);
    try out.print("zignanogpt-web {s}: {s} (base dir {s})\n", .{ mod.build_options.version, url, config.base_dir });
    if (!assets.isPopulated()) try out.print("warning: no web assets found. {s}\n", .{build_hint});
    try out.flush();
    if (open_browser and hasDisplay(init.minimal.environ)) server.openInBrowser(url);
    try server.serve();
    return 0;
}

/// Whether a browser can be shown: always but on Linux and the BSDs, where a
/// session with neither X nor Wayland (over SSH to the Spark, say) has none.
fn hasDisplay(environ: std.process.Environ) bool {
    return switch (builtin.os.tag) {
        .macos, .windows => true,
        else => environ.getPosix("DISPLAY") != null or environ.getPosix("WAYLAND_DISPLAY") != null,
    };
}

fn usageError(out: *std.Io.Writer) !u8 {
    try out.writeAll(usage);
    try out.flush();
    return 2;
}
