const std = @import("std");
const log = std.log.scoped(.zignanogpt_config);
const mod = @import("module.zig");

/// Runtime configuration: where the port reads and writes its artifacts.
///
/// Two directories, both resolved from the environment:
/// - `base_dir`: everything this port writes (tokenizer, checkpoints, downloads,
///   run logs). `$ZIGNANOGPT_BASE_DIR`, else `~/.cache/zignanogpt`.
/// - `nanochat_dir`: the Python nanochat's base dir, read-only (ClimbMix shards,
///   Python checkpoints and tokenizer). `$NANOCHAT_BASE_DIR`, else
///   `~/.cache/nanochat`, matching `nanochat/common.py:get_base_dir`.
pub const Config = struct {
    const Self = @This();

    pub const base_dir_env = "ZIGNANOGPT_BASE_DIR";
    pub const nanochat_dir_env = "NANOCHAT_BASE_DIR";

    allocator: std.mem.Allocator,
    /// Directory this port writes into.
    base_dir: []const u8,
    /// The Python nanochat directory, never written.
    nanochat_dir: []const u8,

    /// Resolves both directories from `environ`.
    ///
    /// An override variable that is set but empty counts as unset, as in Python.
    /// The home directory is `$HOME`, falling back to `$USERPROFILE` (Windows).
    ///
    /// Parameters:
    /// - `allocator`: owns the resolved paths until `deinit`.
    /// - `environ`: the process environment.
    ///
    /// Return: the config; `error.HomeNotFound` when a default is needed and no
    /// home directory is set.
    pub fn init(allocator: std.mem.Allocator, environ: *const std.process.Environ.Map) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const home = nonEmpty(environ.get("HOME")) orelse nonEmpty(environ.get("USERPROFILE"));
        const base_dir = try resolve(allocator, nonEmpty(environ.get(base_dir_env)), home, "zignanogpt");
        errdefer allocator.free(base_dir);
        const nanochat_dir = try resolve(allocator, nonEmpty(environ.get(nanochat_dir_env)), home, "nanochat");
        return Self{
            .allocator = allocator,
            .base_dir = base_dir,
            .nanochat_dir = nanochat_dir,
        };
    }

    /// Frees the resolved paths.
    ///
    /// Parameters:
    /// - `self`: the config.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.nanochat_dir);
        self.allocator.free(self.base_dir);
    }

    /// Picks the override, else `<home>/.cache/<leaf>`.
    ///
    /// Parameters:
    /// - `allocator`: allocates the returned path.
    /// - `override`: the environment override, if set.
    /// - `home`: the home directory, if known.
    /// - `leaf`: the directory name under `~/.cache`.
    ///
    /// Return: the path, owned by the caller.
    fn resolve(allocator: std.mem.Allocator, override: ?[]const u8, home: ?[]const u8, leaf: []const u8) ![]const u8 {
        if (override) |path| return allocator.dupe(u8, path);
        const home_dir = home orelse return error.HomeNotFound;
        return std.fs.path.join(allocator, &.{ home_dir, ".cache", leaf });
    }

    /// Treats an empty value as absent.
    ///
    /// Parameters:
    /// - `value`: an environment value.
    ///
    /// Return: `value` when non-empty, else null.
    fn nonEmpty(value: ?[]const u8) ?[]const u8 {
        const text = value orelse return null;
        return if (text.len == 0) null else text;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "config defaults to ~/.cache" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put("HOME", "/home/u");

    var config = try mod.Config.init(std.testing.allocator, &environ);
    defer config.deinit();
    try std.testing.expectEqualStrings("/home/u/.cache/zignanogpt", config.base_dir);
    try std.testing.expectEqualStrings("/home/u/.cache/nanochat", config.nanochat_dir);
}

test "config honours overrides and ignores empty ones" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put("HOME", "/home/u");
    try environ.put(mod.Config.base_dir_env, "/data/zig");
    try environ.put(mod.Config.nanochat_dir_env, "");

    var config = try mod.Config.init(std.testing.allocator, &environ);
    defer config.deinit();
    try std.testing.expectEqualStrings("/data/zig", config.base_dir);
    try std.testing.expectEqualStrings("/home/u/.cache/nanochat", config.nanochat_dir);
}

test "config falls back to USERPROFILE" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put("USERPROFILE", "/users/u");

    var config = try mod.Config.init(std.testing.allocator, &environ);
    defer config.deinit();
    try std.testing.expectEqualStrings("/users/u/.cache/nanochat", config.nanochat_dir);
}

test "config without a home needs both overrides" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try std.testing.expectError(error.HomeNotFound, mod.Config.init(std.testing.allocator, &environ));

    try environ.put(mod.Config.base_dir_env, "/a");
    try environ.put(mod.Config.nanochat_dir_env, "/b");
    var config = try mod.Config.init(std.testing.allocator, &environ);
    defer config.deinit();
    try std.testing.expectEqualStrings("/a", config.base_dir);
}
