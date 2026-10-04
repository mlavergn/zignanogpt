const std = @import("std");
const log = std.log.scoped(.zignanogpt_storage);
const mod = @import("module.zig");

/// Whole-file reads and atomic writes over zigstorage, for the port's own files
/// (tokenizer, checkpoints, metrics). Paths are absolute filesystem paths or URLs.
pub const Storage = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,

    /// Creates the helper.
    ///
    /// Parameters:
    /// - `allocator`: allocates read results and node state.
    /// - `io`: the Io every operation runs on.
    ///
    /// Return: the helper.
    pub fn init(allocator: std.mem.Allocator, io: std.Io) Self {
        return Self{ .allocator = allocator, .io = io };
    }

    /// Reads a whole file.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    ///
    /// Return: the bytes, owned by the caller; `error.NotFound` and other storage errors.
    pub fn read(self: Self, path: []const u8) ![]u8 {
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        return node.read(.all) catch |err| {
            log.debug("cannot read {s} [{t}]", .{ path, err });
            return err;
        };
    }

    /// Replaces a file atomically, creating its directory first.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    /// - `bytes`: the new contents.
    ///
    /// Return: nothing; storage errors.
    pub fn write(self: Self, path: []const u8, bytes: []const u8) !void {
        log.debug("writing {d} bytes to {s}", .{ bytes.len, path });
        if (std.fs.path.dirname(path)) |dir| try self.makeDir(dir);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        try node.open();
        try node.append(bytes);
        try node.save();
    }

    /// Creates a directory and its parents (succeeds if it exists).
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the directory.
    ///
    /// Return: nothing; storage errors.
    pub fn makeDir(self: Self, path: []const u8) !void {
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        try node.createDirectory(null);
    }

    /// Whether a file exists.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    ///
    /// Return: true when present; storage errors other than absence.
    pub fn exists(self: Self, path: []const u8) !bool {
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        return node.exists();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "storage writes atomically into a new directory and reads back" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(std.testing.io, &root_buf);
    const path = try std.fs.path.join(allocator, &.{ root_buf[0..root_len], "a", "b", "file.txt" });
    defer allocator.free(path);

    const storage = mod.Storage.init(allocator, std.testing.io);
    try std.testing.expect(!try storage.exists(path));
    try storage.write(path, "hello");
    try storage.write(path, "replaced");
    try std.testing.expect(try storage.exists(path));
    const bytes = try storage.read(path);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("replaced", bytes);
}
