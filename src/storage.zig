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

    /// An absolute form of `path`: URLs and absolute paths unchanged, relative
    /// paths resolved against the working directory (zigstorage would read a
    /// scheme-less relative path as a URL host).
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: a path or URL.
    ///
    /// Return: the absolute form, owned by the caller.
    pub fn absolute(self: Self, path: []const u8) ![]u8 {
        if (std.fs.path.isAbsolute(path) or std.mem.indexOf(u8, path, "://") != null) return self.allocator.dupe(u8, path);
        const cwd = try std.process.currentPathAlloc(self.io, self.allocator);
        defer self.allocator.free(cwd);
        return std.fs.path.resolve(self.allocator, &.{ cwd, path });
    }

    /// Reads a whole file.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    ///
    /// Return: the bytes, owned by the caller; `error.NotFound` and other storage errors.
    pub fn read(self: Self, path_in: []const u8) ![]u8 {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
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
    pub fn write(self: Self, path_in: []const u8, bytes: []const u8) !void {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
        log.debug("writing {d} bytes to {s}", .{ bytes.len, path });
        if (std.fs.path.dirname(path)) |dir| try self.makeDir(dir);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        try node.open();
        try node.append(bytes);
        try node.save();
    }

    /// Appends to a file, creating it (and its directory) first if needed.
    /// Not atomic: for logs such as `metrics.jsonl`.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    /// - `bytes`: what to append.
    ///
    /// Return: nothing; storage errors.
    pub fn append(self: Self, path_in: []const u8, bytes: []const u8) !void {
        if (!try self.exists(path_in)) return self.write(path_in, bytes);
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        try node.append(bytes);
    }

    /// Creates a directory and its parents (succeeds if it exists).
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the directory.
    ///
    /// Return: nothing; storage errors.
    pub fn makeDir(self: Self, path_in: []const u8) !void {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
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
    pub fn exists(self: Self, path_in: []const u8) !bool {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
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
    try storage.append(path, "+more");
    try std.testing.expect(try storage.exists(path));
    const bytes = try storage.read(path);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("replaced+more", bytes);
    const rel = try storage.absolute("x/../y.txt");
    defer allocator.free(rel);
    try std.testing.expect(std.fs.path.isAbsolute(rel) and std.mem.endsWith(u8, rel, "/y.txt"));
    const url = try storage.absolute("https://h/p");
    defer allocator.free(url);
    try std.testing.expectEqualStrings("https://h/p", url);
}
