const std = @import("std");
const log = std.log.scoped(.zignanogpt_storage);
const mod = @import("../module.zig");

/// A file being written: appended in pieces, invisible until `save` commits it
/// atomically. `deinit` without `save` discards it.
pub const StorageSession = struct {
    const Self = @This();

    node: mod.zigstorage.Node,
    /// Bytes appended so far.
    written: u64 = 0,

    /// Appends to the staged contents.
    ///
    /// Parameters:
    /// - `self`: the session.
    /// - `bytes`: what to append.
    ///
    /// Return: nothing; storage errors.
    pub fn append(self: *Self, bytes: []const u8) !void {
        try self.node.append(bytes);
        self.written += bytes.len;
    }

    /// Commits the file atomically.
    pub fn save(self: *Self) !void {
        try self.node.save();
    }

    /// Ends the session (discarding it unless saved) and frees the node.
    pub fn deinit(self: *Self) void {
        self.node.close();
        self.node.deinit();
    }
};

/// Whole-file reads and atomic writes over zigstorage, for the port's own files
/// (tokenizer, checkpoints, metrics, repackaged shards). Paths are absolute
/// filesystem paths or URLs.
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

    /// Starts writing a file in pieces, creating its directory first. Nothing
    /// is visible at `path` until the session's `save`.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    ///
    /// Return: the open session (the caller deinits it); storage errors.
    pub fn create(self: Self, path_in: []const u8) !StorageSession {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
        if (std.fs.path.dirname(path)) |dir| try self.makeDir(dir);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        errdefer node.deinit();
        try node.open();
        return .{ .node = node };
    }

    /// Reads up to `length` bytes at `offset` (fewer at the end of the file).
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    /// - `offset`: the first byte.
    /// - `length`: the most bytes to read.
    ///
    /// Return: the bytes, owned by the caller; storage errors.
    pub fn readRange(self: Self, path_in: []const u8, offset: u64, length: u64) ![]u8 {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        return node.read(.{ .offset = offset, .length = length });
    }

    /// A file's size in bytes.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    ///
    /// Return: the size; storage errors (`error.NotFound`).
    pub fn size(self: Self, path_in: []const u8) !u64 {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        try node.fetchAttributes();
        return node.size orelse error.UnknownSize;
    }

    /// Deletes a file.
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `path`: the file.
    ///
    /// Return: nothing; storage errors (`error.NotFound` when absent).
    pub fn remove(self: Self, path_in: []const u8) !void {
        const path = try self.absolute(path_in);
        defer self.allocator.free(path);
        log.debug("removing {s}", .{path});
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, path);
        defer node.deinit();
        try node.remove();
    }

    /// Deletes a directory and everything in it (links are unlinked, never followed).
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `dir`: the directory.
    ///
    /// Return: nothing; storage errors (`error.NotFound` when absent).
    pub fn removeTree(self: Self, dir_in: []const u8) !void {
        const dir = try self.absolute(dir_in);
        defer self.allocator.free(dir);
        log.debug("removing tree {s}", .{dir});
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, dir);
        defer node.deinit();
        try node.removeTree();
    }

    /// The entries of a directory (files and subdirectories, not recursive).
    ///
    /// Parameters:
    /// - `self`: the helper.
    /// - `allocator`: owns the result (each path and the list).
    /// - `dir`: the directory.
    ///
    /// Return: absolute paths sorted by name (a directory's ends in `/`);
    /// storage errors (`error.NotFound` when absent).
    pub fn list(self: Self, allocator: std.mem.Allocator, dir_in: []const u8) ![][]const u8 {
        const dir = try self.absolute(dir_in);
        defer self.allocator.free(dir);
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, dir);
        defer node.deinit();
        var paths: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (paths.items) |p| allocator.free(p);
            paths.deinit(allocator);
        }
        var it = try node.list();
        defer it.close();
        while (try it.next()) |url| {
            const trimmed = std.mem.trimEnd(u8, url, "/");
            const name = trimmed[(std.mem.lastIndexOfScalar(u8, trimmed, '/') orelse continue) + 1 ..];
            const suffix: []const u8 = if (trimmed.len < url.len) "/" else "";
            try paths.append(allocator, try std.mem.concat(allocator, u8, &.{ std.mem.trimEnd(u8, dir, "/"), "/", name, suffix }));
        }
        std.mem.sort([]const u8, paths.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);
        return paths.toOwnedSlice(allocator);
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

test "storage sessions stage until saved; ranges, sizes, listings and removal" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    const path = try std.fs.path.join(allocator, &.{ root, "d", "staged.bin" });
    defer allocator.free(path);

    var dropped = try storage.create(path);
    try dropped.append("never saved");
    dropped.deinit();
    try std.testing.expect(!try storage.exists(path));

    var session = try storage.create(path);
    try session.append("0123");
    try session.append("456789");
    try std.testing.expectEqual(@as(u64, 10), session.written);
    try session.save();
    session.deinit();
    try std.testing.expectEqual(@as(u64, 10), try storage.size(path));
    const middle = try storage.readRange(path, 3, 4);
    defer allocator.free(middle);
    try std.testing.expectEqualStrings("3456", middle);
    const tail = try storage.readRange(path, 8, 100);
    defer allocator.free(tail);
    try std.testing.expectEqualStrings("89", tail);

    const other = try std.fs.path.join(allocator, &.{ root, "d", "a.txt" });
    defer allocator.free(other);
    try storage.write(other, "a");
    const sub = try std.fs.path.join(allocator, &.{ root, "d", "sub" });
    defer allocator.free(sub);
    try storage.makeDir(sub);
    const dir = try std.fs.path.join(allocator, &.{ root, "d" });
    defer allocator.free(dir);
    const entries = try storage.list(allocator, dir);
    defer {
        for (entries) |e| allocator.free(e);
        allocator.free(entries);
    }
    try std.testing.expectEqual(@as(usize, 3), entries.len);
    try std.testing.expectEqualStrings(other, entries[0]);
    try std.testing.expect(std.mem.endsWith(u8, entries[2], "/sub/"));
    try storage.remove(path);
    try std.testing.expect(!try storage.exists(path));
    try std.testing.expectError(error.NotFound, storage.remove(path));
    try std.testing.expectError(error.NotADirectory, storage.list(allocator, other));
    try storage.removeTree(dir);
    try std.testing.expect(!try storage.exists(other));
}
