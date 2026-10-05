const std = @import("std");
const log = std.log.scoped(.zignanogpt_dataset);
const mod = @import("../module.zig");

/// The pretraining shards (nanochat's `dataset.py`): ClimbMix-400B as
/// `shard_00000.parquet` .. `shard_06542.parquet`, the last one the
/// validation split. Shards live in `<base dir>/base_data_climbmix`; a shard
/// already in the Python nanochat directory is used from there, never copied.
pub const Dataset = struct {
    const Self = @This();

    /// The last shard; always the validation split.
    pub const max_shard = 6542;
    pub const dir_name = "base_data_climbmix";
    /// Download attempts per shard, with 2^attempt seconds between them.
    pub const max_attempts = 5;

    allocator: std.mem.Allocator,
    io: std.Io,
    storage: mod.Storage,
    /// Where this port keeps (and downloads) shards.
    dir: []const u8,
    /// The Python nanochat shard directory, read-only.
    nanochat_dir: []const u8,
    url: []const u8,

    /// Resolves the shard directories from a config.
    ///
    /// Parameters:
    /// - `allocator`: owns the paths.
    /// - `io`: for storage and downloads.
    /// - `config`: base directories and the data URL (borrowed).
    ///
    /// Return: the dataset; allocation errors.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const dir = try std.fs.path.join(allocator, &.{ config.base_dir, dir_name });
        errdefer allocator.free(dir);
        const nanochat_dir = try std.fs.path.join(allocator, &.{ config.nanochat_dir, dir_name });
        return Self{ .allocator = allocator, .io = io, .storage = mod.Storage.init(allocator, io), .dir = dir, .nanochat_dir = nanochat_dir, .url = config.data_url };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.nanochat_dir);
        self.allocator.free(self.dir);
    }

    /// `shard_NNNNN.parquet`.
    pub fn shardName(buf: []u8, index: usize) ![]const u8 {
        return std.fmt.bufPrint(buf, "shard_{d:0>5}.parquet", .{index});
    }

    /// Every local shard, sorted by name, from both directories (this port's
    /// copy wins a name clash). Train = all but the last, val = the last.
    ///
    /// Parameters:
    /// - `self`: the dataset.
    /// - `allocator`: owns the result (each path and the list).
    ///
    /// Return: the paths; storage errors.
    pub fn list(self: *const Self, allocator: std.mem.Allocator) ![][]const u8 {
        var paths: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (paths.items) |p| allocator.free(p);
            paths.deinit(allocator);
        }
        for ([_][]const u8{ self.dir, self.nanochat_dir }) |dir| {
            var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, dir);
            defer node.deinit();
            var it = node.list() catch |err| switch (err) {
                error.NotFound => continue,
                else => return err,
            };
            defer it.close();
            while (try it.next()) |url| {
                const name = url[(std.mem.lastIndexOfScalar(u8, url, '/') orelse continue) + 1 ..];
                if (!std.mem.endsWith(u8, name, ".parquet")) continue;
                var duplicate = false;
                for (paths.items) |p| duplicate = duplicate or std.mem.eql(u8, std.fs.path.basename(p), name);
                if (duplicate) continue;
                try paths.append(allocator, try std.fs.path.join(allocator, &.{ dir, name }));
            }
        }
        std.mem.sort([]const u8, paths.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, std.fs.path.basename(a), std.fs.path.basename(b));
            }
        }.lessThan);
        return paths.toOwnedSlice(allocator);
    }

    /// Downloads one shard unless a copy exists in either directory.
    /// The body is checked to be parquet before it is committed atomically.
    ///
    /// Parameters:
    /// - `self`: the dataset.
    /// - `index`: the shard number.
    /// - `out`: progress messages.
    ///
    /// Return: true if downloaded, false if already present; the last error after `max_attempts`.
    pub fn download(self: *const Self, index: usize, out: *std.Io.Writer) !bool {
        var name_buf: [32]u8 = undefined;
        const name = try shardName(&name_buf, index);
        const path = try std.fs.path.join(self.allocator, &.{ self.dir, name });
        defer self.allocator.free(path);
        const existing = try std.fs.path.join(self.allocator, &.{ self.nanochat_dir, name });
        defer self.allocator.free(existing);
        if (try self.storage.exists(path) or try self.storage.exists(existing)) {
            try out.print("Skipping {s} (already exists)\n", .{name});
            return false;
        }
        const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.url, name });
        defer self.allocator.free(url);
        var attempt: u6 = 1;
        while (true) : (attempt += 1) {
            if (self.fetch(url, path)) {
                try out.print("Downloaded {s}\n", .{name});
                return true;
            } else |err| {
                try out.print("Attempt {d}/{d} failed for {s}: {t}\n", .{ attempt, max_attempts, name, err });
                try out.flush();
                if (attempt >= max_attempts) return err;
                try self.io.sleep(.fromSeconds(@as(i64, 1) << attempt), .awake);
            }
        }
    }

    fn fetch(self: *const Self, url: []const u8, path: []const u8) !void {
        var node = try mod.zigstorage.Node.init(self.allocator, self.io, .empty, url);
        defer node.deinit();
        const body = try node.read(.all);
        defer self.allocator.free(body);
        if (body.len < 12 or !std.mem.eql(u8, body[0..4], "PAR1") or !std.mem.eql(u8, body[body.len - 4 ..], "PAR1")) {
            log.warn("{s} did not return a parquet file ({d} bytes)", .{ url, body.len });
            return error.InvalidParquet;
        }
        try self.storage.write(path, body);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "dataset lists shards from both directories, sorted, without duplicates" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    const files = [_][]const u8{ "zig/base_data_climbmix/shard_00002.parquet", "zig/base_data_climbmix/shard_00000.parquet", "py/base_data_climbmix/shard_00001.parquet", "py/base_data_climbmix/shard_00002.parquet", "py/base_data_climbmix/notes.txt" };
    for (files) |f| {
        const p = try std.fs.path.join(allocator, &.{ root, f });
        defer allocator.free(p);
        try storage.write(p, "PAR1");
    }
    const zig_dir = try std.fs.path.join(allocator, &.{ root, "zig" });
    defer allocator.free(zig_dir);
    const py_dir = try std.fs.path.join(allocator, &.{ root, "py" });
    defer allocator.free(py_dir);
    const config = mod.Config{ .allocator = allocator, .base_dir = zig_dir, .nanochat_dir = py_dir, .data_url = "http://unused" };
    var dataset = try mod.Dataset.init(allocator, std.testing.io, &config);
    defer dataset.deinit();
    const paths = try dataset.list(allocator);
    defer {
        for (paths) |p| allocator.free(p);
        allocator.free(paths);
    }
    try std.testing.expectEqual(@as(usize, 3), paths.len);
    try std.testing.expectEqualStrings("shard_00000.parquet", std.fs.path.basename(paths[0]));
    try std.testing.expect(std.mem.indexOf(u8, paths[1], "/py/") != null);
    try std.testing.expect(std.mem.indexOf(u8, paths[2], "/zig/") != null); // this port's copy wins
}
