const std = @import("std");
const log = std.log.scoped(.zignanogpt_dataset);
const mod = @import("../module.zig");

/// The pretraining shards (nanochat's `dataset.py`): `shard_NNNNN.parquet`
/// files with a `text` column, sorted by name, the last one the validation
/// split. A dataset is named: `climbmix` (the default) is ClimbMix-400B as
/// `shard_00000` .. `shard_06542`, downloaded by `download`; others are made by
/// `repackage`. Shards live in `<base dir>/base_data_<name>`; a shard already
/// in the Python nanochat directory is used from there, never copied.
pub const Dataset = struct {
    const Self = @This();

    /// The last shard; always the validation split.
    pub const max_shard = 6542;
    /// The dataset `download` fetches and every command reads by default.
    pub const default_name = "climbmix";
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

    /// Resolves a dataset's shard directories from a config.
    ///
    /// Parameters:
    /// - `allocator`: owns the paths.
    /// - `io`: for storage and downloads.
    /// - `config`: base directories and the data URL (borrowed).
    /// - `name`: the dataset (`default_name` for ClimbMix).
    ///
    /// Return: the dataset; `error.InvalidDatasetName`, allocation errors.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, name: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const dir = try directory(allocator, config.base_dir, name);
        errdefer allocator.free(dir);
        const nanochat_dir = try directory(allocator, config.nanochat_dir, name);
        return Self{ .allocator = allocator, .io = io, .storage = mod.Storage.init(allocator, io), .dir = dir, .nanochat_dir = nanochat_dir, .url = config.data_url };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.nanochat_dir);
        self.allocator.free(self.dir);
    }

    /// Whether `name` can name a dataset: 1-64 of `A-Z a-z 0-9 _ -`.
    pub fn validName(name: []const u8) bool {
        if (name.len == 0 or name.len > 64) return false;
        for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-')) return false;
        return true;
    }

    /// A dataset's directory under a base directory: `<base>/base_data_<name>`.
    ///
    /// Parameters:
    /// - `allocator`: owns the result.
    /// - `base`: the base directory.
    /// - `name`: the dataset.
    ///
    /// Return: the path; `error.InvalidDatasetName`, allocation errors.
    pub fn directory(allocator: std.mem.Allocator, base: []const u8, name: []const u8) ![]u8 {
        if (!validName(name)) {
            log.warn("invalid dataset name '{s}': use 1-64 letters, digits, '_' or '-'", .{name});
            return error.InvalidDatasetName;
        }
        const leaf = try allocator.print("base_data_{s}", .{name});
        defer allocator.free(leaf);
        return std.Io.Dir.path.join(allocator, &.{ base, leaf });
    }

    /// The datasets under a base directory (`base_data_<name>` directories).
    ///
    /// Parameters:
    /// - `allocator`: owns the result (each name and the list).
    /// - `storage`: the file helper.
    /// - `base`: the base directory.
    ///
    /// Return: the names, sorted; empty when `base` is missing; storage errors.
    pub fn names(allocator: std.mem.Allocator, storage: mod.Storage, base: []const u8) ![][]const u8 {
        const entries = storage.list(allocator, base) catch |err| switch (err) {
            error.NotFound => return &.{},
            else => return err,
        };
        defer {
            for (entries) |e| allocator.free(e);
            allocator.free(entries);
        }
        var out: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (out.items) |n| allocator.free(n);
            out.deinit(allocator);
        }
        for (entries) |e| {
            if (!std.mem.endsWith(u8, e, "/")) continue;
            const leaf = std.Io.Dir.path.basename(std.mem.trimEnd(u8, e, "/"));
            if (!std.mem.startsWith(u8, leaf, "base_data_") or !validName(leaf["base_data_".len..])) continue;
            try out.append(allocator, try allocator.dupe(u8, leaf["base_data_".len..]));
        }
        return out.toOwnedSlice(allocator);
    }

    /// `shard_NNNNN.parquet`.
    pub fn shardName(buf: []u8, index: usize) ![]const u8 {
        return std.mem.print(buf, "shard_{d:0>5}.parquet", .{index});
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
                const name = url[(std.mem.findScalarLast(u8, url, '/') orelse continue) + 1 ..];
                if (!std.mem.endsWith(u8, name, ".parquet")) continue;
                var duplicate = false;
                for (paths.items) |p| duplicate = duplicate or std.mem.eql(u8, std.Io.Dir.path.basename(p), name);
                if (duplicate) continue;
                try paths.append(allocator, try std.Io.Dir.path.join(allocator, &.{ dir, name }));
            }
        }
        std.mem.sort([]const u8, paths.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, std.Io.Dir.path.basename(a), std.Io.Dir.path.basename(b));
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
        const path = try std.Io.Dir.path.join(self.allocator, &.{ self.dir, name });
        defer self.allocator.free(path);
        const existing = try std.Io.Dir.path.join(self.allocator, &.{ self.nanochat_dir, name });
        defer self.allocator.free(existing);
        if (try self.storage.exists(path) or try self.storage.exists(existing)) {
            try out.print("Skipping {s} (already exists)\n", .{name});
            return false;
        }
        const url = try self.allocator.print("{s}/{s}", .{ self.url, name });
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
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    const files = [_][]const u8{ "zig/base_data_climbmix/shard_00002.parquet", "zig/base_data_climbmix/shard_00000.parquet", "py/base_data_climbmix/shard_00001.parquet", "py/base_data_climbmix/shard_00002.parquet", "py/base_data_climbmix/notes.txt" };
    for (files) |f| {
        const p = try std.Io.Dir.path.join(allocator, &.{ root, f });
        defer allocator.free(p);
        try storage.write(p, "PAR1");
    }
    const zig_dir = try std.Io.Dir.path.join(allocator, &.{ root, "zig" });
    defer allocator.free(zig_dir);
    const py_dir = try std.Io.Dir.path.join(allocator, &.{ root, "py" });
    defer allocator.free(py_dir);
    const config = mod.Config{ .allocator = allocator, .base_dir = zig_dir, .nanochat_dir = py_dir, .data_url = "http://unused" };
    try std.testing.expectError(error.InvalidDatasetName, mod.Dataset.init(allocator, std.testing.io, &config, "../up"));
    var dataset = try mod.Dataset.init(allocator, std.testing.io, &config, mod.Dataset.default_name);
    defer dataset.deinit();
    const paths = try dataset.list(allocator);
    defer {
        for (paths) |p| allocator.free(p);
        allocator.free(paths);
    }
    try std.testing.expectEqual(@as(usize, 3), paths.len);
    try std.testing.expectEqualStrings("shard_00000.parquet", std.Io.Dir.path.basename(paths[0]));
    try std.testing.expect(std.mem.find(u8, paths[1], "/py/") != null);
    try std.testing.expect(std.mem.find(u8, paths[2], "/zig/") != null); // this port's copy wins
}
