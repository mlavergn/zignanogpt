const std = @import("std");
const log = std.log.scoped(.zignanogpt_hub_dataset);
const mod = @import("module.zig");

/// One loaded column of a `HubDataset`: all shards' values, and where each row's start.
pub const HubColumn = struct {
    info: mod.ParquetColumn,
    values: mod.ParquetValues = .{},
    /// `rows + 1` offsets into the values (`ParquetValues.rowStarts`).
    starts: []usize = &.{},
};

/// A Hugging Face dataset split as nanochat's `load_hub_dataset` sees it: the
/// hub's parquet export, downloaded once into
/// `<base>/task_data/<owner>--<name>/<subset>/<split>/NNNNN.parquet` with a
/// `manifest.json` written last, shards concatenated in manifest order, and an
/// optional `shuffle(seed)` that reproduces `datasets.Dataset.shuffle` (numpy's
/// `default_rng(seed).permutation`). Only the requested columns are loaded,
/// whole, into memory.
pub const HubDataset = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    columns: []HubColumn,
    rows: usize,
    /// Logical row -> physical row, after `shuffle`.
    permutation: ?[]u32 = null,

    /// Where the hub's export of a split lists its parquet files.
    pub const api_url = "https://huggingface.co/api/datasets";

    /// Finds (in this port's base dir, then nanochat's) or downloads a split, then loads columns.
    ///
    /// Parameters:
    /// - `allocator`: owns the data.
    /// - `io`: file and network access.
    /// - `config`: the base directories.
    /// - `repo`: e.g. `cais/mmlu`.
    /// - `subset`: e.g. `all` (`default` for datasets without subsets).
    /// - `split`: e.g. `test`.
    /// - `names`: the leaf columns to load (logical paths, e.g. `choices.text`).
    /// - `out`: download progress, or null.
    ///
    /// Return: the dataset; download, storage and parquet errors, `error.MissingColumn`.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, repo: []const u8, subset: []const u8, split: []const u8, names: []const []const u8, out: ?*std.Io.Writer) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const storage = mod.Storage.init(allocator, io);
        const dir = try findOrDownload(allocator, io, storage, config, repo, subset, split, out);
        defer allocator.free(dir);
        return load(allocator, io, storage, dir, names);
    }

    /// Loads columns from a split directory holding `manifest.json` and its shards.
    ///
    /// Parameters:
    /// - `allocator`: owns the data.
    /// - `io`: file access.
    /// - `storage`: reads the manifest.
    /// - `dir`: the split directory.
    /// - `names`: the leaf columns to load.
    ///
    /// Return: the dataset; storage and parquet errors.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, storage: mod.Storage, dir: []const u8, names: []const []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const manifest_path = try std.fs.path.join(allocator, &.{ dir, "manifest.json" });
        defer allocator.free(manifest_path);
        const manifest = try storage.read(manifest_path);
        defer allocator.free(manifest);
        const files = try std.json.parseFromSlice([]const []const u8, allocator, manifest, .{});
        defer files.deinit();

        const columns = try allocator.alloc(HubColumn, names.len);
        var loaded: usize = 0;
        errdefer {
            for (columns[0..loaded]) |*c| {
                c.values.deinit(allocator);
                allocator.free(c.starts);
            }
            allocator.free(columns);
        }
        for (columns) |*c| {
            c.* = .{ .info = undefined };
            loaded += 1;
        }
        var rows: usize = 0;
        for (files.value) |name| {
            const path = try std.fs.path.join(allocator, &.{ dir, name });
            defer allocator.free(path);
            const location = try storage.absolute(path);
            defer allocator.free(location);
            var file = try mod.ParquetFile.open(allocator, io, location);
            defer file.deinit();
            for (columns, names) |*c, column_name| {
                const index = try file.column(column_name);
                c.info = file.columns[index];
                for (0..file.row_groups.len) |rg| try file.readColumn(rg, index, &c.values);
            }
            rows += std.math.cast(usize, file.num_rows) orelse return error.InvalidParquet;
        }
        for (columns) |*c| {
            c.starts = try c.values.rowStarts(allocator, c.info);
            if (c.starts.len != rows + 1) {
                log.warn("{s}: column {s} has {d} rows, expected {d}", .{ dir, c.info.name, c.starts.len - 1, rows });
                return error.InvalidParquet;
            }
        }
        log.debug("loaded {d} rows from {s}", .{ rows, dir });
        return Self{ .allocator = allocator, .columns = columns, .rows = rows };
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (self.columns) |*c| {
            c.values.deinit(self.allocator);
            self.allocator.free(c.starts);
        }
        self.allocator.free(self.columns);
        if (self.permutation) |p| self.allocator.free(p);
    }

    /// Reorders the rows as `datasets.Dataset.shuffle(seed=seed)` does.
    pub fn shuffle(self: *Self, seed: u64) !void {
        var rng = mod.NumpyRandom.init(seed);
        const p = try rng.permutation(self.allocator, self.rows);
        if (self.permutation) |old| self.allocator.free(old);
        self.permutation = p;
    }

    pub fn len(self: *const Self) usize {
        return self.rows;
    }

    /// A string column's value at a (shuffled) row; null when absent.
    ///
    /// Parameters:
    /// - `self`: the dataset.
    /// - `col`: the index in the `names` it was loaded with.
    /// - `index`: the row.
    ///
    /// Return: the string (borrowed), or null.
    pub fn string(self: *const Self, col: usize, index: usize) ?[]const u8 {
        const c = &self.columns[col];
        const r = self.physical(index);
        if (c.starts[r] == c.starts[r + 1]) return null;
        return c.values.strings.get(c.starts[r]);
    }

    /// An integer column's value at a row; null when absent.
    pub fn int(self: *const Self, col: usize, index: usize) ?i64 {
        const c = &self.columns[col];
        const r = self.physical(index);
        if (c.starts[r] == c.starts[r + 1]) return null;
        return c.values.ints.items[c.starts[r]];
    }

    /// A list-of-strings column's items at a row.
    ///
    /// Parameters:
    /// - `self`: the dataset.
    /// - `allocator`: owns the slice (the strings are borrowed).
    /// - `col`: the column index.
    /// - `index`: the row.
    ///
    /// Return: the items; allocation errors.
    pub fn strings(self: *const Self, allocator: std.mem.Allocator, col: usize, index: usize) ![]const []const u8 {
        const c = &self.columns[col];
        const r = self.physical(index);
        const items = try allocator.alloc([]const u8, c.starts[r + 1] - c.starts[r]);
        for (items, c.starts[r]..) |*item, v| item.* = c.values.strings.get(v);
        return items;
    }

    fn physical(self: *const Self, index: usize) usize {
        return if (self.permutation) |p| p[index] else index;
    }

    /// The split directory that has a manifest: this port's, nanochat's, or a new download.
    fn findOrDownload(allocator: std.mem.Allocator, io: std.Io, storage: mod.Storage, config: *const mod.Config, repo: []const u8, subset: []const u8, split: []const u8, out: ?*std.Io.Writer) ![]u8 {
        const slug = try std.mem.replaceOwned(u8, allocator, repo, "/", "--");
        defer allocator.free(slug);
        for ([_][]const u8{ config.base_dir, config.nanochat_dir }) |root| {
            const dir = try std.fs.path.join(allocator, &.{ root, "task_data", slug, subset, split });
            errdefer allocator.free(dir);
            const manifest = try std.fs.path.join(allocator, &.{ dir, "manifest.json" });
            defer allocator.free(manifest);
            if (try storage.exists(manifest)) return dir;
            allocator.free(dir);
        }
        const dir = try std.fs.path.join(allocator, &.{ config.base_dir, "task_data", slug, subset, split });
        errdefer allocator.free(dir);
        try download(allocator, io, storage, repo, subset, split, dir, out);
        return dir;
    }

    /// Lists the split's parquet files through the hub API and downloads them.
    fn download(allocator: std.mem.Allocator, io: std.Io, storage: mod.Storage, repo: []const u8, subset: []const u8, split: []const u8, dir: []const u8, out: ?*std.Io.Writer) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const listing_url = try std.fmt.allocPrint(allocator, "{s}/{s}/parquet/{s}/{s}", .{ api_url, repo, subset, split });
        defer allocator.free(listing_url);
        const listing = try fetch(allocator, io, listing_url);
        defer allocator.free(listing);
        const urls = std.json.parseFromSlice([]const []const u8, allocator, listing, .{}) catch |err| {
            log.warn("{s}: unexpected listing [{t}]: {s}", .{ listing_url, err, listing[0..@min(listing.len, 200)] });
            return error.InvalidListing;
        };
        defer urls.deinit();
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }
        for (urls.value, 0..) |url, i| {
            if (out) |w| {
                try w.print("Downloading {s} ...\n", .{url});
                try w.flush();
            }
            const body = try fetch(allocator, io, url);
            defer allocator.free(body);
            if (body.len < 12 or !std.mem.eql(u8, body[0..4], "PAR1")) {
                log.warn("{s} did not return a parquet file ({d} bytes)", .{ url, body.len });
                return error.InvalidParquet;
            }
            const name = try std.fmt.allocPrint(allocator, "{d:0>5}.parquet", .{i});
            try names.append(allocator, name);
            const path = try std.fs.path.join(allocator, &.{ dir, name });
            defer allocator.free(path);
            try storage.write(path, body);
        }
        // The manifest goes last: its presence means the download completed.
        const manifest = try std.json.Stringify.valueAlloc(allocator, names.items, .{});
        defer allocator.free(manifest);
        const path = try std.fs.path.join(allocator, &.{ dir, "manifest.json" });
        defer allocator.free(path);
        try storage.write(path, manifest);
    }

    fn fetch(allocator: std.mem.Allocator, io: std.Io, url: []const u8) ![]u8 {
        var node = try mod.zigstorage.Node.init(allocator, io, .empty, url);
        defer node.deinit();
        return node.read(.all);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "hub dataset loads nested columns across shards and shuffles like numpy" {
    const allocator = std.testing.allocator;
    const storage = mod.Storage.init(allocator, std.testing.io);
    const dir = mod.build_options.source_root ++ "/testdata/task_base/task_data/allenai--ai2_arc/ARC-Easy/test";
    var ds = try HubDataset.load(allocator, std.testing.io, storage, dir, &.{ "question", "choices.text", "choices.label", "answerKey" });
    defer ds.deinit();
    try std.testing.expectEqual(@as(usize, 30), ds.len());
    const labels = try ds.strings(allocator, 2, 0);
    defer allocator.free(labels);
    try std.testing.expect(labels.len >= 3);
    try std.testing.expectEqualStrings("A", labels[0]);
    try ds.shuffle(42);
    // default_rng(42).permutation(30)[0]
    var rng = mod.NumpyRandom.init(42);
    const p = try rng.permutation(allocator, 30);
    defer allocator.free(p);
    try std.testing.expectEqual(p[0], ds.permutation.?[0]);
    try std.testing.expect(ds.string(3, 0) != null);
}
