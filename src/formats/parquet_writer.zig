const std = @import("std");
const log = std.log.scoped(.zignanogpt_parquet_writer);
const mod = @import("../module.zig");

/// One row group's place in the file, for the footer.
const RowGroupMeta = struct {
    rows: usize,
    first_page: u64,
    uncompressed: u64,
    compressed: u64,
};

/// Writes a Parquet file with one required UTF-8 string column, the layout of
/// nanochat's pretraining shards: PLAIN-encoded v1 data pages of about
/// `page_bytes` each, Snappy-compressed, no statistics or dictionary. Rows are
/// cut into row groups of `row_group_size`. Streams to the file through a
/// storage session, so only one page is held; `finish` commits the file.
pub const ParquetWriter = struct {
    const Self = @This();

    /// Uncompressed bytes per data page (a longer value gets a page to itself).
    pub const page_bytes = 1 << 20;
    const magic = "PAR1";

    allocator: std.mem.Allocator,
    session: mod.StorageSession,
    column: []const u8,
    row_group_size: usize,
    /// The page being filled: PLAIN values (u32 length, bytes).
    page: std.ArrayList(u8) = .empty,
    page_rows: usize = 0,
    /// Compression and page-header scratch, reused.
    scratch: std.ArrayList(u8) = .empty,
    header: std.ArrayList(u8) = .empty,
    current: ?RowGroupMeta = null,
    groups: std.ArrayList(RowGroupMeta) = .empty,
    rows: usize = 0,

    /// Starts a file at `path` (invisible there until `finish`).
    ///
    /// Parameters:
    /// - `allocator`: page buffers and metadata.
    /// - `storage`: where the file is written.
    /// - `path`: the file.
    /// - `column`: the column's name (borrowed until `deinit`).
    /// - `row_group_size`: rows per row group; at least 1.
    ///
    /// Return: the writer; storage errors.
    pub fn create(allocator: std.mem.Allocator, storage: mod.Storage, path: []const u8, column: []const u8, row_group_size: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        std.debug.assert(row_group_size > 0);
        var session = try storage.create(path);
        errdefer session.deinit();
        try session.append(magic);
        return .{ .allocator = allocator, .session = session, .column = column, .row_group_size = row_group_size };
    }

    /// Discards the file unless `finish` committed it, and frees the buffers.
    pub fn deinit(self: *Self) void {
        self.session.deinit();
        self.page.deinit(self.allocator);
        self.scratch.deinit(self.allocator);
        self.header.deinit(self.allocator);
        self.groups.deinit(self.allocator);
    }

    /// Appends one row.
    ///
    /// Parameters:
    /// - `self`: the writer.
    /// - `value`: the string (UTF-8; at most 4 GiB).
    ///
    /// Return: nothing; storage and allocation errors.
    pub fn append(self: *Self, value: []const u8) !void {
        if (self.page.items.len > 0 and self.page.items.len + 4 + value.len > page_bytes) try self.flushPage();
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(value.len), .little);
        try self.page.appendSlice(self.allocator, &len);
        try self.page.appendSlice(self.allocator, value);
        self.page_rows += 1;
        self.rows += 1;
        if (self.rowsInGroup() == self.row_group_size) try self.endRowGroup();
    }

    /// Writes the last page and the footer, then commits the file.
    ///
    /// Parameters:
    /// - `self`: the writer.
    ///
    /// Return: the file's size in bytes; storage and allocation errors.
    pub fn finish(self: *Self) !u64 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        try self.endRowGroup();
        var footer: std.ArrayList(u8) = .empty;
        defer footer.deinit(self.allocator);
        try self.writeFooter(&footer);
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(footer.items.len), .little);
        try self.session.append(footer.items);
        try self.session.append(&len);
        try self.session.append(magic);
        try self.session.save();
        return self.session.written;
    }

    // -------------------------------------------------------------------------
    // Private helpers

    fn rowsInGroup(self: *const Self) usize {
        return (if (self.current) |g| g.rows else 0) + self.page_rows;
    }

    /// Compresses the page and writes it with its header.
    fn flushPage(self: *Self) !void {
        if (self.page_rows == 0) return;
        const raw = self.page.items;
        try self.scratch.resize(self.allocator, mod.Snappy.maxCompressedLength(raw.len));
        const compressed = self.scratch.items[0..mod.Snappy.compress(raw, self.scratch.items)];

        self.header.clearRetainingCapacity();
        var w = mod.ThriftWriter.init(self.allocator, &self.header);
        try w.beginStruct();
        try w.int32(1, 0); // DATA_PAGE
        try w.int32(2, @intCast(raw.len));
        try w.int32(3, @intCast(compressed.len));
        try w.structField(5);
        try w.int32(1, @intCast(self.page_rows));
        try w.int32(2, 0); // PLAIN
        try w.int32(3, 3); // RLE (no levels: the column is required)
        try w.int32(4, 3);
        try w.endStruct();
        try w.endStruct();

        var group = self.current orelse RowGroupMeta{ .rows = 0, .first_page = self.session.written, .uncompressed = 0, .compressed = 0 };
        group.rows += self.page_rows;
        group.uncompressed += self.header.items.len + raw.len;
        group.compressed += self.header.items.len + compressed.len;
        self.current = group;
        try self.session.append(self.header.items);
        try self.session.append(compressed);
        self.page.clearRetainingCapacity();
        self.page_rows = 0;
    }

    fn endRowGroup(self: *Self) !void {
        try self.flushPage();
        const group = self.current orelse return;
        try self.groups.append(self.allocator, group);
        self.current = null;
    }

    /// `FileMetaData`: the schema, every row group's one column chunk, the writer.
    fn writeFooter(self: *Self, out: *std.ArrayList(u8)) !void {
        var w = mod.ThriftWriter.init(self.allocator, out);
        try w.beginStruct();
        try w.int32(1, 1); // version
        try w.list(2, .@"struct", 2);
        try w.beginStruct(); // the root
        try w.binary(4, "schema");
        try w.int32(5, 1);
        try w.endStruct();
        try w.beginStruct(); // the column
        try w.int32(1, 6); // BYTE_ARRAY
        try w.int32(3, 0); // REQUIRED
        try w.binary(4, self.column);
        try w.int32(6, 0); // converted type UTF8
        try w.structField(10); // logical type
        try w.structField(1); // STRING
        try w.endStruct();
        try w.endStruct();
        try w.endStruct();
        try w.int64(3, @intCast(self.rows));
        try w.list(4, .@"struct", self.groups.items.len);
        for (self.groups.items) |g| {
            try w.beginStruct();
            try w.list(1, .@"struct", 1);
            try w.beginStruct(); // ColumnChunk
            try w.int64(2, @intCast(g.first_page));
            try w.structField(3); // ColumnMetaData
            try w.int32(1, 6);
            try w.list(2, .i32, 1);
            try w.listInt32(0); // PLAIN
            try w.list(3, .binary, 1);
            try w.listBinary(self.column);
            try w.int32(4, 1); // SNAPPY
            try w.int64(5, @intCast(g.rows));
            try w.int64(6, @intCast(g.uncompressed));
            try w.int64(7, @intCast(g.compressed));
            try w.int64(9, @intCast(g.first_page));
            try w.endStruct();
            try w.endStruct();
            try w.int64(2, @intCast(g.uncompressed));
            try w.int64(3, @intCast(g.rows));
            try w.int64(5, @intCast(g.first_page));
            try w.int64(6, @intCast(g.compressed));
            try w.endStruct();
        }
        try w.binary(6, "zignanogpt version " ++ mod.build_options.version);
        try w.endStruct();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "parquet writer output reads back by row group, pages and long values included" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const path = try std.Io.Dir.path.join(allocator, &.{ root, "shard_00000.parquet" });
    defer allocator.free(path);
    const storage = mod.Storage.init(allocator, std.testing.io);

    // 2,500 rows in groups of 1,024 (the last one short); one value spans pages.
    const long = try allocator.alloc(u8, ParquetWriter.page_bytes + 10);
    defer allocator.free(long);
    for (long, 0..) |*c, i| c.* = 'a' + @as(u8, @intCast(i % 26));
    var writer = try ParquetWriter.create(allocator, storage, path, "text", 1024);
    defer writer.deinit();
    var buf: [64]u8 = undefined;
    for (0..2500) |i| {
        if (i == 1500) try writer.append(long) else try writer.append(try std.mem.print(&buf, "document {d} — ünïcode", .{i}));
    }
    try std.testing.expect(!try storage.exists(path));
    _ = try writer.finish();

    var file = try mod.ParquetFile.open(allocator, std.testing.io, path);
    defer file.deinit();
    try std.testing.expectEqual(@as(usize, 3), file.row_groups.len);
    const col = try file.column("text");
    var strings: mod.ParquetStrings = .{};
    defer strings.deinit(allocator);
    var row: usize = 0;
    for (0..file.row_groups.len) |rg| {
        strings.clear();
        try file.readStrings(rg, col, &strings);
        for (0..strings.len()) |i| {
            if (row == 1500) {
                try std.testing.expectEqualSlices(u8, long, strings.get(i));
            } else {
                try std.testing.expectEqualStrings(try std.mem.print(&buf, "document {d} — ünïcode", .{row}), strings.get(i));
            }
            row += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 2500), row);
}
