const std = @import("std");
const log = std.log.scoped(.zignanogpt_parquet_file);
const mod = @import("module.zig");

/// Parquet codecs this reader decodes.
const Codec = enum(i32) { uncompressed = 0, snappy = 1, zstd = 6, _ };
/// Parquet page types.
const PageType = enum(i32) { data = 0, index = 1, dictionary = 2, data_v2 = 3, _ };
/// Parquet value encodings this reader decodes.
const Encoding = enum(i32) { plain = 0, plain_dictionary = 2, rle = 3, rle_dictionary = 8, _ };
/// Parquet physical types this reader decodes.
const PhysicalType = enum(i32) { int32 = 1, int64 = 2, byte_array = 6, _ };

/// A leaf column: its path and how deeply it nests.
pub const ParquetColumn = struct {
    /// The dotted path, as pyarrow's `path_in_schema` (e.g. `choices.list.item`).
    name: []const u8,
    /// The path without list wrappers (e.g. `choices`): writers name the
    /// repeated group and its element differently (`list.item`, `list.element`).
    logical: []const u8,
    physical_type: i32,
    /// Optional or repeated ancestors, the column included: a value is present
    /// at definition level `max_def`.
    max_def: u8,
    /// Repeated ancestors: 0 for a flat column, 1 for a list.
    max_rep: u8,
};

/// One column chunk's location in the file.
pub const ParquetChunk = struct {
    codec: i32,
    num_values: i64,
    /// First page (the dictionary page when there is one).
    offset: u64,
    size: u64,
};

pub const ParquetRowGroup = struct {
    num_rows: i64,
    chunks: []ParquetChunk,
};

/// The strings of one column chunk, packed into one buffer.
pub const ParquetStrings = struct {
    const Self = @This();

    bytes: std.ArrayList(u8) = .empty,
    /// `offsets[i]..offsets[i + 1]` is string `i`.
    offsets: std.ArrayList(usize) = .empty,

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.bytes.deinit(allocator);
        self.offsets.deinit(allocator);
    }

    pub fn clear(self: *Self) void {
        self.bytes.clearRetainingCapacity();
        self.offsets.clearRetainingCapacity();
    }

    pub fn len(self: *const Self) usize {
        return if (self.offsets.items.len == 0) 0 else self.offsets.items.len - 1;
    }

    pub fn get(self: *const Self, i: usize) []const u8 {
        return self.bytes.items[self.offsets.items[i]..self.offsets.items[i + 1]];
    }

    fn append(self: *Self, allocator: std.mem.Allocator, s: []const u8) !void {
        if (self.offsets.items.len == 0) try self.offsets.append(allocator, 0);
        try self.bytes.appendSlice(allocator, s);
        try self.offsets.append(allocator, self.bytes.items.len);
    }
};

/// One column chunk's values and levels. Only present values are stored
/// (strings for byte arrays, `ints` for int32/int64); `def` and `rep` hold one
/// level per entry (empty when the column has none). `rowStarts` turns the
/// levels into rows for list columns.
pub const ParquetValues = struct {
    const Self = @This();

    strings: ParquetStrings = .{},
    ints: std.ArrayList(i64) = .empty,
    def: std.ArrayList(u8) = .empty,
    rep: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.strings.deinit(allocator);
        self.ints.deinit(allocator);
        self.def.deinit(allocator);
        self.rep.deinit(allocator);
    }

    pub fn clear(self: *Self) void {
        self.strings.clear();
        self.ints.clearRetainingCapacity();
        self.def.clearRetainingCapacity();
        self.rep.clearRetainingCapacity();
    }

    /// Where each row's values start: row `r` holds values
    /// `starts[r] .. starts[r + 1]` (a list column's items; for a flat column
    /// one value, or none when null).
    ///
    /// Parameters:
    /// - `self`: the values read for `column`.
    /// - `allocator`: owns the result.
    /// - `column`: the column they came from.
    ///
    /// Return: `rows + 1` offsets; allocation errors.
    pub fn rowStarts(self: *const Self, allocator: std.mem.Allocator, column: ParquetColumn) ![]usize {
        var starts: std.ArrayList(usize) = .empty;
        errdefer starts.deinit(allocator);
        const entries = @max(self.def.items.len, self.rep.items.len);
        if (entries == 0) {
            // Required flat column: one value per row.
            const count = @max(self.strings.len(), self.ints.items.len);
            for (0..count + 1) |i| try starts.append(allocator, i);
            return starts.toOwnedSlice(allocator);
        }
        var value: usize = 0;
        for (0..entries) |e| {
            if (column.max_rep == 0 or self.rep.items[e] == 0) try starts.append(allocator, value);
            if (column.max_def == 0 or self.def.items[e] == column.max_def) value += 1;
        }
        try starts.append(allocator, value);
        return starts.toOwnedSlice(allocator);
    }
};

/// A Parquet file read through zigstorage: the footer once, then one ranged
/// read per column chunk. Decodes byte-array, int32 and int64 leaf columns,
/// nested ones included (repetition and definition levels): data pages v1 and
/// v2, PLAIN and dictionary encodings, UNCOMPRESSED / SNAPPY / ZSTD codecs.
/// That covers nanochat's pretraining shards (`text`, ZSTD, PLAIN) and the
/// Hugging Face parquet exports of its SFT and eval tasks (SNAPPY, dictionary).
pub const ParquetFile = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    node: mod.zigstorage.Node,
    /// The footer bytes; column names borrow from them.
    footer: []u8,
    columns: []ParquetColumn,
    row_groups: []ParquetRowGroup,
    num_rows: i64,
    /// Reused across pages: decompressed page and zstd output.
    page: std.Io.Writer.Allocating,

    /// Opens a file and parses its footer.
    ///
    /// Parameters:
    /// - `allocator`: owns the metadata and buffers.
    /// - `io`: the Io reads run on.
    /// - `location`: a path or URL zigstorage can read ranges of.
    ///
    /// Return: the file; storage errors, `error.InvalidParquet`, `error.UnsupportedParquet`.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, location: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = undefined;
        self.allocator = allocator;
        self.node = try mod.zigstorage.Node.init(allocator, io, .empty, location);
        errdefer self.node.deinit();
        try self.node.fetchAttributes();
        const size = self.node.size orelse return error.InvalidParquet;
        if (size < 12) return error.InvalidParquet;
        const tail = try self.node.read(.{ .offset = size - 8, .length = 8 });
        defer allocator.free(tail);
        if (tail.len != 8 or !std.mem.eql(u8, tail[4..8], "PAR1")) {
            log.warn("{s} is not a parquet file", .{location});
            return error.InvalidParquet;
        }
        const footer_len = std.mem.readInt(u32, tail[0..4], .little);
        if (footer_len > size - 12) return error.InvalidParquet;
        self.footer = try self.node.read(.{ .offset = size - 8 - footer_len, .length = footer_len });
        errdefer allocator.free(self.footer);
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        try self.parseFooter();
        self.page = .init(allocator);
        return self;
    }

    /// Frees the metadata and buffers.
    ///
    /// Parameters:
    /// - `self`: the file.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.page.deinit();
        self.arena.deinit();
        self.allocator.free(self.footer);
        self.node.deinit();
    }

    /// The index of a leaf column.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `name`: the column's path or logical path (`choices` for `choices.list.item`).
    ///
    /// Return: the index, or `error.MissingColumn`.
    pub fn column(self: *const Self, name: []const u8) !usize {
        for (self.columns, 0..) |c, i| {
            if (std.mem.eql(u8, c.name, name) or std.mem.eql(u8, c.logical, name)) return i;
        }
        log.warn("no parquet column named {s}", .{name});
        return error.MissingColumn;
    }

    /// Reads one row group's strings from a byte-array column.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `row_group`: the row group index.
    /// - `col`: the column index (`column`).
    /// - `out`: receives the non-null strings, appended.
    ///
    /// Return: nothing; storage, decoding and unsupported-feature errors.
    pub fn readStrings(self: *Self, row_group: usize, col: usize, out: *ParquetStrings) !void {
        if (@as(PhysicalType, @enumFromInt(self.columns[col].physical_type)) != .byte_array) return unsupported("non byte-array column");
        var values = ParquetValues{ .strings = out.* };
        defer {
            out.* = values.strings;
            values.strings = .{};
            values.deinit(self.allocator);
        }
        try self.readColumn(row_group, col, &values);
    }

    /// Reads one row group of a leaf column: present values plus levels.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `row_group`: the row group index.
    /// - `col`: the column index (`column`).
    /// - `out`: receives the values and levels, appended.
    ///
    /// Return: nothing; storage, decoding and unsupported-feature errors.
    pub fn readColumn(self: *Self, row_group: usize, col: usize, out: *ParquetValues) !void {
        const info = self.columns[col];
        switch (@as(PhysicalType, @enumFromInt(info.physical_type))) {
            .byte_array, .int32, .int64 => {},
            _ => return unsupported("physical type"),
        }
        const chunk = self.row_groups[row_group].chunks[col];
        const bytes = try self.node.read(.{ .offset = chunk.offset, .length = chunk.size });
        defer self.allocator.free(bytes);
        if (bytes.len != chunk.size) return error.InvalidParquet;

        var dictionary: ParquetValues = .{};
        defer dictionary.deinit(self.allocator);
        var pos: usize = 0;
        var values_left = chunk.num_values;
        while (values_left > 0 and pos < bytes.len) {
            var reader = mod.ThriftReader.init(bytes[pos..]);
            const header = try parsePageHeader(&reader);
            pos += reader.pos;
            if (header.compressed_size > bytes.len - pos) return error.InvalidParquet;
            const body = bytes[pos..][0..header.compressed_size];
            pos += header.compressed_size;
            switch (header.type) {
                .dictionary => {
                    const page = try self.decompress(@enumFromInt(chunk.codec), body, header.uncompressed_size);
                    try readPlain(self.allocator, info, page, std.math.cast(usize, header.num_values) orelse return error.InvalidParquet, &dictionary);
                },
                .data, .data_v2 => {
                    try self.decodeDataPage(info, @enumFromInt(chunk.codec), header, body, &dictionary, out);
                    values_left -= header.num_values;
                },
                else => {}, // index pages carry nothing we need
            }
        }
    }

    /// Decodes one data page (v1 or v2) into `out`.
    fn decodeDataPage(self: *Self, info: ParquetColumn, codec: Codec, header: PageHeader, body: []const u8, dictionary: *const ParquetValues, out: *ParquetValues) !void {
        const n: usize = @intCast(header.num_values);
        var rep_levels: []const u8 = &.{};
        var def_levels: []const u8 = &.{};
        var values: []const u8 = undefined;
        if (header.type == .data_v2) {
            if (header.rep_bytes + header.def_bytes > body.len) return error.InvalidParquet;
            rep_levels = body[0..header.rep_bytes];
            def_levels = body[header.rep_bytes..][0..header.def_bytes];
            const rest = body[header.rep_bytes + header.def_bytes ..];
            values = if (header.is_compressed)
                try self.decompress(codec, rest, header.uncompressed_size - header.rep_bytes - header.def_bytes)
            else
                rest;
        } else {
            values = try self.decompress(codec, body, header.uncompressed_size);
            // v1: each level stream is prefixed by its 4-byte length, repetition first.
            if (info.max_rep > 0) rep_levels = try takeLevels(&values);
            if (info.max_def > 0) def_levels = try takeLevels(&values);
        }

        if (info.max_rep > 0) {
            const start = out.rep.items.len;
            try out.rep.resize(self.allocator, start + n);
            try decodeLevels(rep_levels, bitWidth(info.max_rep), out.rep.items[start..]);
        }
        var count = n;
        if (info.max_def > 0) {
            const start = out.def.items.len;
            try out.def.resize(self.allocator, start + n);
            const levels = out.def.items[start..];
            try decodeLevels(def_levels, bitWidth(info.max_def), levels);
            count = 0;
            for (levels) |l| count += @intFromBool(l == info.max_def);
        }
        if (count == 0) return;

        switch (header.encoding) {
            .plain => try readPlain(self.allocator, info, values, count, out),
            .plain_dictionary, .rle_dictionary => {
                if (values.len == 0) return error.InvalidParquet;
                const width = values[0];
                const indices = try self.allocator.alloc(u32, count);
                defer self.allocator.free(indices);
                try decodeHybrid(values[1..], width, indices);
                const strings = @as(PhysicalType, @enumFromInt(info.physical_type)) == .byte_array;
                for (indices) |i| {
                    if (strings) {
                        if (i >= dictionary.strings.len()) return error.InvalidParquet;
                        try out.strings.append(self.allocator, dictionary.strings.get(i));
                    } else {
                        if (i >= dictionary.ints.items.len) return error.InvalidParquet;
                        try out.ints.append(self.allocator, dictionary.ints.items[i]);
                    }
                }
            },
            else => return unsupported("value encoding"),
        }
    }

    /// Splits a v1 level stream (u32 length, then the levels) off the front of `page`.
    fn takeLevels(page: *[]const u8) ![]const u8 {
        if (page.len < 4) return error.InvalidParquet;
        const len = std.mem.readInt(u32, page.*[0..4], .little);
        if (len > page.len - 4) return error.InvalidParquet;
        const levels = page.*[4..][0..len];
        page.* = page.*[4 + len ..];
        return levels;
    }

    /// Bits needed for levels up to `max`.
    fn bitWidth(max: u8) u8 {
        return 8 - @clz(max);
    }

    /// Decompresses a page body into the shared page buffer.
    fn decompress(self: *Self, codec: Codec, src: []const u8, size: usize) ![]const u8 {
        switch (codec) {
            .uncompressed => return src,
            .snappy => {
                self.page.clearRetainingCapacity();
                const dst = try self.page.writer.writableSliceGreedy(size);
                try mod.Snappy.decompress(src, dst[0..size]);
                return dst[0..size];
            },
            .zstd => {
                self.page.clearRetainingCapacity();
                var in: std.Io.Reader = .fixed(src);
                var zstd: std.compress.zstd.Decompress = .init(&in, &.{}, .{});
                _ = zstd.reader.streamRemaining(&self.page.writer) catch |err| {
                    log.warn("zstd page failed [{t}]", .{zstd.err orelse err});
                    return error.InvalidParquet;
                };
                const page = self.page.written();
                if (page.len != size) return error.InvalidParquet;
                return page;
            },
            _ => return unsupported("compression codec"),
        }
    }

    // -------------------------------------------------------------------------
    // Footer

    fn parseFooter(self: *Self) !void {
        const arena = self.arena.allocator();
        var r = mod.ThriftReader.init(self.footer);
        var columns: std.ArrayList(ParquetColumn) = .empty;
        var row_groups: std.ArrayList(ParquetRowGroup) = .empty;
        self.num_rows = 0;
        while (true) {
            const f = try r.field();
            if (f.type == .stop) break;
            switch (f.id) {
                2 => { // schema: list<SchemaElement>, depth first, the root first
                    const list = try r.listHeader();
                    // Open groups: children left, their levels and path.
                    const Group = struct { left: i32, def: u8, rep: u8, path: []const u8, logical: []const u8, repeated: bool };
                    var stack: std.ArrayList(Group) = .empty;
                    for (0..list.len) |i| {
                        const element = try parseSchemaElement(&r);
                        if (i == 0) {
                            try stack.append(arena, .{ .left = element.num_children, .def = 0, .rep = 0, .path = "", .logical = "", .repeated = false });
                            continue;
                        }
                        while (stack.items.len > 0 and stack.items[stack.items.len - 1].left == 0) _ = stack.pop();
                        if (stack.items.len == 0) return error.InvalidParquet;
                        const parent = &stack.items[stack.items.len - 1];
                        parent.left -= 1;
                        // repetition: 0 required, 1 optional, 2 repeated
                        const def = parent.def + @intFromBool(element.repetition != 0);
                        const rep_level = parent.rep + @intFromBool(element.repetition == 2);
                        const path = try join(arena, parent.path, element.name);
                        // A list is `<name>` / repeated group / element: the logical
                        // path keeps `<name>` and drops the other two.
                        const repeated_group = element.repetition == 2 and element.num_children > 0;
                        const logical = if (repeated_group or parent.repeated) parent.logical else try join(arena, parent.logical, element.name);
                        if (element.num_children > 0) {
                            try stack.append(arena, .{ .left = element.num_children, .def = def, .rep = rep_level, .path = path, .logical = logical, .repeated = repeated_group });
                        } else {
                            if (rep_level > 1) return unsupported("lists of lists");
                            try columns.append(arena, .{ .name = path, .logical = logical, .physical_type = element.type, .max_def = def, .max_rep = rep_level });
                        }
                    }
                },
                3 => self.num_rows = try r.readI64(),
                4 => {
                    const list = try r.listHeader();
                    for (0..list.len) |_| try row_groups.append(arena, try parseRowGroup(arena, &r));
                },
                else => try r.skip(f.type),
            }
        }
        self.columns = try columns.toOwnedSlice(arena);
        self.row_groups = try row_groups.toOwnedSlice(arena);
        for (self.row_groups) |rg| {
            if (rg.chunks.len != self.columns.len) return error.InvalidParquet;
        }
    }

    fn join(arena: std.mem.Allocator, prefix: []const u8, name: []const u8) ![]const u8 {
        return if (prefix.len == 0) name else std.mem.concat(arena, u8, &.{ prefix, ".", name });
    }

    const SchemaElement = struct { type: i32 = 0, repetition: i32 = 0, name: []const u8 = "", num_children: i32 = 0 };

    fn parseSchemaElement(r: *mod.ThriftReader) !SchemaElement {
        var e: SchemaElement = .{};
        const saved = r.beginStruct();
        defer r.endStruct(saved);
        while (true) {
            const f = try r.field();
            if (f.type == .stop) break;
            switch (f.id) {
                1 => e.type = try r.readI32(),
                3 => e.repetition = try r.readI32(),
                4 => e.name = try r.readBinary(),
                5 => e.num_children = try r.readI32(),
                else => try r.skip(f.type),
            }
        }
        return e;
    }

    fn parseRowGroup(arena: std.mem.Allocator, r: *mod.ThriftReader) !ParquetRowGroup {
        var rg: ParquetRowGroup = .{ .num_rows = 0, .chunks = &.{} };
        const saved = r.beginStruct();
        defer r.endStruct(saved);
        while (true) {
            const f = try r.field();
            if (f.type == .stop) break;
            switch (f.id) {
                1 => {
                    const list = try r.listHeader();
                    const chunks = try arena.alloc(ParquetChunk, list.len);
                    for (chunks) |*c| c.* = try parseColumnChunk(r);
                    rg.chunks = chunks;
                },
                3 => rg.num_rows = try r.readI64(),
                else => try r.skip(f.type),
            }
        }
        return rg;
    }

    fn parseColumnChunk(r: *mod.ThriftReader) !ParquetChunk {
        var chunk: ?ParquetChunk = null;
        const saved = r.beginStruct();
        defer r.endStruct(saved);
        while (true) {
            const f = try r.field();
            if (f.type == .stop) break;
            switch (f.id) {
                3 => chunk = try parseColumnMetaData(r),
                else => try r.skip(f.type),
            }
        }
        return chunk orelse unsupported("column chunk without inline metadata");
    }

    fn parseColumnMetaData(r: *mod.ThriftReader) !ParquetChunk {
        var codec: i32 = 0;
        var num_values: i64 = 0;
        var size: i64 = 0;
        var data_offset: i64 = 0;
        var dict_offset: ?i64 = null;
        const saved = r.beginStruct();
        defer r.endStruct(saved);
        while (true) {
            const f = try r.field();
            if (f.type == .stop) break;
            switch (f.id) {
                4 => codec = try r.readI32(),
                5 => num_values = try r.readI64(),
                7 => size = try r.readI64(),
                9 => data_offset = try r.readI64(),
                11 => dict_offset = try r.readI64(),
                else => try r.skip(f.type),
            }
        }
        const first = if (dict_offset) |d| @min(d, data_offset) else data_offset;
        if (first < 0 or size < 0) return error.InvalidParquet;
        return ParquetChunk{ .codec = codec, .num_values = num_values, .offset = @intCast(first), .size = @intCast(size) };
    }

    // -------------------------------------------------------------------------
    // Pages

    const PageHeader = struct {
        type: PageType = .data,
        uncompressed_size: usize = 0,
        compressed_size: usize = 0,
        num_values: i64 = 0,
        encoding: Encoding = .plain,
        def_bytes: usize = 0,
        rep_bytes: usize = 0,
        is_compressed: bool = true,
    };

    fn parsePageHeader(r: *mod.ThriftReader) !PageHeader {
        var h: PageHeader = .{};
        while (true) {
            const f = try r.field();
            if (f.type == .stop) break;
            switch (f.id) {
                1 => h.type = @enumFromInt(try r.readI32()),
                2 => h.uncompressed_size = try toUsize(try r.readI32()),
                3 => h.compressed_size = try toUsize(try r.readI32()),
                5, 7, 8 => {
                    // data page v1, dictionary page, data page v2 headers
                    const saved = r.beginStruct();
                    defer r.endStruct(saved);
                    while (true) {
                        const g = try r.field();
                        if (g.type == .stop) break;
                        switch (f.id) {
                            5, 7 => switch (g.id) {
                                1 => h.num_values = try r.readI32(),
                                2 => h.encoding = @enumFromInt(try r.readI32()),
                                else => try r.skip(g.type),
                            },
                            else => switch (g.id) {
                                1 => h.num_values = try r.readI32(),
                                4 => h.encoding = @enumFromInt(try r.readI32()),
                                5 => h.def_bytes = try toUsize(try r.readI32()),
                                6 => h.rep_bytes = try toUsize(try r.readI32()),
                                7 => h.is_compressed = try mod.ThriftReader.boolOf(g.type),
                                else => try r.skip(g.type),
                            },
                        }
                    }
                },
                else => try r.skip(f.type),
            }
        }
        return h;
    }

    /// PLAIN values: byte arrays with a 4-byte little-endian length before
    /// each, or little-endian int32/int64.
    fn readPlain(allocator: std.mem.Allocator, info: ParquetColumn, data: []const u8, count: usize, out: *ParquetValues) !void {
        switch (@as(PhysicalType, @enumFromInt(info.physical_type))) {
            .byte_array => {
                var pos: usize = 0;
                for (0..count) |_| {
                    if (data.len - pos < 4) return error.InvalidParquet;
                    const n = std.mem.readInt(u32, data[pos..][0..4], .little);
                    pos += 4;
                    if (n > data.len - pos) return error.InvalidParquet;
                    try out.strings.append(allocator, data[pos..][0..n]);
                    pos += n;
                }
            },
            .int32 => {
                if (data.len < count * 4) return error.InvalidParquet;
                for (0..count) |i| try out.ints.append(allocator, std.mem.readInt(i32, data[i * 4 ..][0..4], .little));
            },
            .int64 => {
                if (data.len < count * 8) return error.InvalidParquet;
                for (0..count) |i| try out.ints.append(allocator, std.mem.readInt(i64, data[i * 8 ..][0..8], .little));
            },
            _ => return unsupported("physical type"),
        }
    }

    /// Repetition or definition levels.
    fn decodeLevels(data: []const u8, width: u8, levels: []u8) !void {
        var buf: [256]u32 = undefined;
        var done: usize = 0;
        var r = Hybrid{ .data = data, .width = width };
        while (done < levels.len) {
            const got = try r.next(buf[0..@min(buf.len, levels.len - done)]);
            for (buf[0..got], levels[done..][0..got]) |v, *l| l.* = std.math.cast(u8, v) orelse return error.InvalidParquet;
            done += got;
        }
    }

    /// Dictionary indices.
    fn decodeHybrid(data: []const u8, width: u8, out: []u32) !void {
        var r = Hybrid{ .data = data, .width = width };
        var done: usize = 0;
        while (done < out.len) done += try r.next(out[done..]);
    }

    /// Parquet's RLE / bit-packed hybrid encoding.
    const Hybrid = struct {
        data: []const u8,
        width: u8,
        pos: usize = 0,
        /// Remaining values in the current run.
        rle_left: usize = 0,
        rle_value: u32 = 0,
        packed_left: usize = 0,
        bit_pos: usize = 0,

        fn next(h: *Hybrid, out: []u32) !usize {
            if (h.rle_left == 0 and h.packed_left == 0) try h.header();
            if (h.rle_left > 0) {
                const n = @min(h.rle_left, out.len);
                @memset(out[0..n], h.rle_value);
                h.rle_left -= n;
                return n;
            }
            const n = @min(h.packed_left, out.len);
            for (out[0..n]) |*v| {
                var value: u32 = 0;
                for (0..h.width) |b| {
                    const bit = h.bit_pos + b;
                    if (bit / 8 >= h.data.len) return error.InvalidParquet;
                    value |= @as(u32, (h.data[bit / 8] >> @intCast(bit % 8)) & 1) << @intCast(b);
                }
                h.bit_pos += h.width;
                v.* = value;
            }
            h.packed_left -= n;
            if (h.packed_left == 0) h.pos = (h.bit_pos + 7) / 8;
            return n;
        }

        fn header(h: *Hybrid) !void {
            var value: u64 = 0;
            var shift: u6 = 0;
            while (true) {
                if (h.pos >= h.data.len) return error.InvalidParquet;
                const b = h.data[h.pos];
                h.pos += 1;
                value |= @as(u64, b & 0x7f) << shift;
                if (b & 0x80 == 0) break;
                if (shift >= 35) return error.InvalidParquet;
                shift += 7;
            }
            if (value & 1 == 0) {
                h.rle_left = @intCast(value >> 1);
                const bytes = (h.width + 7) / 8;
                if (h.pos + bytes > h.data.len) return error.InvalidParquet;
                h.rle_value = 0;
                for (0..bytes) |i| h.rle_value |= @as(u32, h.data[h.pos + i]) << @intCast(8 * i);
                h.pos += bytes;
                if (h.rle_left == 0) return error.InvalidParquet;
            } else {
                h.packed_left = @intCast((value >> 1) * 8);
                h.bit_pos = h.pos * 8;
                if (h.packed_left == 0) return error.InvalidParquet;
            }
        }
    };

    fn toUsize(v: i32) !usize {
        return std.math.cast(usize, v) orelse error.InvalidParquet;
    }

    fn unsupported(what: []const u8) error{UnsupportedParquet} {
        log.warn("unsupported parquet feature: {s}", .{what});
        return error.UnsupportedParquet;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "parquet reader matches pyarrow on every supported variant" {
    const allocator = std.testing.allocator;
    const dir = mod.build_options.source_root ++ "/testdata/parquet/";
    const storage = mod.Storage.init(allocator, std.testing.io);
    const json = try storage.read(dir ++ "expected.json");
    defer allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    var it = parsed.value.object.iterator();
    var variants: usize = 0;
    while (it.next()) |kv| : (variants += 1) {
        const path = try std.mem.concat(allocator, u8, &.{ dir, kv.key_ptr.*, ".parquet" });
        defer allocator.free(path);
        var file = try mod.ParquetFile.open(allocator, std.testing.io, path);
        defer file.deinit();
        const col = try file.column("text");
        const groups = kv.value_ptr.array.items;
        try std.testing.expectEqual(groups.len, file.row_groups.len);
        var strings: mod.ParquetStrings = .{};
        defer strings.deinit(allocator);
        for (groups, 0..) |group, rg| {
            strings.clear();
            try file.readStrings(rg, col, &strings);
            try std.testing.expectEqual(group.array.items.len, strings.len());
            for (group.array.items, 0..) |want, i| {
                std.testing.expectEqualStrings(want.string, strings.get(i)) catch |err| {
                    std.debug.print("{s}: row group {d}, string {d}\n", .{ kv.key_ptr.*, rg, i });
                    return err;
                };
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 4), variants);
}
