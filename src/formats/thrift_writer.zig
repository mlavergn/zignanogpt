const std = @import("std");
const log = std.log.scoped(.zignanogpt_thrift_writer);
const mod = @import("../module.zig");

const ThriftType = mod.ThriftType;

/// Writes Thrift's compact protocol (Parquet's page headers and footer): the
/// inverse of `ThriftReader`. Fields must be written in increasing id order
/// within a struct; `beginStruct`/`endStruct` bracket nested structs, list
/// elements included.
pub const ThriftWriter = struct {
    const Self = @This();
    const max_depth = 16;

    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    last_field: i16 = 0,
    stack: [max_depth]i16 = undefined,
    depth: usize = 0,

    /// Appends to `out`.
    ///
    /// Parameters:
    /// - `allocator`: grows `out`.
    /// - `out`: the destination.
    ///
    /// Return: the writer.
    pub fn init(allocator: std.mem.Allocator, out: *std.ArrayList(u8)) Self {
        return .{ .allocator = allocator, .out = out };
    }

    /// An i32 field.
    pub fn int32(self: *Self, id: i16, value: i32) !void {
        try self.header(id, .i32);
        try self.varint(zigzag(value));
    }

    /// An i64 field.
    pub fn int64(self: *Self, id: i16, value: i64) !void {
        try self.header(id, .i64);
        try self.varint(zigzag(value));
    }

    /// A binary (string) field.
    pub fn binary(self: *Self, id: i16, bytes: []const u8) !void {
        try self.header(id, .binary);
        try self.listBinary(bytes);
    }

    /// Opens a struct field; close it with `endStruct`.
    pub fn structField(self: *Self, id: i16) !void {
        try self.header(id, .@"struct");
        try self.beginStruct();
    }

    /// Opens a list field of `len` elements of `elem`.
    pub fn list(self: *Self, id: i16, elem: ThriftType, len: usize) !void {
        try self.header(id, .list);
        if (len < 15) {
            try self.out.append(self.allocator, @as(u8, @intCast(len)) << 4 | @intFromEnum(elem));
        } else {
            try self.out.append(self.allocator, 0xf0 | @as(u8, @intFromEnum(elem)));
            try self.varint(len);
        }
    }

    /// An i32 list element.
    pub fn listInt32(self: *Self, value: i32) !void {
        try self.varint(zigzag(value));
    }

    /// A binary list element (also the body of a binary field).
    pub fn listBinary(self: *Self, bytes: []const u8) !void {
        try self.varint(bytes.len);
        try self.out.appendSlice(self.allocator, bytes);
    }

    /// Starts a struct: a list element, or the top-level struct.
    pub fn beginStruct(self: *Self) !void {
        if (self.depth == max_depth) return error.ThriftTooDeep;
        self.stack[self.depth] = self.last_field;
        self.depth += 1;
        self.last_field = 0;
    }

    /// Ends the innermost struct (writes its stop byte).
    pub fn endStruct(self: *Self) !void {
        try self.out.append(self.allocator, 0);
        if (self.depth == 0) return;
        self.depth -= 1;
        self.last_field = self.stack[self.depth];
    }

    // -------------------------------------------------------------------------
    // Private helpers

    /// A field header: the id as a delta from the previous field when it fits in 4 bits.
    fn header(self: *Self, id: i16, kind: ThriftType) !void {
        const delta = id - self.last_field;
        if (delta > 0 and delta <= 15) {
            try self.out.append(self.allocator, @as(u8, @intCast(delta)) << 4 | @intFromEnum(kind));
        } else {
            try self.out.append(self.allocator, @intFromEnum(kind));
            try self.varint(zigzag(id));
        }
        self.last_field = id;
    }

    fn varint(self: *Self, value: u64) !void {
        var v = value;
        while (v >= 0x80) : (v >>= 7) try self.out.append(self.allocator, @as(u8, @truncate(v)) | 0x80);
        try self.out.append(self.allocator, @truncate(v));
    }

    fn zigzag(value: i64) u64 {
        return @bitCast((value << 1) ^ (value >> 63));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    // The logger ZIGSTYLE asks every file for; encoding has nothing to log.
    _ = log;
}

test "thrift writer output reads back through the reader" {
    const allocator = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(allocator);
    var w = ThriftWriter.init(allocator, &bytes);
    try w.beginStruct();
    try w.int32(1, -3);
    try w.list(2, .binary, 2);
    try w.listBinary("a");
    try w.listBinary("text");
    try w.structField(3);
    try w.int64(20, 1 << 40); // a delta over 15: the long header form
    try w.endStruct();
    try w.list(4, .@"struct", 1);
    try w.beginStruct();
    try w.int32(1, 7);
    try w.endStruct();
    try w.binary(5, "done");
    try w.endStruct();

    var r = mod.ThriftReader.init(bytes.items);
    var f = try r.field();
    try std.testing.expectEqual(@as(i16, 1), f.id);
    try std.testing.expectEqual(@as(i32, -3), try r.readI32());
    f = try r.field();
    const strings = try r.listHeader();
    try std.testing.expectEqual(@as(usize, 2), strings.len);
    try std.testing.expectEqualStrings("a", try r.readBinary());
    try std.testing.expectEqualStrings("text", try r.readBinary());
    f = try r.field();
    try std.testing.expectEqual(@as(i16, 3), f.id);
    var saved = r.beginStruct();
    f = try r.field();
    try std.testing.expectEqual(@as(i16, 20), f.id);
    try std.testing.expectEqual(@as(i64, 1 << 40), try r.readI64());
    try std.testing.expectEqual(ThriftType.stop, (try r.field()).type);
    r.endStruct(saved);
    f = try r.field();
    try std.testing.expectEqual(@as(i16, 4), f.id);
    _ = try r.listHeader();
    saved = r.beginStruct();
    _ = try r.field();
    try std.testing.expectEqual(@as(i32, 7), try r.readI32());
    try std.testing.expectEqual(ThriftType.stop, (try r.field()).type);
    r.endStruct(saved);
    f = try r.field();
    try std.testing.expectEqual(@as(i16, 5), f.id);
    try std.testing.expectEqualStrings("done", try r.readBinary());
    try std.testing.expectEqual(ThriftType.stop, (try r.field()).type);
}
