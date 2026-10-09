const std = @import("std");
const log = std.log.scoped(.zignanogpt_thrift_reader);
const mod = @import("../module.zig");

/// Thrift compact-protocol field and element types.
pub const ThriftType = enum(u4) {
    stop = 0,
    bool_true = 1,
    bool_false = 2,
    byte = 3,
    i16 = 4,
    i32 = 5,
    i64 = 6,
    double = 7,
    binary = 8,
    list = 9,
    set = 10,
    map = 11,
    @"struct" = 12,
    _,
};

/// A struct field header: its id and type (`.stop` ends the struct).
pub const ThriftField = struct { id: i16, type: ThriftType };

/// A list or set header.
pub const ThriftList = struct { len: usize, type: ThriftType };

/// Reads Thrift's compact protocol, the encoding of Parquet's metadata and
/// page headers: ULEB128 varints, zigzag integers, field-id deltas.
pub const ThriftReader = struct {
    const Self = @This();

    data: []const u8,
    pos: usize = 0,
    /// Field ids are deltas from the previous field in the same struct.
    last_field: i16 = 0,

    /// Reads from the start of `data`.
    pub fn init(data: []const u8) Self {
        return Self{ .data = data };
    }

    /// The next field header, or `.stop` at the end of a struct.
    ///
    /// Parameters:
    /// - `self`: the reader.
    ///
    /// Return: the header; `error.EndOfStream` on truncated input.
    pub fn field(self: *Self) !ThriftField {
        const byte = try self.readByte();
        const kind: ThriftType = @fromBackingInt(@intCast(@as(u4, @truncate(byte))));
        if (kind == .stop) return .{ .id = 0, .type = .stop };
        const delta: i16 = @intCast(byte >> 4);
        self.last_field = if (delta == 0) try self.readI16() else self.last_field + delta;
        return .{ .id = self.last_field, .type = kind };
    }

    /// Enters a nested struct; pair with `endStruct`.
    ///
    /// Parameters:
    /// - `self`: the reader.
    ///
    /// Return: the outer struct's field-id state, to restore.
    pub fn beginStruct(self: *Self) i16 {
        const saved = self.last_field;
        self.last_field = 0;
        return saved;
    }

    pub fn endStruct(self: *Self, saved: i16) void {
        self.last_field = saved;
    }

    pub fn readI16(self: *Self) !i16 {
        return @intCast(zigzag(try self.varint()));
    }

    pub fn readI32(self: *Self) !i32 {
        return std.math.cast(i32, zigzag(try self.varint())) orelse error.InvalidThrift;
    }

    pub fn readI64(self: *Self) !i64 {
        return zigzag(try self.varint());
    }

    /// A length-prefixed byte string, borrowed from the input.
    pub fn readBinary(self: *Self) ![]const u8 {
        const len = std.math.cast(usize, try self.varint()) orelse return error.InvalidThrift;
        if (len > self.data.len - self.pos) return error.EndOfStream;
        defer self.pos += len;
        return self.data[self.pos..][0..len];
    }

    /// A bool field's value, carried in its type.
    pub fn boolOf(t: ThriftType) !bool {
        return switch (t) {
            .bool_true => true,
            .bool_false => false,
            else => error.InvalidThrift,
        };
    }

    /// A list or set header.
    pub fn listHeader(self: *Self) !ThriftList {
        const byte = try self.readByte();
        var len: usize = byte >> 4;
        if (len == 15) len = std.math.cast(usize, try self.varint()) orelse return error.InvalidThrift;
        return .{ .len = len, .type = @fromBackingInt(@intCast(@as(u4, @truncate(byte)))) };
    }

    /// Skips a value of type `t` (and everything nested in it).
    ///
    /// Parameters:
    /// - `self`: the reader.
    /// - `t`: the value's type.
    ///
    /// Return: nothing; `error.InvalidThrift` on an unknown type.
    pub fn skip(self: *Self, t: ThriftType) anyerror!void {
        switch (t) {
            .bool_true, .bool_false => {},
            .byte => _ = try self.readByte(),
            .i16, .i32, .i64 => _ = try self.varint(),
            .double => {
                if (self.data.len - self.pos < 8) return error.EndOfStream;
                self.pos += 8;
            },
            .binary => _ = try self.readBinary(),
            .list, .set => {
                const header = try self.listHeader();
                for (0..header.len) |_| {
                    // Booleans inside a list are a byte each.
                    if (header.type == .bool_true or header.type == .bool_false) {
                        _ = try self.readByte();
                    } else {
                        try self.skip(header.type);
                    }
                }
            },
            .map => {
                const len = std.math.cast(usize, try self.varint()) orelse return error.InvalidThrift;
                if (len == 0) return;
                const types = try self.readByte();
                for (0..len) |_| {
                    try self.skip(@fromBackingInt(@intCast(@as(u4, @truncate(types >> 4)))));
                    try self.skip(@fromBackingInt(@intCast(@as(u4, @truncate(types)))));
                }
            },
            .@"struct" => {
                const saved = self.beginStruct();
                while (true) {
                    const f = try self.field();
                    if (f.type == .stop) break;
                    try self.skip(f.type);
                }
                self.endStruct(saved);
            },
            .stop, _ => {
                log.debug("cannot skip thrift type {d}", .{@backingInt(t)});
                return error.InvalidThrift;
            },
        }
    }

    fn readByte(self: *Self) !u8 {
        if (self.pos >= self.data.len) return error.EndOfStream;
        defer self.pos += 1;
        return self.data[self.pos];
    }

    /// ULEB128.
    fn varint(self: *Self) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        while (true) {
            const byte = try self.readByte();
            result |= @as(u64, byte & 0x7f) << shift;
            if (byte & 0x80 == 0) return result;
            if (shift >= 63) return error.InvalidThrift;
            shift += 7;
        }
    }

    fn zigzag(n: u64) i64 {
        return @as(i64, @bitCast(n >> 1)) ^ -@as(i64, @intCast(n & 1));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "thrift reader decodes compact structs, lists and skips" {
    // struct { 1: i32 = -3, 2: binary "hi", 5: list<i32> [1, 2], 20: bool true }, then stop
    const bytes = [_]u8{
        0x15, 0x05, // field 1 (delta 1), i32, zigzag(-3) = 5
        0x18, 0x02, 'h', 'i', // field 2, binary
        0x39, 0x25, 0x02, 0x04, // field 5 (delta 3), list of 2 i32: 1, 2
        0x01, 0x28, // field 20 (delta 0 -> explicit zigzag i16 20 = 40), bool true
        0x00, // stop
    };
    var r = mod.ThriftReader.init(&bytes);
    var f = try r.field();
    try std.testing.expectEqual(@as(i16, 1), f.id);
    try std.testing.expectEqual(@as(i32, -3), try r.readI32());
    f = try r.field();
    try std.testing.expectEqualStrings("hi", try r.readBinary());
    f = try r.field();
    try std.testing.expectEqual(@as(i16, 5), f.id);
    try r.skip(f.type);
    f = try r.field();
    try std.testing.expectEqual(@as(i16, 20), f.id);
    try std.testing.expect(try mod.ThriftReader.boolOf(f.type));
    try std.testing.expectEqual(mod.ThriftType.stop, (try r.field()).type);
}
