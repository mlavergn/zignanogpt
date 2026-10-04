const std = @import("std");
const log = std.log.scoped(.zignanogpt_shape);
const mod = @import("module.zig");

/// A tensor's dimensions, row-major, rank 0 to 4 (enough for `(B, T, H, D)`).
pub const Shape = struct {
    const Self = @This();

    pub const max_rank = 4;

    dims: [max_rank]usize = .{ 0, 0, 0, 0 },
    rank: u8 = 0,

    /// Builds a shape from its dimensions.
    ///
    /// Parameters:
    /// - `dims`: the dimensions, outermost first; at most `max_rank`.
    ///
    /// Return: the shape; `error.RankTooHigh` past `max_rank`.
    pub fn init(dims: []const usize) error{RankTooHigh}!Self {
        if (dims.len > max_rank) {
            log.debug("rank {d} exceeds {d}", .{ dims.len, max_rank });
            return error.RankTooHigh;
        }
        var self = Self{ .rank = @intCast(dims.len) };
        @memcpy(self.dims[0..dims.len], dims);
        return self;
    }

    /// The dimensions as a slice.
    ///
    /// Parameters:
    /// - `self`: the shape.
    ///
    /// Return: `rank` dimensions.
    pub fn slice(self: *const Self) []const usize {
        return self.dims[0..self.rank];
    }

    /// The element count: the product of the dimensions (1 for rank 0).
    ///
    /// Parameters:
    /// - `self`: the shape.
    ///
    /// Return: the count.
    pub fn numel(self: Self) usize {
        var count: usize = 1;
        for (self.slice()) |d| count *= d;
        return count;
    }

    /// The innermost dimension: a row's length when viewed as a matrix.
    ///
    /// Parameters:
    /// - `self`: the shape.
    ///
    /// Return: the last dimension (1 for rank 0).
    pub fn cols(self: Self) usize {
        return if (self.rank == 0) 1 else self.dims[self.rank - 1];
    }

    /// The product of every dimension but the last: the row count when viewed as a matrix.
    ///
    /// Parameters:
    /// - `self`: the shape.
    ///
    /// Return: the row count (1 for rank 0 and 1).
    pub fn rows(self: Self) usize {
        var count: usize = 1;
        if (self.rank > 1) {
            for (self.dims[0 .. self.rank - 1]) |d| count *= d;
        }
        return count;
    }

    /// Compares two shapes dimension by dimension.
    ///
    /// Parameters:
    /// - `self`: one shape.
    /// - `other`: the other.
    ///
    /// Return: true when rank and dimensions match.
    pub fn eql(self: Self, other: Self) bool {
        return self.rank == other.rank and std.mem.eql(usize, self.slice(), other.slice());
    }

    /// Writes the shape as `[d0, d1, ...]` (for `{f}`).
    ///
    /// Parameters:
    /// - `self`: the shape.
    /// - `writer`: the destination.
    ///
    /// Return: nothing; propagates write errors.
    pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.writeByte('[');
        for (self.slice(), 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{d});
        }
        try writer.writeByte(']');
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "shape counts elements, rows and cols" {
    const shape = try mod.Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), shape.numel());
    try std.testing.expectEqual(@as(usize, 6), shape.rows());
    try std.testing.expectEqual(@as(usize, 4), shape.cols());

    const scalar = try mod.Shape.init(&.{});
    try std.testing.expectEqual(@as(usize, 1), scalar.numel());
    try std.testing.expectEqual(@as(usize, 1), scalar.rows());
}

test "shape compares and formats" {
    const a = try mod.Shape.init(&.{ 2, 3 });
    const b = try mod.Shape.init(&.{ 2, 3 });
    const c = try mod.Shape.init(&.{ 3, 2 });
    try std.testing.expect(a.eql(b));
    try std.testing.expect(!a.eql(c));
    try std.testing.expectError(error.RankTooHigh, mod.Shape.init(&.{ 1, 1, 1, 1, 1 }));

    var buffer: [32]u8 = undefined;
    const text = try std.fmt.bufPrint(&buffer, "{f}", .{a});
    try std.testing.expectEqualStrings("[2, 3]", text);
}
