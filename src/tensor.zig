const std = @import("std");
const log = std.log.scoped(.zignanogpt_tensor);
const mod = @import("module.zig");

/// Element type of a tensor. Weights and activations are f32; token ids and
/// targets are i32.
pub const Dtype = enum {
    const Self = @This();

    f32,
    i32,

    /// The dtype holding Zig type `T`.
    ///
    /// Parameters:
    /// - `T`: `f32` or `i32`.
    ///
    /// Return: the dtype; a compile error for any other type.
    pub fn of(comptime T: type) Self {
        return switch (T) {
            f32 => .f32,
            i32 => .i32,
            else => @compileError("unsupported tensor element type: " ++ @typeName(T)),
        };
    }

    /// Bytes per element.
    ///
    /// Parameters:
    /// - `self`: the dtype.
    ///
    /// Return: the size.
    pub fn size(self: Self) usize {
        return switch (self) {
            .f32, .i32 => 4,
        };
    }
};

/// A shaped, typed view of backend memory.
///
/// The storage is the backend's opaque `Buffer`: code outside a backend never
/// touches elements directly, only through backend ops and explicit host copies
/// (`Backend.upload` / `Backend.download`). Views (`reshape`, `rows`) share the
/// buffer; only the tensor `Backend.alloc` returned is passed to `Backend.free`.
pub const Tensor = struct {
    const Self = @This();

    buffer: mod.Backend.Buffer,
    /// Offset of the first element in the buffer, in elements.
    offset: usize = 0,
    dtype: Dtype,
    shape: mod.Shape,

    /// The element count.
    ///
    /// Parameters:
    /// - `self`: the tensor.
    ///
    /// Return: the count.
    pub fn numel(self: Self) usize {
        return self.shape.numel();
    }

    /// The same elements under another shape.
    ///
    /// Parameters:
    /// - `self`: the tensor.
    /// - `dims`: the new dimensions; their product must equal `numel`.
    ///
    /// Return: the view; `error.ShapeMismatch` when the counts differ.
    pub fn reshape(self: Self, dims: []const usize) !Self {
        const shape = try mod.Shape.init(dims);
        if (shape.numel() != self.numel()) {
            log.debug("reshape {f} to {f} changes the element count", .{ self.shape, shape });
            return error.ShapeMismatch;
        }
        var view = self;
        view.shape = shape;
        return view;
    }

    /// A contiguous slice along the outermost dimension.
    ///
    /// Parameters:
    /// - `self`: the tensor; rank at least 1.
    /// - `start`: the first index along dimension 0.
    /// - `count`: how many indices to keep.
    ///
    /// Return: the view; `error.OutOfBounds` when the range exceeds dimension 0.
    pub fn rows(self: Self, start: usize, count: usize) !Self {
        if (self.shape.rank == 0 or start + count > self.shape.dims[0]) {
            log.debug("rows {d}..{d} outside {f}", .{ start, start + count, self.shape });
            return error.OutOfBounds;
        }
        const stride = self.numel() / self.shape.dims[0];
        var view = self;
        view.offset += start * stride;
        view.shape.dims[0] = count;
        return view;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "tensor views share the buffer" {
    var backend = try mod.Backend.init(std.testing.allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();

    const t = try backend.alloc(.f32, &.{ 4, 3 });
    defer backend.free(t);
    try backend.upload(t, f32, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });

    const middle = try t.rows(1, 2);
    try std.testing.expectEqual(@as(usize, 3), middle.offset);
    var out: [6]f32 = undefined;
    try backend.download(middle, f32, &out);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 5, 6, 7, 8 }, &out);

    const flat = try t.reshape(&.{12});
    try std.testing.expectEqual(@as(usize, 12), flat.shape.cols());
    try std.testing.expectError(error.ShapeMismatch, t.reshape(&.{5}));
    try std.testing.expectError(error.OutOfBounds, t.rows(3, 2));
}

test "dtype maps Zig types" {
    try std.testing.expectEqual(mod.Dtype.f32, mod.Dtype.of(f32));
    try std.testing.expectEqual(mod.Dtype.i32, mod.Dtype.of(i32));
    try std.testing.expectEqual(@as(usize, 4), mod.Dtype.i32.size());
}
