const std = @import("std");
const log = std.log.scoped(.zignanogpt_scalar);
const mod = @import("module.zig");

/// A scalar operand resolved on the device: `factor * tensor[0]`, or just
/// `factor` when there is no tensor.
///
/// Learnable scalars (`resid_lambdas[i]`, `backout_lambda`, ...) live in device
/// tensors; passing them by reference keeps an op from needing a host read.
pub const Scalar = struct {
    const Self = @This();

    /// One-element tensor (or a one-row view); null means 1.
    tensor: ?mod.Tensor = null,
    factor: f32 = 1,

    /// A constant.
    ///
    /// Parameters:
    /// - `value`: the constant.
    ///
    /// Return: the scalar.
    pub fn constant(value: f32) Self {
        return Self{ .factor = value };
    }

    /// The first element of `tensor`, times `factor`.
    ///
    /// Parameters:
    /// - `tensor`: a tensor with at least one element; only element 0 is read.
    /// - `factor`: a constant multiplier.
    ///
    /// Return: the scalar.
    pub fn of(tensor: mod.Tensor, factor: f32) Self {
        return Self{ .tensor = tensor, .factor = factor };
    }

    /// Checks the tensor (if any) is a non-empty f32.
    ///
    /// Parameters:
    /// - `self`: the scalar.
    ///
    /// Return: nothing; `error.DtypeMismatch` or `error.ShapeMismatch`.
    pub fn validate(self: Self) !void {
        const tensor = self.tensor orelse return;
        if (tensor.dtype != .f32) return error.DtypeMismatch;
        if (tensor.numel() == 0) {
            log.debug("scalar operand is an empty tensor", .{});
            return error.ShapeMismatch;
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "scalar constants carry no tensor" {
    const s = mod.Scalar.constant(-0.5);
    try std.testing.expect(s.tensor == null);
    try std.testing.expectEqual(@as(f32, -0.5), s.factor);
    try s.validate();
}
