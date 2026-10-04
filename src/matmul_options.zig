const std = @import("std");
const log = std.log.scoped(.zignanogpt_matmul_options);
const mod = @import("module.zig");

/// How `Backend.matmul` combines its operands: `C = alpha * op(A) @ op(B)`, or
/// `C += ...` with `accumulate`.
///
/// Operands are viewed as matrices: rows = product of all but the last
/// dimension, cols = the last. The transposes cover every product the model
/// needs without copies: `x @ W^T` (forward, `transpose_b`), `dy @ W` (input
/// grad), `dy^T @ x` (weight grad, `transpose_a` + `accumulate`), and Muon's
/// `X^T X` / `X X^T`.
pub const MatmulOptions = struct {
    /// Use A transposed: A is stored `[K, M]`.
    transpose_a: bool = false,
    /// Use B transposed: B is stored `[N, K]` (PyTorch's `Linear.weight` layout).
    transpose_b: bool = false,
    /// Scales the product.
    alpha: f32 = 1,
    /// Add into C instead of overwriting it.
    accumulate: bool = false,

    /// The problem size `(m, n, k)` of `c = op(a) @ op(b)`, validated.
    ///
    /// Shared by every backend so shape rules cannot drift between them.
    ///
    /// Parameters:
    /// - `self`: the options (the transposes decide which dimension is which).
    /// - `c`: the result's shape, viewed as `[M, N]`.
    /// - `a`: A's shape, viewed as `[M, K]` (`[K, M]` transposed).
    /// - `b`: B's shape, viewed as `[K, N]` (`[N, K]` transposed).
    ///
    /// Return: `.{ m, n, k }`; `error.ShapeMismatch` when they do not chain.
    pub fn dims(self: MatmulOptions, c: mod.Shape, a: mod.Shape, b: mod.Shape) error{ShapeMismatch}![3]usize {
        const m = if (self.transpose_a) a.cols() else a.rows();
        const k = if (self.transpose_a) a.rows() else a.cols();
        const kb = if (self.transpose_b) b.cols() else b.rows();
        const n = if (self.transpose_b) b.rows() else b.cols();
        if (k != kb or c.rows() != m or c.cols() != n) {
            log.debug("matmul shapes do not chain: c {f} = a {f} @ b {f} (ta={}, tb={})", .{ c, a, b, self.transpose_a, self.transpose_b });
            return error.ShapeMismatch;
        }
        return .{ m, n, k };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "matmul options default to a plain overwrite" {
    const options = mod.MatmulOptions{};
    try std.testing.expect(!options.transpose_a and !options.transpose_b and !options.accumulate);
    try std.testing.expectEqual(@as(f32, 1), options.alpha);
}

test "matmul options resolve and check dimensions" {
    const x = try mod.Shape.init(&.{ 2, 5, 8 }); // viewed as [10, 8]
    const w = try mod.Shape.init(&.{ 3, 8 });
    const y = try mod.Shape.init(&.{ 10, 3 });
    const dims = try (mod.MatmulOptions{ .transpose_b = true }).dims(y, x, w);
    try std.testing.expectEqual([3]usize{ 10, 3, 8 }, dims);
    try std.testing.expectError(error.ShapeMismatch, (mod.MatmulOptions{}).dims(y, x, w));
}
