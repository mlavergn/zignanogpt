const std = @import("std");
const log = std.log.scoped(.zignanogpt_muon_params);
const mod = @import("module.zig");

/// The Polar Express coefficients `(a, b, c)` (num_iters 5, safety 2e-2,
/// cushion 2; https://arxiv.org/pdf/2505.16932), as in nanochat's `optim.py`.
pub const polar_express_coeffs = [5][3]f32{
    .{ 8.156554524902461, -22.48329292557795, 15.878769915207462 },
    .{ 4.042929935166739, -2.808917465908714, 0.5000178451051316 },
    .{ 3.8916678022926607, -2.772484153217685, 0.5060648178503393 },
    .{ 3.285753657755655, -2.3681294933425376, 0.46449024233003106 },
    .{ 2.3465413258596377, -1.7097828382687081, 0.42323551169305323 },
};

/// The hyperparameters of Muon's final stage (`muon_step_fused` after the
/// orthogonalization): Muon+ renormalization, NorMuon variance reduction,
/// cautious weight decay and the update.
pub const MuonParams = struct {
    const Self = @This();

    /// Already scaled by `max(1, rows / cols)^0.5`.
    lr: f32,
    weight_decay: f32,
    /// Decay of the factored second-moment buffer.
    beta2: f32,

    /// The second-moment buffer's shape for a `[rows, cols]` matrix: per row
    /// when the matrix is square or tall, per column when wide.
    ///
    /// Parameters:
    /// - `rows`: the matrix rows.
    /// - `cols`: the matrix columns.
    ///
    /// Return: `[rows, 1]` or `[1, cols]`.
    pub fn secondShape(rows: usize, cols: usize) [2]usize {
        log.debug("second moment for [{d}, {d}] reduces over {s}", .{ rows, cols, if (rows >= cols) "columns" else "rows" });
        return if (rows >= cols) .{ rows, 1 } else .{ 1, cols };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "muon second-moment shape follows the matrix aspect" {
    try std.testing.expectEqual([2]usize{ 8, 1 }, mod.MuonParams.secondShape(8, 4));
    try std.testing.expectEqual([2]usize{ 1, 8 }, mod.MuonParams.secondShape(4, 8));
    try std.testing.expectEqual(@as(usize, 5), mod.polar_express_coeffs.len);
}
