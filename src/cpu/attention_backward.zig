const std = @import("std");
const log = std.log.scoped(.zignanogpt_cpu_attention_backward);
const mod = @import("../module.zig");

const Vec = mod.CpuMath.Vec;
const lanes = mod.CpuMath.lanes;

/// Backward of `CpuAttention`.
///
/// Probabilities are recomputed from the saved log-sum-exp:
/// `p_ij = exp(s_ij - lse_i)`, `D_i = <dout_i, out_i>`,
/// `dv_j += p_ij dout_i`, `ds_ij = p_ij (<dout_i, v_j> - D_i)`,
/// `dq_i = scale * sum_j ds_ij k_j`, `dk_j += scale * ds_ij q_i`.
/// A work item is one `(batch, kv head)`: it owns that head's `dk`/`dv` rows
/// and the `dq` rows of every query head in its group, so nothing races and
/// the summation order is fixed.
pub const CpuAttentionBackward = struct {
    const Self = @This();

    dq: []f32,
    dk: []f32,
    dv: []f32,
    dout: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []const f32,
    lse: []const f32,
    dims: mod.AttentionOptions.Dims,
    window: usize,

    /// Runs the backward pass, spread over `parallel`.
    ///
    /// Parameters:
    /// - `self`: the problem.
    /// - `parallel`: the dispatcher.
    ///
    /// Return: nothing; `error.HeadDimTooLarge`, or `error.Canceled`.
    pub fn run(self: *const Self, parallel: *const mod.Parallel) !void {
        if (self.dims.d > mod.CpuAttention.max_head_dim) {
            log.err("head dim {d} exceeds {d}", .{ self.dims.d, mod.CpuAttention.max_head_dim });
            return error.HeadDimTooLarge;
        }
        try parallel.run(self.dims.b * self.dims.hkv, self, work);
    }

    /// `Parallel` work item: one (batch, kv head), its query heads in blocks
    /// of 16 (one per lane). Per key: the block's scores and `dout . v` as
    /// vector FMAs over the transposed queries and output gradients, `dq` the
    /// same way, and `dk`/`dv` row updates kept in the cache line of that key.
    fn work(self: *const Self, item: usize, worker: usize) void {
        _ = worker;
        const d = self.dims;
        const dd = d.d;
        const b = item / d.hkv;
        const kv_head = item % d.hkv;
        const group = d.h / d.hkv;
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(dd)));
        const zero: Vec = @splat(0);

        for (0..d.tk) |j| {
            const base = ((b * d.tk + j) * d.hkv + kv_head) * dd;
            @memset(self.dk[base..][0..dd], 0);
            @memset(self.dv[base..][0..dd], 0);
        }
        var qt_buf: [mod.CpuAttention.max_head_dim]Vec = undefined;
        var gt_buf: [mod.CpuAttention.max_head_dim]Vec = undefined;
        var dqt_buf: [mod.CpuAttention.max_head_dim]Vec = undefined;
        const qt = qt_buf[0..dd];
        const gt = gt_buf[0..dd];
        const dqt = dqt_buf[0..dd];
        for (kv_head * group..(kv_head + 1) * group) |head| {
            var first_query: usize = 0;
            while (first_query < d.tq) : (first_query += lanes) {
                const span = mod.CpuAttention.Span.init(d, self.window, first_query);
                var lse_lanes: [lanes]f32 = @splat(0);
                var d_lanes: [lanes]f32 = @splat(0);
                for (qt, gt, dqt, 0..) |*qv, *gv, *dv, c| {
                    var ql: [lanes]f32 = @splat(0);
                    var gl: [lanes]f32 = @splat(0);
                    for (0..span.count) |i| {
                        const row = ((b * d.tq + first_query + i) * d.h + head) * dd;
                        ql[i] = self.q[row + c];
                        gl[i] = self.dout[row + c];
                    }
                    qv.* = ql;
                    gv.* = gl;
                    dv.* = zero;
                }
                for (0..span.count) |i| {
                    const t = first_query + i;
                    const row = ((b * d.tq + t) * d.h + head) * dd;
                    lse_lanes[i] = self.lse[(b * d.h + head) * d.tq + t];
                    d_lanes[i] = dot(self.dout[row..][0..dd], self.out[row..][0..dd]);
                }
                const lse: Vec = lse_lanes;
                const big_d: Vec = d_lanes;
                const scale_v: Vec = @splat(scale);
                for (span.first..span.last + 1) |j| {
                    const base = ((b * d.tk + j) * d.hkv + kv_head) * dd;
                    const k_row = self.k[base..][0..dd];
                    const v_row = self.v[base..][0..dd];
                    var s: Vec = zero;
                    var dp: Vec = zero;
                    for (qt, gt, k_row, v_row) |qv, gv, kx, vx| {
                        s += qv * @as(Vec, @splat(kx));
                        dp += gv * @as(Vec, @splat(vx));
                    }
                    const p = @select(f32, span.valid(j), mod.CpuMath.exp(s * scale_v - lse), zero);
                    const ds = p * (dp - big_d);
                    const ds_scaled = ds * scale_v;
                    for (dqt, k_row) |*dv, kx| dv.* += ds_scaled * @as(Vec, @splat(kx));
                    const dk_row = self.dk[base..][0..dd];
                    const dv_row = self.dv[base..][0..dd];
                    const p_lanes: [lanes]f32 = p;
                    const ds_lanes: [lanes]f32 = ds_scaled;
                    for (0..span.count) |i| {
                        if (p_lanes[i] == 0 and ds_lanes[i] == 0) continue;
                        const row = ((b * d.tq + first_query + i) * d.h + head) * dd;
                        const g_row = self.dout[row..][0..dd];
                        const q_row = self.q[row..][0..dd];
                        for (dv_row, g_row) |*o, g| o.* += p_lanes[i] * g;
                        for (dk_row, q_row) |*o, qx| o.* += ds_lanes[i] * qx;
                    }
                }
                for (dqt, 0..) |dv, c| {
                    const column: [lanes]f32 = dv;
                    for (0..span.count) |i| self.dq[((b * d.tq + first_query + i) * d.h + head) * dd + c] = column[i];
                }
            }
        }
    }

    /// Dot product, 8 lanes at a time.
    fn dot(a: []const f32, b: []const f32) f32 {
        const V = @Vector(8, f32);
        var acc: V = @splat(0);
        var i: usize = 0;
        while (i + 8 <= a.len) : (i += 8) {
            const av: V = a[i..][0..8].*;
            const bv: V = b[i..][0..8].*;
            acc += av * bv;
        }
        var sum = @reduce(.Add, acc);
        while (i < a.len) : (i += 1) sum += a[i] * b[i];
        return sum;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "cpu attention backward with one key passes the gradient straight to v" {
    // Window 0: each query attends only its own key with probability 1, so
    // dv = dout (summed over the group), and dq = dk = 0 (softmax of one score).
    const dims = mod.AttentionOptions.Dims{ .b = 1, .tq = 2, .tk = 2, .keys = 2, .h = 2, .hkv = 1, .d = 2 };
    const q = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const k = [_]f32{ 1, 0, 0, 1 };
    const v = [_]f32{ 2, 3, 4, 5 };
    var out: [8]f32 = undefined;
    var lse: [4]f32 = undefined;
    const parallel = mod.Parallel.init(std.testing.io, 1);
    const forward = mod.CpuAttention{ .out = &out, .q = &q, .k = &k, .v = &v, .lse = &lse, .dims = dims, .window = 0 };
    try forward.run(&parallel);

    const dout = [_]f32{ 1, 1, 2, 2, 3, 3, 4, 4 };
    var dq: [8]f32 = undefined;
    var dk: [4]f32 = undefined;
    var dv: [4]f32 = undefined;
    const backward = mod.CpuAttentionBackward{ .dq = &dq, .dk = &dk, .dv = &dv, .dout = &dout, .q = &q, .k = &k, .v = &v, .out = &out, .lse = &lse, .dims = dims, .window = 0 };
    try backward.run(&parallel);
    try std.testing.expectEqualSlices(f32, &.{ 3, 3, 7, 7 }, &dv);
    for (dq) |x| try std.testing.expectApproxEqAbs(@as(f32, 0), x, 1e-6);
    for (dk) |x| try std.testing.expectApproxEqAbs(@as(f32, 0), x, 1e-6);
}
