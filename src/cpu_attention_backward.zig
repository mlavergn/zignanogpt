const std = @import("std");
const log = std.log.scoped(.zignanogpt_cpu_attention_backward);
const mod = @import("module.zig");

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

    /// `Parallel` work item: one (batch, kv head).
    fn work(self: *const Self, item: usize, worker: usize) void {
        _ = worker;
        const d = self.dims;
        const b = item / d.hkv;
        const kv_head = item % d.hkv;
        const group = d.h / d.hkv;
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(d.d)));

        for (0..d.tk) |j| {
            const base = ((b * d.tk + j) * d.hkv + kv_head) * d.d;
            @memset(self.dk[base..][0..d.d], 0);
            @memset(self.dv[base..][0..d.d], 0);
        }
        for (kv_head * group..(kv_head + 1) * group) |head| {
            for (0..d.tq) |t| {
                const pos = d.keys - d.tq + t;
                const lo = if (pos > self.window) pos - self.window else 0;
                const row = ((b * d.tq + t) * d.h + head) * d.d;
                const q_row = self.q[row..][0..d.d];
                const g_row = self.dout[row..][0..d.d];
                const lse = self.lse[(b * d.h + head) * d.tq + t];
                const big_d = dot(g_row, self.out[row..][0..d.d]);
                var acc_buf: [mod.CpuAttention.max_head_dim]f32 = undefined;
                const acc = acc_buf[0..d.d];
                @memset(acc, 0);
                for (lo..pos + 1) |j| {
                    const base = ((b * d.tk + j) * d.hkv + kv_head) * d.d;
                    const k_row = self.k[base..][0..d.d];
                    const p = @exp(dot(q_row, k_row) * scale - lse);
                    for (self.dv[base..][0..d.d], g_row) |*o, g| o.* += p * g;
                    const ds = p * (dot(g_row, self.v[base..][0..d.d]) - big_d);
                    for (acc, k_row) |*a, kv| a.* += ds * kv;
                    for (self.dk[base..][0..d.d], q_row) |*o, qv| o.* += ds * scale * qv;
                }
                for (self.dq[row..][0..d.d], acc) |*o, a| o.* = a * scale;
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
