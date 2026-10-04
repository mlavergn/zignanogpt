const std = @import("std");
const log = std.log.scoped(.zignanogpt_cpu_attention);
const mod = @import("module.zig");

/// Queries per work item.
const query_block = 16;

/// Causal sliding-window GQA attention on the CPU; see `AttentionOptions`.
///
/// One pass per query with an online softmax (running max and sum, the output
/// rescaled when the max grows), so no score buffer is needed whatever the
/// window. Work items are blocks of queries of one `(batch, head)`.
pub const CpuAttention = struct {
    const Self = @This();

    /// Largest head dimension the per-query accumulator holds.
    pub const max_head_dim = 256;

    out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    /// Optional `[B, H, Tq]` log-sum-exp of the scaled scores (for backward).
    lse: ?[]f32,
    dims: mod.AttentionOptions.Dims,
    window: usize,

    /// Runs the attention, spread over `parallel`.
    ///
    /// Parameters:
    /// - `self`: the problem.
    /// - `parallel`: the dispatcher.
    ///
    /// Return: nothing; `error.HeadDimTooLarge` past `max_head_dim`, or
    /// `error.Canceled`.
    pub fn run(self: *const Self, parallel: *const mod.Parallel) !void {
        if (self.dims.d > max_head_dim) {
            log.err("head dim {d} exceeds {d}", .{ self.dims.d, max_head_dim });
            return error.HeadDimTooLarge;
        }
        const blocks = (self.dims.tq + query_block - 1) / query_block;
        try parallel.run(self.dims.b * self.dims.h * blocks, self, work);
    }

    /// `Parallel` work item: one block of queries of one (batch, head).
    fn work(self: *const Self, item: usize, worker: usize) void {
        _ = worker;
        const d = self.dims;
        const blocks = (d.tq + query_block - 1) / query_block;
        const bh = item / blocks;
        const b = bh / d.h;
        const head = bh % d.h;
        const kv_head = head / (d.h / d.hkv);
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(d.d)));
        const first = (item % blocks) * query_block;
        for (first..@min(first + query_block, d.tq)) |t| {
            const pos = d.keys - d.tq + t;
            const lo = if (pos > self.window) pos - self.window else 0;
            const q_row = self.q[((b * d.tq + t) * d.h + head) * d.d ..][0..d.d];
            var acc_buf: [max_head_dim]f32 = undefined;
            const acc = acc_buf[0..d.d];
            @memset(acc, 0);
            var max: f32 = -std.math.inf(f32);
            var sum: f32 = 0;
            for (lo..pos + 1) |j| {
                const base = ((b * d.tk + j) * d.hkv + kv_head) * d.d;
                const s = dot(q_row, self.k[base..][0..d.d]) * scale;
                if (s > max) {
                    const correction = @exp(max - s);
                    sum *= correction;
                    for (acc) |*a| a.* *= correction;
                    max = s;
                }
                const p = @exp(s - max);
                sum += p;
                for (acc, self.v[base..][0..d.d]) |*a, x| a.* += p * x;
            }
            const out_row = self.out[((b * d.tq + t) * d.h + head) * d.d ..][0..d.d];
            for (out_row, acc) |*o, a| o.* = a / sum;
            if (self.lse) |lse| lse[(b * d.h + head) * d.tq + t] = max + @log(sum);
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

test "cpu attention with a window of zero copies each query's own value" {
    // With window 0 a query sees only itself, so softmax weight 1 on its own key.
    const dims = mod.AttentionOptions.Dims{ .b = 1, .tq = 3, .tk = 3, .keys = 3, .h = 2, .hkv = 1, .d = 2 };
    const q = [_]f32{ 1, 0, 0, 1, 1, 1, 2, 2, 3, 0, 0, 3 };
    const k = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const v = [_]f32{ 10, 20, 30, 40, 50, 60 };
    var out: [12]f32 = undefined;
    var lse: [6]f32 = undefined;
    const attention = mod.CpuAttention{ .out = &out, .q = &q, .k = &k, .v = &v, .lse = &lse, .dims = dims, .window = 0 };
    try attention.run(&mod.Parallel.init(std.testing.io, 2));
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 10, 20, 30, 40, 30, 40, 50, 60, 50, 60 }, &out);
    // lse of a single score is the score itself: q(t=0,h=0) . k(0) / sqrt(2)
    try std.testing.expectApproxEqAbs(1.0 / @sqrt(@as(f32, 2.0)), lse[0], 1e-6);
}
