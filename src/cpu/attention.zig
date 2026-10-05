const std = @import("std");
const log = std.log.scoped(.zignanogpt_cpu_attention);
const mod = @import("../module.zig");

/// Queries per work item: one vector lane each.
const query_block = mod.CpuMath.lanes;
/// Keys per online-softmax update.
const key_chunk = 16;

const Vec = mod.CpuMath.Vec;
const lanes = mod.CpuMath.lanes;

/// Causal sliding-window GQA attention on the CPU; see `AttentionOptions`.
///
/// Flash-attention style: a work item is a block of 16 queries of one
/// `(batch, head)`, one per SIMD lane (the queries are transposed so each
/// head-dimension step is one vector FMA). Keys are read once per block, in
/// chunks of 16 with one online-softmax rescale per chunk; no score buffer is
/// needed whatever the window.
pub const CpuAttention = struct {
    const Self = @This();

    /// Largest head dimension the per-block accumulators hold.
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

    /// The keys each lane of a block attends: `lo[i] <= j <= hi[i]`; lanes past
    /// the queries get an empty range.
    pub const Span = struct {
        lo: [lanes]i64,
        hi: [lanes]i64,
        first: usize,
        last: usize,
        count: usize,

        pub fn init(d: mod.AttentionOptions.Dims, window: usize, first_query: usize) Span {
            var lo: [lanes]i64 = @splat(0);
            var hi: [lanes]i64 = @splat(-1);
            const count = @min(lanes, d.tq - first_query);
            var first: usize = std.math.maxInt(usize);
            var last: usize = 0;
            for (0..count) |i| {
                const pos = d.keys - d.tq + first_query + i;
                const l = if (pos > window) pos - window else 0;
                lo[i] = @intCast(l);
                hi[i] = @intCast(pos);
                first = @min(first, l);
                last = @max(last, pos);
            }
            return .{ .lo = lo, .hi = hi, .first = first, .last = last, .count = count };
        }

        /// Which lanes see key `j`.
        pub fn valid(self: Span, j: usize) @Vector(lanes, bool) {
            const jv: @Vector(lanes, i64) = @splat(@intCast(j));
            const lo: @Vector(lanes, i64) = self.lo;
            const hi: @Vector(lanes, i64) = self.hi;
            return (jv >= lo) & (jv <= hi);
        }
    };

    /// `Parallel` work item: one block of queries of one (batch, head).
    fn work(self: *const Self, item: usize, worker: usize) void {
        _ = worker;
        const d = self.dims;
        const dd = d.d;
        const blocks = (d.tq + query_block - 1) / query_block;
        const bh = item / blocks;
        const b = bh / d.h;
        const head = bh % d.h;
        const kv_head = head / (d.h / d.hkv);
        const first_query = (item % blocks) * query_block;
        const span = Span.init(d, self.window, first_query);
        const scale: Vec = @splat(1 / @sqrt(@as(f32, @floatFromInt(dd))));

        // Transposed, pre-scaled queries and the output accumulator: [d][lane].
        var qt_buf: [max_head_dim]Vec = undefined;
        var acc_buf: [max_head_dim]Vec = undefined;
        const qt = qt_buf[0..dd];
        const acc = acc_buf[0..dd];
        for (qt, acc, 0..) |*qv, *av, c| {
            var lane: [lanes]f32 = @splat(0);
            for (0..span.count) |i| lane[i] = self.q[((b * d.tq + first_query + i) * d.h + head) * dd + c];
            qv.* = @as(Vec, lane) * scale;
            av.* = @splat(0);
        }
        const neg_inf: Vec = @splat(-std.math.inf(f32));
        var max: Vec = neg_inf;
        var sum: Vec = @splat(0);
        var scores: [key_chunk]Vec = undefined;
        var j0 = span.first;
        while (j0 <= span.last) : (j0 += key_chunk) {
            const n = @min(key_chunk, span.last + 1 - j0);
            // Scores for the chunk, masked, and the chunk's running max.
            var chunk_max = neg_inf;
            for (0..n) |k| {
                const key = self.k[((b * d.tk + j0 + k) * d.hkv + kv_head) * dd ..][0..dd];
                var s: Vec = @splat(0);
                for (qt, key) |qv, kx| s += qv * @as(Vec, @splat(kx));
                s = @select(f32, span.valid(j0 + k), s, neg_inf);
                scores[k] = s;
                chunk_max = @max(chunk_max, s);
            }
            const new_max = @max(max, chunk_max);
            // Rescale what is accumulated so far (lanes still empty keep 1).
            const correction = @select(f32, max == neg_inf, @as(Vec, @splat(1)), mod.CpuMath.exp(max - new_max));
            sum *= correction;
            for (acc) |*av| av.* *= correction;
            max = new_max;
            for (0..n) |k| {
                const p = @select(f32, scores[k] == neg_inf, @as(Vec, @splat(0)), mod.CpuMath.exp(scores[k] - max));
                sum += p;
                const value = self.v[((b * d.tk + j0 + k) * d.hkv + kv_head) * dd ..][0..dd];
                for (acc, value) |*av, vx| av.* += p * @as(Vec, @splat(vx));
            }
        }
        const sums: [lanes]f32 = sum;
        const maxes: [lanes]f32 = max;
        for (acc, 0..) |av, c| {
            const column: [lanes]f32 = av / sum;
            for (0..span.count) |i| self.out[((b * d.tq + first_query + i) * d.h + head) * dd + c] = column[i];
        }
        if (self.lse) |lse| {
            for (0..span.count) |i| lse[(b * d.h + head) * d.tq + first_query + i] = maxes[i] + @log(sums[i]);
        }
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
