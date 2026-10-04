const std = @import("std");
const builtin = @import("builtin");
const log = std.log.scoped(.zignanogpt_cpu_matmul);
const mod = @import("module.zig");

/// Rows of C one kernel call produces.
const mr = 6;
/// Columns of C one kernel call produces: one `@Vector(16, f32)`.
const nr = 16;
/// Depth of one packed block (K direction).
const kc = 256;
/// Rows of C per tile: a multiple of `mr`.
const tm = 96;
/// Columns of C per tile: a multiple of `nr`.
const tn = 256;

const Vec = @Vector(nr, f32);

/// Whether the target fuses multiply-add; without it `@mulAdd` is a libcall.
const has_fma = switch (builtin.cpu.arch) {
    .aarch64 => true,
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    else => false,
};

/// One CPU matrix product `C (+)= alpha * op(A) @ op(B)`, GotoBLAS-style.
///
/// C is cut into `tm x tn` tiles, run in parallel. Per tile and per `kc` block
/// of K, the tile's slices of A and B are packed into contiguous panels (the
/// packing absorbs both transposes, so one kernel serves all four layouts),
/// then a `mr x nr` register-blocked kernel sweeps them. With `mr = 6`,
/// `nr = 16` the accumulators fit the vector registers of both NEON (32 x 128
/// bit) and AVX2 (16 x 256 bit).
pub const CpuMatmul = struct {
    const Self = @This();

    /// f32 slots of packing scratch each worker needs.
    pub const scratch_len = tm * kc + kc * tn;

    c: []f32,
    a: []const f32,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
    options: mod.MatmulOptions,
    /// `scratch_len` floats per worker.
    scratch: []f32,

    /// Runs the product, tiles spread over `parallel`.
    ///
    /// Parameters:
    /// - `self`: the problem; slices sized for `m`, `n`, `k`.
    /// - `parallel`: the dispatcher; `scratch` must hold `scratch_len` per thread.
    ///
    /// Return: nothing; `error.Canceled` when the wait is canceled.
    pub fn run(self: *const Self, parallel: *const mod.Parallel) std.Io.Cancelable!void {
        if (self.m == 0 or self.n == 0) return;
        if (self.k == 0) {
            log.debug("matmul with empty inner dimension: {d}x{d}", .{ self.m, self.n });
            if (!self.options.accumulate) @memset(self.c[0 .. self.m * self.n], 0);
            return;
        }
        const tiles = divCeil(self.m, tm);
        const tiles_n = divCeil(self.n, tn);
        try parallel.run(tiles * tiles_n, self, tileWork);
    }

    /// `Parallel` work item: one tile.
    fn tileWork(self: *const Self, index: usize, worker: usize) void {
        self.tile(index, worker);
    }

    /// Computes one `tm x tn` tile of C.
    fn tile(self: *const Self, index: usize, worker: usize) void {
        const tiles_n = divCeil(self.n, tn);
        const row0 = (index / tiles_n) * tm;
        const col0 = (index % tiles_n) * tn;
        const rows = @min(tm, self.m - row0);
        const cols = @min(tn, self.n - col0);
        const scratch = self.scratch[worker * scratch_len ..][0..scratch_len];
        const pack_a = scratch[0 .. tm * kc];
        const pack_b = scratch[tm * kc ..];

        var k0: usize = 0;
        while (k0 < self.k) : (k0 += kc) {
            const depth = @min(kc, self.k - k0);
            self.packA(pack_a, row0, rows, k0, depth);
            self.packB(pack_b, col0, cols, k0, depth);
            const overwrite = k0 == 0 and !self.options.accumulate;
            var p: usize = 0;
            while (p < rows) : (p += mr) {
                const panel_a = pack_a[(p / mr) * mr * depth ..].ptr;
                var q: usize = 0;
                while (q < cols) : (q += nr) {
                    const panel_b = pack_b[(q / nr) * nr * depth ..].ptr;
                    const acc = kernel(depth, panel_a, panel_b);
                    self.store(acc, row0 + p, col0 + q, @min(mr, rows - p), @min(nr, cols - q), overwrite);
                }
            }
        }
    }

    /// Packs `op(A)[row0 .. row0 + rows, k0 .. k0 + depth]` into `mr`-row panels,
    /// k-major within a panel; rows past the edge are zero.
    fn packA(self: *const Self, dst: []f32, row0: usize, rows: usize, k0: usize, depth: usize) void {
        const panels = divCeil(rows, mr);
        for (0..panels) |p| {
            const panel = dst[p * mr * depth ..][0 .. mr * depth];
            for (0..mr) |r| {
                const i = p * mr + r;
                if (i >= rows) {
                    for (0..depth) |kk| panel[kk * mr + r] = 0;
                } else if (self.options.transpose_a) {
                    // A stored [K, M]: element (row, kk) at a[kk * m + row]
                    for (0..depth) |kk| panel[kk * mr + r] = self.a[(k0 + kk) * self.m + row0 + i];
                } else {
                    const src = self.a[(row0 + i) * self.k + k0 ..][0..depth];
                    for (src, 0..) |value, kk| panel[kk * mr + r] = value;
                }
            }
        }
    }

    /// Packs `op(B)[k0 .. k0 + depth, col0 .. col0 + cols]` into `nr`-column
    /// panels, k-major within a panel; columns past the edge are zero.
    fn packB(self: *const Self, dst: []f32, col0: usize, cols: usize, k0: usize, depth: usize) void {
        const panels = divCeil(cols, nr);
        for (0..panels) |q| {
            const panel = dst[q * nr * depth ..][0 .. nr * depth];
            const j = col0 + q * nr;
            const width = @min(nr, cols - q * nr);
            if (self.options.transpose_b) {
                // B stored [N, K]: element (kk, col) at b[col * k + kk]
                for (0..nr) |col| {
                    if (col >= width) {
                        for (0..depth) |kk| panel[kk * nr + col] = 0;
                        continue;
                    }
                    const src = self.b[(j + col) * self.k + k0 ..][0..depth];
                    for (src, 0..) |value, kk| panel[kk * nr + col] = value;
                }
            } else {
                for (0..depth) |kk| {
                    const out = panel[kk * nr ..][0..nr];
                    @memcpy(out[0..width], self.b[(k0 + kk) * self.n + j ..][0..width]);
                    @memset(out[width..], 0);
                }
            }
        }
    }

    /// The register-blocked core: `mr x nr` outer products over `depth`.
    inline fn kernel(depth: usize, panel_a: [*]const f32, panel_b: [*]const f32) [mr]Vec {
        var acc: [mr]Vec = @splat(@as(Vec, @splat(0)));
        for (0..depth) |kk| {
            const bv: Vec = panel_b[kk * nr ..][0..nr].*;
            inline for (0..mr) |r| {
                const av: Vec = @splat(panel_a[kk * mr + r]);
                acc[r] = if (has_fma) @mulAdd(Vec, av, bv, acc[r]) else acc[r] + av * bv;
            }
        }
        return acc;
    }

    /// Writes (or adds) `alpha * acc` into C at `(i, j)`, clipped to `rows x cols`.
    fn store(self: *const Self, acc: [mr]Vec, i: usize, j: usize, rows: usize, cols: usize, overwrite: bool) void {
        const alpha: Vec = @splat(self.options.alpha);
        for (0..rows) |r| {
            const value = acc[r] * alpha;
            const row = self.c[(i + r) * self.n + j ..];
            if (cols == nr) {
                const dst: *[nr]f32 = row[0..nr];
                dst.* = if (overwrite) value else @as(Vec, dst.*) + value;
            } else {
                const values: [nr]f32 = value;
                for (row[0..cols], values[0..cols]) |*dst, v| dst.* = if (overwrite) v else dst.* + v;
            }
        }
    }
};

/// `ceil(a / b)` for a positive `b`.
fn divCeil(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// Unit Tests

/// Reference `op(A) @ op(B)` in f64.
fn naive(out: []f64, a: []const f32, b: []const f32, m: usize, n: usize, k: usize, options: mod.MatmulOptions) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0;
            for (0..k) |kk| {
                const av = if (options.transpose_a) a[kk * m + i] else a[i * k + kk];
                const bv = if (options.transpose_b) b[j * k + kk] else b[kk * n + j];
                sum += @as(f64, av) * @as(f64, bv);
            }
            out[i * n + j] = sum;
        }
    }
}

fn checkCase(m: usize, n: usize, k: usize, options: mod.MatmulOptions, threads: usize) !void {
    const allocator = std.testing.allocator;
    var rng = mod.Random.init(m * 7919 + n * 104729 + k);
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);
    const expected = try allocator.alloc(f64, m * n);
    defer allocator.free(expected);
    const scratch = try allocator.alloc(f32, threads * mod.CpuMatmul.scratch_len);
    defer allocator.free(scratch);

    rng.fillUniform(a, -1, 1);
    rng.fillUniform(b, -1, 1);
    rng.fillUniform(c, -1, 1);
    naive(expected, a, b, m, n, k, options);
    for (expected, c) |*e, prior| {
        e.* *= options.alpha;
        if (options.accumulate) e.* += prior;
    }

    const parallel = mod.Parallel.init(std.testing.io, threads);
    const matmul = mod.CpuMatmul{ .c = c, .a = a, .b = b, .m = m, .n = n, .k = k, .options = options, .scratch = scratch };
    try matmul.run(&parallel);
    const tolerance = 1e-5 * @as(f64, @floatFromInt(k + 1));
    for (expected, c) |e, got| try std.testing.expectApproxEqAbs(e, @as(f64, got), tolerance);
}

test "cpu matmul matches the reference for every layout and edge size" {
    const sizes = [_][3]usize{ .{ 1, 1, 1 }, .{ 5, 7, 3 }, .{ 6, 16, 256 }, .{ 97, 257, 300 }, .{ 200, 33, 513 } };
    for (sizes) |s| {
        for ([_]bool{ false, true }) |ta| {
            for ([_]bool{ false, true }) |tb| {
                try checkCase(s[0], s[1], s[2], .{ .transpose_a = ta, .transpose_b = tb }, 3);
            }
        }
    }
}

test "cpu matmul scales and accumulates" {
    try checkCase(50, 40, 30, .{ .alpha = -0.5, .accumulate = true }, 2);
    try checkCase(50, 40, 300, .{ .transpose_a = true, .accumulate = true }, 1);
}

test "cpu matmul with empty K zeroes or keeps C" {
    var c = [_]f32{ 1, 2, 3, 4 };
    const parallel = mod.Parallel.init(std.testing.io, 1);
    var scratch: [1]f32 = undefined;
    var matmul = mod.CpuMatmul{ .c = &c, .a = &.{}, .b = &.{}, .m = 2, .n = 2, .k = 0, .options = .{ .accumulate = true }, .scratch = &scratch };
    try matmul.run(&parallel);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &c);
    matmul.options.accumulate = false;
    try matmul.run(&parallel);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &c);
}
