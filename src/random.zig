const std = @import("std");
const log = std.log.scoped(.zignanogpt_random);
const mod = @import("module.zig");

/// Seeded random numbers for weight init, sampling, and tests.
///
/// Not bit-compatible with PyTorch's generator: parity tests load weights from
/// fixtures instead of reproducing torch's draws. Same seed, same sequence.
pub const Random = struct {
    const Self = @This();

    prng: std.Random.DefaultPrng,

    /// Creates a generator.
    ///
    /// Parameters:
    /// - `seed`: the seed.
    ///
    /// Return: the generator.
    pub fn init(seed: u64) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        return Self{ .prng = std.Random.DefaultPrng.init(seed) };
    }

    /// The std interface, for anything else `std.Random` offers.
    ///
    /// Parameters:
    /// - `self`: the generator.
    ///
    /// Return: the interface, borrowing `self`.
    pub fn random(self: *Self) std.Random {
        return self.prng.random();
    }

    /// A uniform draw from `[lo, hi)`.
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `lo`: lower bound.
    /// - `hi`: upper bound.
    ///
    /// Return: the draw.
    pub fn uniform(self: *Self, lo: f32, hi: f32) f32 {
        return lo + (hi - lo) * self.random().float(f32);
    }

    /// A normal draw.
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `mean`: the mean.
    /// - `stddev`: the standard deviation.
    ///
    /// Return: the draw.
    pub fn normal(self: *Self, mean: f32, stddev: f32) f32 {
        return mean + stddev * self.random().floatNorm(f32);
    }

    /// Fills `out` with uniform draws from `[lo, hi)`.
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `out`: the destination.
    /// - `lo`: lower bound.
    /// - `hi`: upper bound.
    ///
    /// Return: nothing.
    pub fn fillUniform(self: *Self, out: []f32, lo: f32, hi: f32) void {
        for (out) |*x| x.* = self.uniform(lo, hi);
    }

    /// Fills `out` with normal draws.
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `out`: the destination.
    /// - `mean`: the mean.
    /// - `stddev`: the standard deviation.
    ///
    /// Return: nothing.
    pub fn fillNormal(self: *Self, out: []f32, mean: f32, stddev: f32) void {
        for (out) |*x| x.* = self.normal(mean, stddev);
    }

    /// A uniform integer in `[0, n)`.
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `n`: the bound; must be positive.
    ///
    /// Return: the draw.
    pub fn below(self: *Self, n: usize) usize {
        return self.random().uintLessThan(usize, n);
    }

    /// Draws an index with probability proportional to its weight.
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `weights`: non-negative weights, not necessarily normalized, not all zero.
    ///
    /// Return: the drawn index.
    pub fn categorical(self: *Self, weights: []const f32) usize {
        var total: f64 = 0;
        for (weights) |w| total += w;
        const target = self.random().float(f64) * total;
        var cumulative: f64 = 0;
        for (weights, 0..) |w, i| {
            cumulative += w;
            if (target < cumulative) return i;
        }
        // Rounding can leave `target` at the very top; take the last non-zero weight.
        var i = weights.len;
        while (i > 0) {
            i -= 1;
            if (weights[i] > 0) return i;
        }
        return weights.len - 1;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "random is reproducible from its seed" {
    var a = mod.Random.init(42);
    var b = mod.Random.init(42);
    for (0..100) |_| try std.testing.expectEqual(a.normal(0, 1), b.normal(0, 1));
}

test "random draws have the requested moments" {
    var rng = mod.Random.init(1);
    var values: [20000]f32 = undefined;
    rng.fillNormal(&values, 2, 0.5);
    var mean: f64 = 0;
    for (values) |v| mean += v;
    mean /= values.len;
    var variance: f64 = 0;
    for (values) |v| variance += (v - mean) * (v - mean);
    variance /= values.len;
    try std.testing.expectApproxEqAbs(@as(f64, 2), mean, 0.02);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), variance, 0.02);

    rng.fillUniform(&values, -3, -1);
    for (values) |v| try std.testing.expect(v >= -3 and v < -1);
}

test "random categorical follows the weights" {
    var rng = mod.Random.init(7);
    var counts = [_]usize{ 0, 0, 0 };
    for (0..30000) |_| counts[rng.categorical(&.{ 1, 0, 3 })] += 1;
    try std.testing.expectEqual(@as(usize, 0), counts[1]);
    const ratio = @as(f64, @floatFromInt(counts[2])) / @as(f64, @floatFromInt(counts[0]));
    try std.testing.expectApproxEqAbs(@as(f64, 3), ratio, 0.2);
    try std.testing.expect(rng.below(5) < 5);
}
