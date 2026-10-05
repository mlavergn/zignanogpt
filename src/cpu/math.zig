const std = @import("std");

/// SIMD math shared by the CPU kernels: a vectorized `exp` (Cephes `expf`
/// polynomial, within 2 ulp of the libm result) and the reductions built on
/// it. LLVM lowers `@exp` on vectors to one libm call per lane, so the hot
/// softmax loops use this instead.
pub const CpuMath = struct {
    pub const lanes = 16;
    pub const Vec = @Vector(lanes, f32);

    /// `e^x` per lane; underflows to 0 below about -87.3, saturates above 88.7.
    pub fn exp(x_in: Vec) Vec {
        const hi: Vec = @splat(88.3762626647949);
        const lo: Vec = @splat(-88.3762626647949);
        var x = @min(@max(x_in, lo), hi);
        // n = round(x / ln 2); x - n ln 2 in two parts for accuracy.
        const fx = @floor(x * @as(Vec, @splat(1.44269504088896341)) + @as(Vec, @splat(0.5)));
        x = x - fx * @as(Vec, @splat(0.693359375)) - fx * @as(Vec, @splat(-2.12194440e-4));
        const z = x * x;
        var y: Vec = @splat(1.9875691500e-4);
        y = y * x + @as(Vec, @splat(1.3981999507e-3));
        y = y * x + @as(Vec, @splat(8.3334519073e-3));
        y = y * x + @as(Vec, @splat(4.1665795894e-2));
        y = y * x + @as(Vec, @splat(1.6666665459e-1));
        y = y * x + @as(Vec, @splat(5.0000001201e-1));
        y = y * z + x + @as(Vec, @splat(1.0));
        // 2^n through the exponent bits (n >= -127 after the clamp: 0 bits give 0).
        const n: @Vector(lanes, i32) = @intFromFloat(fx);
        const bits: @Vector(lanes, u32) = @bitCast((n + @as(@Vector(lanes, i32), @splat(127))) << @splat(23));
        return y * @as(Vec, @bitCast(bits));
    }

    /// `e^x` for one value (the vector path, one lane used).
    pub fn exp1(x: f32) f32 {
        return exp(@splat(x))[0];
    }

    /// `log(sum(exp(row)))`, the exponentials summed in f64.
    pub fn logSumExp(row: []const f32) f64 {
        const max = maxOf(row);
        var sum: @Vector(lanes, f64) = @splat(0);
        const m: Vec = @splat(max);
        var i: usize = 0;
        while (i + lanes <= row.len) : (i += lanes) {
            const v: Vec = row[i..][0..lanes].*;
            sum += @as(@Vector(lanes, f64), exp(v - m));
        }
        var total = @reduce(.Add, sum);
        while (i < row.len) : (i += 1) total += exp1(row[i] - max);
        return @as(f64, max) + @log(total);
    }

    /// The largest element (-inf for an empty row).
    pub fn maxOf(row: []const f32) f32 {
        var acc: Vec = @splat(-std.math.inf(f32));
        var i: usize = 0;
        while (i + lanes <= row.len) : (i += lanes) acc = @max(acc, @as(Vec, row[i..][0..lanes].*));
        var max = @reduce(.Max, acc);
        while (i < row.len) : (i += 1) max = @max(max, row[i]);
        return max;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "cpu math exp is within 2 ulp of libm over the softmax range" {
    var worst: f32 = 0;
    var x: f32 = -87;
    while (x < 88) : (x += 0.0137) {
        const want = @exp(x);
        const got = CpuMath.exp1(x);
        const ulp = std.math.floatEps(f32) * @max(@abs(want), std.math.floatMin(f32));
        worst = @max(worst, @abs(got - want) / ulp);
    }
    try std.testing.expect(worst <= 2);
    try std.testing.expectEqual(@as(f32, 1), CpuMath.exp1(0));
    try std.testing.expectEqual(@as(f32, 0), CpuMath.exp1(-std.math.inf(f32)));
    const row = [_]f32{ 1, 2, 3, -1, 0.5, 7, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 9, -3 };
    var want: f64 = 0;
    for (row) |r| want += @exp(@as(f64, r));
    try std.testing.expectApproxEqAbs(@log(want), CpuMath.logSumExp(&row), 1e-6);
}
