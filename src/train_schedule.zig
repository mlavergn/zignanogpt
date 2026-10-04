const std = @import("std");
const log = std.log.scoped(.zignanogpt_train_schedule);
const mod = @import("module.zig");

/// `base_train.py`'s schedules: LR multiplier (linear warmup, constant, linear
/// warmdown to `final_lr_frac`), Muon momentum (0.85 -> 0.97 over 400 steps,
/// 0.97 -> 0.90 over the warmdown) and Muon weight decay (cosine to zero).
pub const TrainSchedule = struct {
    const Self = @This();

    num_iterations: usize,
    warmup_steps: usize = 40,
    warmdown_ratio: f64 = 0.65,
    final_lr_frac: f64 = 0.05,
    /// The (already scaled) Muon weight decay at step 0.
    weight_decay: f64 = 0,

    /// Steps in the warmdown: Python's `round(warmdown_ratio * num_iterations)`.
    ///
    /// Parameters:
    /// - `self`: the schedule.
    ///
    /// Return: the count (ties round to even, as Python's `round`).
    pub fn warmdownSteps(self: Self) usize {
        const x = self.warmdown_ratio * @as(f64, @floatFromInt(self.num_iterations));
        const floor = @floor(x);
        const diff = x - floor;
        const rounded = if (diff > 0.5 or (diff == 0.5 and @mod(floor, 2) == 1)) floor + 1 else floor;
        return @intFromFloat(rounded);
    }

    /// `get_lr_multiplier(it)`.
    ///
    /// Parameters:
    /// - `self`: the schedule.
    /// - `it`: the step.
    ///
    /// Return: the multiplier on every group's initial LR.
    pub fn lrMultiplier(self: Self, it: usize) f64 {
        const warmdown = self.warmdownSteps();
        if (it < self.warmup_steps) return @as(f64, @floatFromInt(it + 1)) / @as(f64, @floatFromInt(self.warmup_steps));
        if (it + warmdown <= self.num_iterations) return 1.0;
        const progress = @as(f64, @floatFromInt(self.num_iterations - it)) / @as(f64, @floatFromInt(warmdown));
        return progress + (1 - progress) * self.final_lr_frac;
    }

    /// `get_muon_momentum(it)`.
    ///
    /// Parameters:
    /// - `self`: the schedule.
    /// - `it`: the step.
    ///
    /// Return: the Muon momentum.
    pub fn muonMomentum(self: Self, it: usize) f64 {
        const warmdown = self.warmdownSteps();
        const start = self.num_iterations - warmdown;
        if (it < 400) {
            const frac = @as(f64, @floatFromInt(it)) / 400;
            return (1 - frac) * 0.85 + frac * 0.97;
        }
        if (it >= start) {
            const progress = @as(f64, @floatFromInt(it - start)) / @as(f64, @floatFromInt(warmdown));
            return 0.97 * (1 - progress) + 0.90 * progress;
        }
        return 0.97;
    }

    /// `get_weight_decay(it)`.
    ///
    /// Parameters:
    /// - `self`: the schedule.
    /// - `it`: the step.
    ///
    /// Return: the Muon weight decay.
    pub fn weightDecay(self: Self, it: usize) f64 {
        const t = @as(f64, @floatFromInt(it)) / @as(f64, @floatFromInt(self.num_iterations));
        return self.weight_decay * 0.5 * (1 + @cos(std.math.pi * t));
    }

    /// Applies step `it`'s values to an optimizer.
    ///
    /// Parameters:
    /// - `self`: the schedule.
    /// - `optimizer`: receives the LR multiplier, momentum and weight decay.
    /// - `it`: the step.
    ///
    /// Return: nothing.
    pub fn apply(self: Self, optimizer: *mod.MuonAdamW, it: usize) void {
        log.debug("step {d}: lrm {d:.4}", .{ it, self.lrMultiplier(it) });
        optimizer.setSchedule(self.lrMultiplier(it), self.muonMomentum(it), self.weightDecay(it));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "train schedule matches base_train.py's formulas" {
    // Expected values printed by base_train.py's own functions with
    // num_iterations=1000, warmup 40, warmdown 0.65, final 0.05, weight_decay 0.2.
    const s = mod.TrainSchedule{ .num_iterations = 1000, .weight_decay = 0.2 };
    try std.testing.expectEqual(@as(usize, 650), s.warmdownSteps());
    const lr = [_]struct { usize, f64 }{ .{ 0, 0.025 }, .{ 39, 1.0 }, .{ 350, 1.0 }, .{ 351, 0.9985384615384616 }, .{ 999, 0.05146153846153847 } };
    for (lr) |case| try std.testing.expectApproxEqAbs(case[1], s.lrMultiplier(case[0]), 1e-12);
    const momentum = [_]struct { usize, f64 }{ .{ 0, 0.85 }, .{ 200, 0.91 }, .{ 349, 0.9547 }, .{ 400, 0.9646153846153847 }, .{ 650, 0.9376923076923076 }, .{ 999, 0.9001076923076924 } };
    for (momentum) |case| try std.testing.expectApproxEqAbs(case[1], s.muonMomentum(case[0]), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), s.weightDecay(500), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.934798141786878e-07), s.weightDecay(999), 1e-15);
}

test "train schedule rounds the warmdown like python" {
    // round(0.5 * 5) == 2 and round(0.5 * 7) == 4 in Python (ties to even).
    try std.testing.expectEqual(@as(usize, 2), (mod.TrainSchedule{ .num_iterations = 5, .warmdown_ratio = 0.5 }).warmdownSteps());
    try std.testing.expectEqual(@as(usize, 4), (mod.TrainSchedule{ .num_iterations = 7, .warmdown_ratio = 0.5 }).warmdownSteps());
}
