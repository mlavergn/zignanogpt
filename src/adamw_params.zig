const std = @import("std");
const log = std.log.scoped(.zignanogpt_adamw_params);
const mod = @import("module.zig");

/// One AdamW update's hyperparameters (`adamw_step_fused` in nanochat's
/// `optim.py`): decoupled weight decay, then bias-corrected Adam.
pub const AdamWParams = struct {
    const Self = @This();

    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// The update count for this parameter, starting at 1.
    step: u32,

    /// Rejects values the update cannot use.
    ///
    /// Parameters:
    /// - `self`: the parameters.
    ///
    /// Return: nothing; `error.InvalidOptimizerParams`.
    pub fn validate(self: Self) !void {
        if (self.step == 0 or self.beta1 < 0 or self.beta1 >= 1 or self.beta2 < 0 or self.beta2 >= 1) {
            log.debug("invalid AdamW params: step {d}, betas {d} {d}", .{ self.step, self.beta1, self.beta2 });
            return error.InvalidOptimizerParams;
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "adamw params require a positive step and betas in [0, 1)" {
    const ok = mod.AdamWParams{ .lr = 1, .beta1 = 0.8, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0, .step = 1 };
    try ok.validate();
    var bad = ok;
    bad.step = 0;
    try std.testing.expectError(error.InvalidOptimizerParams, bad.validate());
}
