const std = @import("std");
const log = std.log.scoped(.zignanogpt_train_plan);
const mod = @import("module.zig");

/// `base_train.py`'s command-line options (its defaults).
pub const TrainOptions = struct {
    depth: usize = 20,
    aspect_ratio: usize = 64,
    head_dim: usize = 128,
    max_seq_len: usize = 2048,
    window_pattern: []const u8 = "SSSL",
    /// Explicit step count; 0 derives it from the FLOP or data:param target.
    num_iterations: usize = 0,
    /// FLOP budget; 0 disables.
    target_flops: f64 = 0,
    target_param_data_ratio: f64 = 12,
    device_batch_size: usize = 32,
    /// Tokens per optimizer step; 0 derives the compute-optimal size.
    total_batch_size: usize = 0,
    embedding_lr: f64 = 0.3,
    unembedding_lr: f64 = 0.008,
    weight_decay: f64 = 0.28,
    matrix_lr: f64 = 0.02,
    scalar_lr: f64 = 0.5,
    warmup_steps: usize = 40,
    warmdown_ratio: f64 = 0.65,
    final_lr_frac: f64 = 0.05,
    eval_every: usize = 250,
    eval_tokens: usize = 80 * 524288,
    sample_every: usize = 2000,
    /// 0: only at the end.
    save_every: usize = 0,

    /// `runs/runcpu.sh`'s settings, the `--preset=cpu` starting point.
    pub const cpu: TrainOptions = .{
        .depth = 6,
        .head_dim = 64,
        .window_pattern = "L",
        .max_seq_len = 512,
        .device_batch_size = 32,
        .total_batch_size = 16384,
        .eval_every = 100,
        .eval_tokens = 524288,
        .sample_every = 100,
        .num_iterations = 5000,
    };
};

/// Everything `base_train.py` derives before training: the model shape from
/// the depth, the compute-optimal horizon and batch size from scaling laws
/// (relative to a d12 reference, `B_REF = 2^19` tokens), the batch-size LR
/// and weight-decay corrections, and the step count.
pub const TrainPlan = struct {
    const Self = @This();

    /// Optimal batch size at d12, measured by nanochat.
    pub const b_ref: f64 = 524288;

    model: mod.GptConfig,
    scaling_params: usize,
    target_tokens: usize,
    d_ref: f64,
    total_batch_size: usize,
    lr_scale: f64,
    weight_decay: f64,
    num_iterations: usize,
    grad_accum_steps: usize,
    flops_per_token: usize,
    optimizer: mod.OptimizerConfig,
    schedule: mod.TrainSchedule,

    /// The model shape for a depth: width `depth * aspect_ratio` rounded up to a
    /// multiple of `head_dim`, one KV head per query head.
    ///
    /// Parameters:
    /// - `options`: the options.
    /// - `depth`: the layer count.
    /// - `vocab_size`: the tokenizer's vocabulary.
    ///
    /// Return: the config (window pattern borrowed from `options`).
    pub fn modelFor(options: TrainOptions, depth: usize, vocab_size: usize) mod.GptConfig {
        const base = depth * options.aspect_ratio;
        const dim = (base + options.head_dim - 1) / options.head_dim * options.head_dim;
        const heads = dim / options.head_dim;
        return .{
            .sequence_len = options.max_seq_len,
            .vocab_size = vocab_size,
            .n_layer = depth,
            .n_head = heads,
            .n_kv_head = heads,
            .n_embd = dim,
            .window_pattern = options.window_pattern,
        };
    }

    /// Derives the plan.
    ///
    /// Parameters:
    /// - `options`: the options.
    /// - `vocab_size`: the tokenizer's vocabulary.
    ///
    /// Return: the plan; `error.InvalidConfig` when the batch sizes do not divide.
    pub fn init(options: TrainOptions, vocab_size: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const model = modelFor(options, options.depth, vocab_size);
        try model.validate();
        const scaling_params = model.numScalingParams();
        const ratio = options.target_param_data_ratio;
        const target_tokens: usize = @intFromFloat(ratio * @as(f64, @floatFromInt(scaling_params)));
        const d_ref = ratio * @as(f64, @floatFromInt(modelFor(options, 12, vocab_size).numScalingParams()));

        const total = if (options.total_batch_size > 0) options.total_batch_size else blk: {
            const predicted = b_ref * std.math.pow(f64, @as(f64, @floatFromInt(target_tokens)) / d_ref, 0.383);
            break :blk @as(usize, 1) << @intFromFloat(roundHalfEven(std.math.log2(predicted)));
        };
        const batch_ratio = @as(f64, @floatFromInt(total)) / b_ref;
        const lr_scale = if (batch_ratio != 1) @sqrt(batch_ratio) else 1;
        const weight_decay = options.weight_decay * @sqrt(batch_ratio) * (d_ref / @as(f64, @floatFromInt(target_tokens)));

        const per_step = options.device_batch_size * options.max_seq_len;
        if (per_step == 0 or total % per_step != 0) {
            log.warn("total batch size {d} must be a multiple of device_batch_size * max_seq_len = {d}", .{ total, per_step });
            return error.InvalidConfig;
        }
        const flops = model.flopsPerToken();
        const iterations = if (options.num_iterations > 0)
            options.num_iterations
        else if (options.target_flops > 0)
            @as(usize, @intFromFloat(roundHalfEven(options.target_flops / @as(f64, @floatFromInt(flops * total)))))
        else
            target_tokens / total;
        if (iterations == 0) return error.InvalidConfig;

        return Self{
            .model = model,
            .scaling_params = scaling_params,
            .target_tokens = target_tokens,
            .d_ref = d_ref,
            .total_batch_size = total,
            .lr_scale = lr_scale,
            .weight_decay = weight_decay,
            .num_iterations = iterations,
            .grad_accum_steps = total / per_step,
            .flops_per_token = flops,
            .optimizer = .{
                .unembedding_lr = options.unembedding_lr * lr_scale,
                .embedding_lr = options.embedding_lr * lr_scale,
                .scalar_lr = options.scalar_lr * lr_scale,
                .matrix_lr = options.matrix_lr * lr_scale,
                .weight_decay = weight_decay,
            },
            .schedule = .{
                .num_iterations = iterations,
                .warmup_steps = options.warmup_steps,
                .warmdown_ratio = options.warmdown_ratio,
                .final_lr_frac = options.final_lr_frac,
                .weight_decay = weight_decay,
            },
        };
    }

    /// Python's `round`: ties to even.
    fn roundHalfEven(x: f64) f64 {
        const floor = @floor(x);
        const diff = x - floor;
        return if (diff > 0.5 or (diff == 0.5 and @mod(floor, 2) == 1)) floor + 1 else floor;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "train plan matches base_train.py for the cpu preset" {
    // Values printed by base_train.py's own computations (vocab 32768).
    const plan = try mod.TrainPlan.init(mod.TrainOptions.cpu, 32768);
    try std.testing.expectEqual(@as(usize, 384), plan.model.n_embd);
    try std.testing.expectEqual(@as(usize, 6), plan.model.n_head);
    try std.testing.expectEqual(@as(usize, 23199960), plan.scaling_params);
    try std.testing.expectEqual(@as(usize, 73531646), plan.model.numParams());
    try std.testing.expectEqual(@as(usize, 278399520), plan.target_tokens);
    try std.testing.expectApproxEqAbs(@as(f64, 1321216128), plan.d_ref, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1767766952966369), plan.lr_scale, 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2349029260055059), plan.weight_decay, 1e-15);
    try std.testing.expectEqual(@as(usize, 5000), plan.num_iterations);
    try std.testing.expectEqual(@as(usize, 1), plan.grad_accum_steps);
    try std.testing.expectEqual(@as(usize, 153355680), plan.flops_per_token);
}

test "train plan derives the compute-optimal batch for d20" {
    const plan = try mod.TrainPlan.init(.{}, 32768);
    try std.testing.expectEqual(@as(usize, 1280), plan.model.n_embd);
    try std.testing.expectEqual(@as(usize, 896533746), plan.model.numParams());
    try std.testing.expectEqual(@as(usize, 5221922880), plan.target_tokens);
    try std.testing.expectEqual(@as(usize, 1048576), plan.total_batch_size);
    try std.testing.expectApproxEqAbs(@as(f64, 1.4142135623730951), plan.lr_scale, 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1001877764255601), plan.weight_decay, 1e-15);
    try std.testing.expectEqual(@as(usize, 4980), plan.num_iterations);
    try std.testing.expectEqual(@as(usize, 16), plan.grad_accum_steps);
    try std.testing.expectEqual(@as(usize, 2886212784), plan.flops_per_token);
}
