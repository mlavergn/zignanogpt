const std = @import("std");
const log = std.log.scoped(.zignanogpt_rl_trainer);
const mod = @import("module.zig");

/// `chat_rl.py`'s options (0 for `top_k` samples from every token).
pub const RlOptions = struct {
    model_tag: ?[]const u8 = null,
    model_step: ?usize = null,
    num_epochs: usize = 1,
    device_batch_size: usize = 8,
    examples_per_step: usize = 16,
    num_samples: usize = 16,
    max_new_tokens: usize = 256,
    temperature: f64 = 1.0,
    top_k: usize = 50,
    embedding_lr: f64 = 0.2,
    unembedding_lr: f64 = 0.004,
    matrix_lr: f64 = 0.02,
    weight_decay: f64 = 0.0,
    init_lr_frac: f64 = 0.05,
    eval_every: usize = 60,
    eval_examples: usize = 400,
    save_every: usize = 60,
};

/// One example's rollouts: every sample's prompt + completion tokens, the
/// Engine's masks (1 = sampled, 0 = prompt or forced), and the rewards.
pub const Rollout = struct {
    sequences: []const []const u32,
    masks: []const []const u8,
    rewards: []const f32,
};

/// nanochat's RL on GSM8K (`chat_rl.py`): "GRPO" reduced to on-policy
/// REINFORCE. Each step samples `num_samples` completions for each of
/// `examples_per_step` training problems, rewards the correct ones, and
/// follows the token-level policy gradient weighted by `reward - mean`
/// (no KL term, no ratio clipping). Starts from the sft checkpoint; writes
/// `chatrl_checkpoints/<tag>`.
pub const RlTrainer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    backend: *mod.Backend,
    storage: mod.Storage,
    options: RlOptions,
    out: *std.Io.Writer,
    observer: ?mod.TrainObserver = null,

    model: *mod.LoadedModel,
    train: mod.Task,
    val: mod.Task,
    grads: mod.GptWeights,
    optimizer: mod.MuonAdamW,
    checkpoint: mod.Checkpoint,
    /// `<checkpoint dir>/metrics.jsonl`, for the web dashboard.
    metrics_path: []const u8,
    num_steps: usize,
    /// The next training example (cycling).
    cursor: usize = 0,

    /// Loads the sft model and GSM8K, builds the optimizer.
    ///
    /// Parameters:
    /// - `self`: the storage (keep it at this address).
    /// - `allocator`: owns everything.
    /// - `io`: file and network access.
    /// - `backend`: compute.
    /// - `config`: base directories (checkpoints, task data).
    /// - `options`: the RL options (strings outlive the trainer).
    /// - `out`: progress output.
    ///
    /// Return: nothing; loading and setup errors.
    pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io, backend: *mod.Backend, config: *const mod.Config, options: RlOptions, out: *std.Io.Writer) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (options.device_batch_size == 0 or options.num_samples % options.device_batch_size != 0) {
            log.warn("num_samples ({d}) must be a multiple of device_batch_size ({d})", .{ options.num_samples, options.device_batch_size });
            return error.InvalidBatchSize;
        }
        const storage = mod.Storage.init(allocator, io);
        const model = try allocator.create(mod.LoadedModel);
        errdefer allocator.destroy(model);
        try model.init(allocator, backend, storage, config.base_dir, .{ .kind = .sft, .tag = options.model_tag, .step = options.model_step });
        errdefer model.deinit();
        try out.print("Loaded sft model {s} step {d}\n", .{ model.tag, model.step });
        var train = try mod.Task.open(allocator, io, config, .gsm8k, "train", out);
        errdefer train.deinit();
        var val = try mod.Task.open(allocator, io, config, .gsm8k, "test", out);
        errdefer val.deinit();
        const num_steps = train.len() / @max(options.examples_per_step, 1) * options.num_epochs;
        var tag_buf: [32]u8 = undefined;
        const tag = options.model_tag orelse try std.mem.print(&tag_buf, "d{d}", .{model.model.config.n_layer});
        var grads = try mod.GptWeights.init(allocator, backend, model.model.config);
        errdefer grads.deinit();
        var optimizer = try mod.MuonAdamW.init(allocator, backend, &model.model.weights, model.model.config.n_embd, .{
            .unembedding_lr = options.unembedding_lr,
            .embedding_lr = options.embedding_lr,
            .matrix_lr = options.matrix_lr,
            .weight_decay = options.weight_decay,
        });
        errdefer optimizer.deinit();
        optimizer.scaleLearningRates(options.init_lr_frac);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .backend = backend,
            .storage = storage,
            .options = options,
            .out = out,
            .model = model,
            .train = train,
            .val = val,
            .grads = grads,
            .optimizer = optimizer,
            .checkpoint = try mod.Checkpoint.init(allocator, storage, config.base_dir, .rl, tag),
            .metrics_path = undefined,
            .num_steps = num_steps,
        };
        self.metrics_path = try std.Io.Dir.path.join(allocator, &.{ self.checkpoint.dir, "metrics.jsonl" });
        try out.print("Calculated number of steps: {d}\nTotal sequences per step: {d}\n", .{ num_steps, options.examples_per_step * options.num_samples });
        try out.flush();
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.metrics_path);
        self.checkpoint.deinit();
        self.optimizer.deinit();
        self.grads.deinit();
        self.val.deinit();
        self.train.deinit();
        self.model.deinit();
        self.allocator.destroy(self.model);
    }

    /// Trains for `num_steps`: pass@k evals, rollouts, policy-gradient steps,
    /// checkpoints. A stop request (the observer) is honored before each step,
    /// each example's rollouts and each eval problem: it saves the model as of
    /// the last completed step and returns.
    pub fn run(self: *Self) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.steps() catch |err| switch (err) {
            error.Stopped => return,
            else => return err,
        };
    }

    fn steps(self: *Self) !void {
        const o = self.options;
        const model = &self.model.model;
        var step: usize = 0;
        var total_time: f64 = 0;
        while (step < self.num_steps) : (step += 1) {
            try self.checkStop(step);
            if (o.eval_every > 0 and step % o.eval_every == 0) try self.evaluate(step);
            const started = std.Io.Clock.awake.now(self.io);

            try self.grads.zero();
            var reward_sum: f64 = 0;
            var length_sum: usize = 0;
            var sequences: usize = 0;
            for (0..o.examples_per_step) |example_step| {
                // A stop mid-step drops this step's gradients: the model stays at the last completed step.
                try self.checkStop(step);
                mod.TrainObserver.progress(self.observer, "rollouts", example_step, o.examples_per_step);
                var arena = std.heap.ArenaAllocator.init(self.allocator);
                defer arena.deinit();
                const rollout = try self.collect(arena.allocator(), step);
                const losses = try policyGradient(arena.allocator(), model, &self.grads, rollout, o.device_batch_size, o.examples_per_step, try self.model.tokenizer.special("<|assistant_end|>"));
                for (losses, 0..) |l, pass| {
                    var pass_reward: f64 = 0;
                    for (rollout.rewards[pass * o.device_batch_size ..][0..o.device_batch_size]) |r| pass_reward += r;
                    try self.out.print("Step {d}/{d} | Example step {d} | Pass {d} | loss: {d:.6} | Average reward: {d}\n", .{ step, self.num_steps, example_step, pass, l, pass_reward / @as(f64, @floatFromInt(o.device_batch_size)) });
                }
                var mean: f64 = 0;
                for (rollout.rewards) |r| mean += r;
                reward_sum += mean / @as(f64, @floatFromInt(rollout.rewards.len));
                for (rollout.sequences) |s| length_sum += s.len;
                sequences += rollout.sequences.len;
            }
            const mean_reward = reward_sum / @as(f64, @floatFromInt(o.examples_per_step));
            try self.out.print("Step {d}/{d} | Average reward: {d} | Average sequence length: {d:.2}\n", .{ step, self.num_steps, mean_reward, @as(f64, @floatFromInt(length_sum)) / @as(f64, @floatFromInt(@max(sequences, 1))) });
            try self.out.flush();

            const lrm = 1.0 - @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(self.num_steps));
            try self.metric(.{ .step = step, .reward = mean_reward, .lrm = lrm, .sequence_length = @as(f64, @floatFromInt(length_sum)) / @as(f64, @floatFromInt(@max(sequences, 1))) });
            // chat_rl.py only rescales the LRs: Muon keeps momentum 0.95 and the given weight decay.
            self.optimizer.setSchedule(lrm, 0.95, o.weight_decay);
            try self.optimizer.step(&model.weights, &self.grads);
            const dt = @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds)) / 1e9;
            total_time += dt;
            if (self.observer) |obs| obs.onRlStep(obs.context, .{
                .step = step,
                .num_steps = self.num_steps,
                .reward = mean_reward,
                .sequence_length = @as(f64, @floatFromInt(length_sum)) / @as(f64, @floatFromInt(@max(sequences, 1))),
                .lrm = lrm,
                .dt = dt,
                .total_time = total_time,
            });
            if ((step > 0 and o.save_every > 0 and step % o.save_every == 0) or step == self.num_steps - 1) try self.save(step);
        }
    }

    /// One training example's rollouts (`get_batch`): `num_samples` completions
    /// sampled `device_batch_size` at a time, each rewarded by GSM8K's checker.
    fn collect(self: *Self, arena: std.mem.Allocator, step: usize) !Rollout {
        const o = self.options;
        const tok = &self.model.tokenizer;
        const index = self.cursor;
        self.cursor = (self.cursor + 1) % self.train.len();
        const prompt = try tok.renderForCompletion(arena, try self.train.conversation(arena, index));
        const engine = mod.Engine.init(&self.model.model, tok);
        var sequences: std.ArrayList([]const u32) = .empty;
        var masks: std.ArrayList([]const u8) = .empty;
        var rewards: std.ArrayList(f32) = .empty;
        for (0..o.num_samples / o.device_batch_size) |sampling_step| {
            // Python seeds with hash((step, example_idx, sampling_step)); any distinct seed per draw will do.
            const seed = std.hash.Wyhash.hash(0, std.mem.asBytes(&[3]u64{ step, index, sampling_step })) & 0x7FFFFFFF;
            const batch = try engine.generateBatch(arena, prompt, .{
                .num_samples = o.device_batch_size,
                .max_tokens = o.max_new_tokens,
                .temperature = @floatCast(o.temperature),
                .top_k = if (o.top_k == 0) null else o.top_k,
                .seed = seed,
            });
            for (batch.results, batch.masks) |seq, mask| {
                try sequences.append(arena, seq);
                try masks.append(arena, mask);
                const text = try tok.decode(arena, seq[prompt.len..]);
                try rewards.append(arena, if (try self.train.evaluate(arena, index, text)) 1 else 0);
            }
        }
        return .{ .sequences = sequences.items, .masks = masks.items, .rewards = rewards.items };
    }

    /// Accumulates the policy gradient of one example's rollouts into `grads`
    /// (the loss chat_rl.py backpropagates): per pass of `device_batch_size`
    /// samples, `sum(nll * advantage) / (valid tokens * passes * examples)`,
    /// with advantage = reward - mean reward. Sequences are padded with
    /// `<|assistant_end|>`; only sampled tokens (mask 1) count.
    ///
    /// Parameters:
    /// - `arena`: scratch.
    /// - `model`: the policy.
    /// - `grads`: accumulated into.
    /// - `rollout`: the example's samples.
    /// - `device_batch_size`: samples per forward pass.
    /// - `examples_per_step`: examples whose gradients a step sums.
    /// - `pad`: the padding token (`<|assistant_end|>`).
    ///
    /// Return: each pass's loss; backend errors.
    pub fn policyGradient(arena: std.mem.Allocator, model: *mod.Gpt, grads: *mod.GptWeights, rollout: Rollout, device_batch_size: usize, examples_per_step: usize, pad: u32) ![]f32 {
        const n = rollout.sequences.len;
        const b = device_batch_size;
        const passes = n / b;
        var max_len: usize = 0;
        for (rollout.sequences) |s| max_len = @max(max_len, s.len);
        const t = max_len - 1;
        var mean: f32 = 0;
        for (rollout.rewards) |r| mean += r;
        mean /= @floatFromInt(n);

        const be = model.backend;
        var acts = try mod.GptActivations.init(arena, be, model.config, b, t);
        defer acts.deinit();
        var bufs = try mod.GptGradBuffers.init(arena, be, model.config, b, t);
        defer bufs.deinit();
        const idx = try be.alloc(.i32, &.{ b, t });
        defer be.free(idx);
        const target_t = try be.alloc(.i32, &.{ b, t });
        defer be.free(target_t);
        const weight_t = try be.alloc(.f32, &.{b * t});
        defer be.free(weight_t);
        const row_losses = try be.alloc(.f32, &.{b * t});
        defer be.free(row_losses);
        const inputs = try arena.alloc(i32, b * t);
        const targets = try arena.alloc(i32, b * t);
        const weights = try arena.alloc(f32, b * t);
        const losses = try arena.alloc(f32, b * t);
        const out = try arena.alloc(f32, passes);
        const logits = try acts.logits.reshape(&.{ b * t, model.config.vocab_size });

        for (0..passes) |p| {
            var valid: usize = 0;
            for (0..b) |r| {
                const seq = rollout.sequences[p * b + r];
                const mask = rollout.masks[p * b + r];
                for (0..t) |i| {
                    inputs[r * t + i] = @intCast(if (i < seq.len) seq[i] else pad);
                    const next: u32 = if (i + 1 < seq.len) seq[i + 1] else pad;
                    const m: u8 = if (i + 1 < mask.len) mask[i + 1] else 0;
                    targets[r * t + i] = if (m == 0) -1 else @intCast(next);
                    if (m != 0) valid += 1;
                }
            }
            const denom: f32 = @floatFromInt(@max(valid, 1) * passes * examples_per_step);
            for (0..b) |r| {
                const advantage = rollout.rewards[p * b + r] - mean;
                for (weights[r * t ..][0..t], targets[r * t ..][0..t]) |*w, tg| w.* = if (tg < 0) 0 else advantage / denom;
            }
            try be.upload(idx, i32, inputs);
            try be.upload(target_t, i32, targets);
            try be.upload(weight_t, f32, weights);
            try model.forward(&acts, idx);
            try be.crossEntropyRows(row_losses, logits, target_t);
            try be.download(row_losses, f32, losses);
            var loss: f64 = 0;
            for (losses, weights) |l, w| loss += @as(f64, l) * w;
            out[p] = @floatCast(loss);
            try model.backwardWeighted(&acts, &bufs, grads, idx, target_t, weight_t);
        }
        return out;
    }

    /// pass@k on the first `eval_examples` test problems (k = 1 .. device_batch_size).
    fn evaluate(self: *Self, step: usize) !void {
        const o = self.options;
        const tok = &self.model.tokenizer;
        const engine = mod.Engine.init(&self.model.model, tok);
        const k = o.device_batch_size;
        const passk = try self.allocator.alloc(usize, k);
        defer self.allocator.free(passk);
        @memset(passk, 0);
        const n = @min(o.eval_examples, self.val.len());
        for (0..n) |i| {
            try self.checkStop(step);
            mod.TrainObserver.progress(self.observer, "pass@k eval", i, n);
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const prompt = try tok.renderForCompletion(a, try self.val.conversation(a, i));
            const batch = try engine.generateBatch(a, prompt, .{ .num_samples = k, .max_tokens = 256, .temperature = 1.0, .top_k = 50, .seed = 42 + i });
            var any = false;
            for (batch.results, 0..) |seq, j| {
                if (!any and try self.val.evaluate(a, i, try tok.decode(a, seq[prompt.len..]))) any = true;
                if (any) passk[j] += 1;
            }
        }
        try self.out.print("Step {d} |", .{step});
        for (passk, 1..) |p, j| try self.out.print("{s} Pass@{d}: {d:.4}", .{ if (j == 1) "" else ",", j, @as(f64, @floatFromInt(p)) / @as(f64, @floatFromInt(@max(n, 1))) });
        try self.out.writeByte('\n');
        try self.out.flush();
        try self.metric(.{ .step = step, .pass_at_1 = @as(f64, @floatFromInt(passk[0])) / @as(f64, @floatFromInt(@max(n, 1))) });
        if (self.observer) |obs| obs.onEval(obs.context, step, @as(f64, @floatFromInt(passk[0])) / @as(f64, @floatFromInt(@max(n, 1))));
    }

    /// Appends one JSON line to `metrics.jsonl`.
    fn metric(self: *Self, value: anytype) !void {
        var line: std.Io.Writer.Allocating = .init(self.allocator);
        defer line.deinit();
        try std.json.Stringify.value(value, .{}, &line.writer);
        try line.writer.writeByte('\n');
        try self.storage.append(self.metrics_path, line.written());
    }

    /// On a stop request, saves the model at `step` (the steps completed so far)
    /// and returns `error.Stopped` (which `run` turns into a normal return).
    fn checkStop(self: *Self, step: usize) !void {
        if (!mod.TrainObserver.stopRequested(self.observer)) return;
        try self.out.print("Stopping at step {d} on request\n", .{step});
        try self.save(step);
        return error.Stopped;
    }

    /// Saves the model (no optimizer state, as chat_rl.py) under `chatrl_checkpoints/<tag>`.
    fn save(self: *Self, step: usize) !void {
        try self.checkpoint.saveModel(self.backend, step, &self.model.model.weights);
        const json = try std.json.Stringify.valueAlloc(self.allocator, .{ .step = step, .model_config = self.model.model.config, .user_config = self.options }, .{ .whitespace = .indent_2 });
        defer self.allocator.free(json);
        try self.checkpoint.saveMeta(step, json);
        try self.out.print("Saved model checkpoint to {s}\n", .{self.checkpoint.dir});
        try self.out.flush();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "rl policy gradient matches chat_rl.py's loss and gradients" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const meta_bytes = try storage.read(root ++ "/nanochat_base/base_checkpoints/d2/meta_000005.json");
    defer allocator.free(meta_bytes);
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, meta_bytes, .{});
    var model = try mod.Gpt.init(allocator, &backend, try mod.GptConfig.fromJson(a, meta.object.get("model_config").?));
    defer model.deinit();
    var state = try mod.TorchImport.loadStateDict(allocator, std.testing.io, root ++ "/nanochat_base/base_checkpoints/d2/model_000005.pt");
    defer state.deinit();
    try mod.TorchImport.loadWeights(&backend, &state, &model.weights);
    var grads = try mod.GptWeights.init(allocator, &backend, model.config);
    defer grads.deinit();
    try grads.zero();

    var expected = try mod.SafeTensors.load(allocator, std.testing.io, root ++ "/rl.safetensors");
    defer expected.deinit();
    const Raw = struct { sequences: []const []const u32, masks: []const []const u8, rewards: []const f32 };
    const rollouts = try std.json.parseFromSliceLeaky([]const Raw, a, expected.metadata("rollouts").?, .{});
    const want_losses = try std.json.parseFromSliceLeaky([]const f64, a, expected.metadata("losses").?, .{});
    const b = try std.fmt.parseInt(usize, expected.metadata("device_batch_size").?, 10);
    const examples = try std.fmt.parseInt(usize, expected.metadata("examples").?, 10);
    const pad = try tok.special("<|assistant_end|>");
    var losses: std.ArrayList(f32) = .empty;
    for (rollouts) |r| {
        const got = try RlTrainer.policyGradient(a, &model, &grads, .{ .sequences = r.sequences, .masks = r.masks, .rewards = r.rewards }, b, examples, pad);
        try losses.appendSlice(a, got);
    }
    try std.testing.expectEqual(want_losses.len, losses.items.len);
    for (want_losses, losses.items) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
    for (grads.params) |p| {
        var name_buf: [64]u8 = undefined;
        const want = try expected.readAlloc(allocator, try std.mem.print(&name_buf, "grad.{s}", .{p.name}), f32);
        defer allocator.free(want);
        const got = try allocator.alloc(f32, p.tensor.numel());
        defer allocator.free(got);
        try backend.download(p.tensor, f32, got);
        var peak: f32 = 0;
        for (want) |v| peak = @max(peak, @abs(v));
        // The absolute floor covers gradients that cancel to ~0 (backout_lambda
        // sums ~6M products to ~1e-10): their f32 rounding depends on the
        // backend's evaluation order (MPS, GPU kernels) at the 1e-8 level.
        for (want, got) |w, g| {
            if (@abs(w - g) > 2e-5 * peak + 1e-7) {
                std.debug.print("{s}: expected {d}, got {d}\n", .{ p.name, w, g });
                return error.ParityMismatch;
            }
        }
    }
}

test "rl stops on request mid-eval and mid-step, saving the last completed step" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();

    // Stops on the `stop_at`-th check.
    const Stopper = struct {
        calls: usize = 0,
        stop_at: usize,
        fn onStep(_: *anyopaque, _: mod.StepReport) void {}
        fn onEval(_: *anyopaque, _: usize, _: f64) void {}
        fn onSample(_: *anyopaque, _: usize, _: []const u8) void {}
        fn shouldStop(context: *anyopaque) bool {
            const s: *@This() = @ptrCast(@alignCast(context));
            s.calls += 1;
            return s.calls >= s.stop_at;
        }
        fn observer(s: *@This()) mod.TrainObserver {
            return .{ .context = s, .onStep = onStep, .onEval = onEval, .onSample = onSample, .shouldStop = shouldStop };
        }
    };
    // 1) the third check is the second eval problem; 2) the second example's rollouts.
    for ([_]usize{ 1, 0 }) |eval_every| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
        // The fixture base model stands in for the sft checkpoint RL starts from.
        const imported = try mod.TorchImport.importCheckpoint(allocator, &backend, storage, root ++ "/nanochat_base", base, .base, "d2", null);
        allocator.free(imported.tag);
        try tmp.dir.rename("base_checkpoints", tmp.dir, "chatsft_checkpoints", std.testing.io);
        const config = mod.Config{ .allocator = allocator, .base_dir = base, .nanochat_dir = root ++ "/task_base", .data_url = "http://unused" };

        var log_text: std.Io.Writer.Allocating = .init(allocator);
        defer log_text.deinit();
        var stopper = Stopper{ .stop_at = 3 };
        var trainer: RlTrainer = undefined;
        try trainer.init(allocator, std.testing.io, &backend, &config, .{
            .device_batch_size = 2,
            .examples_per_step = 4,
            .num_samples = 2,
            .max_new_tokens = 4,
            .eval_every = eval_every,
            .eval_examples = 4,
            .save_every = 0,
        }, &log_text.writer);
        defer trainer.deinit();
        trainer.observer = stopper.observer();
        try trainer.run();

        const text = log_text.written();
        try std.testing.expectEqual(@as(usize, 3), stopper.calls);
        try std.testing.expect(std.mem.find(u8, text, "Stopping at step 0 on request") != null);
        // Neither the eval nor the step completed.
        try std.testing.expect(std.mem.find(u8, text, "Pass@1") == null);
        try std.testing.expect(std.mem.find(u8, text, "Average sequence length") == null);
        if (eval_every == 0) try std.testing.expect(std.mem.find(u8, text, "Example step 0") != null);

        // The saved model is the starting one: the partial step's gradients were dropped.
        var rl: mod.LoadedModel = undefined;
        try rl.init(allocator, &backend, storage, base, .{ .kind = .rl });
        defer rl.deinit();
        try std.testing.expectEqual(@as(usize, 0), rl.step);
        var sft: mod.LoadedModel = undefined;
        try sft.init(allocator, &backend, storage, base, .{ .kind = .sft });
        defer sft.deinit();
        for ([_]mod.Tensor{ rl.model.weights.wte, rl.model.weights.lm_head }, [_]mod.Tensor{ sft.model.weights.wte, sft.model.weights.lm_head }) |got, want| {
            const g = try allocator.alloc(f32, got.numel());
            defer allocator.free(g);
            const w = try allocator.alloc(f32, want.numel());
            defer allocator.free(w);
            try backend.download(got, f32, g);
            try backend.download(want, f32, w);
            try std.testing.expectEqualSlices(f32, w, g);
        }
    }
}
