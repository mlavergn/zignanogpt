const std = @import("std");
const log = std.log.scoped(.zignanogpt_trainer);
const mod = @import("module.zig");

/// Where the trainer reads documents: pretraining shards (train = all but the
/// last, val = the last) or in-memory documents (e.g. a local text file).
pub const TrainData = union(enum) {
    shards: []const []const u8,
    text: struct { train: []const []const u8, val: []const []const u8 },
};

/// One step's numbers, for logs, `metrics.jsonl` and the TUI.
pub const StepReport = struct {
    step: usize,
    num_iterations: usize,
    /// The debiased EMA of the training loss.
    loss: f64,
    lrm: f64,
    dt: f64,
    tok_per_sec: f64,
    tflops: f64,
    total_time: f64,
    state: mod.DataLoaderState,
};

/// One RL step, as `chat_rl.py` prints it: the mean reward over its rollouts.
pub const RlStepReport = struct {
    step: usize,
    num_steps: usize,
    reward: f64,
    sequence_length: f64,
    lrm: f64,
    /// Seconds this step took (rollouts, gradients, update).
    dt: f64,
    total_time: f64,
};

/// Called after every step, eval and sample (e.g. by the TUI), and during long
/// work inside any job (evaluations, rollouts, downloads, tokenizer training).
pub const TrainObserver = struct {
    context: *anyopaque,
    onStep: *const fn (context: *anyopaque, report: StepReport) void,
    onEval: *const fn (context: *anyopaque, step: usize, val_bpb: f64) void,
    onSample: *const fn (context: *anyopaque, step: usize, text: []const u8) void,
    /// Checked before each step (and, in SFT and RL, between eval problems and
    /// rollouts): true saves a checkpoint and ends the run.
    shouldStop: *const fn (context: *anyopaque) bool,
    /// `done` of `total` units of `label` (an eval's problems, a step's
    /// rollouts, shards downloaded); `label` is borrowed for the call only.
    onProgress: *const fn (context: *anyopaque, label: []const u8, done: usize, total: usize) void = &ignoreProgress,
    /// After each RL step.
    onRlStep: *const fn (context: *anyopaque, report: RlStepReport) void = &ignoreRlStep,

    /// Reports progress to `observer`, if any.
    ///
    /// Parameters:
    /// - `observer`: the hooks, or null (the command line).
    /// - `label`: what is being counted; borrowed for the call.
    /// - `done`: units finished.
    /// - `total`: units in all.
    ///
    /// Return: nothing.
    pub fn progress(observer: ?TrainObserver, label: []const u8, done: usize, total: usize) void {
        const obs = observer orelse return;
        obs.onProgress(obs.context, label, done, total);
    }

    fn ignoreProgress(_: *anyopaque, _: []const u8, _: usize, _: usize) void {}
    fn ignoreRlStep(_: *anyopaque, _: RlStepReport) void {}

    /// Whether `observer` (if any) asks to stop.
    pub fn stopRequested(observer: ?TrainObserver) bool {
        const obs = observer orelse return false;
        return obs.shouldStop(obs.context);
    }
};

/// Prompts sampled during training (`base_train.py`'s list).
const sample_prompts = [_][]const u8{
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
};

/// nanochat's base pretraining loop (`base_train.py`) on one process:
/// gradient accumulation, the LR/momentum/weight-decay schedules, val bpb,
/// greedy samples, checkpoints with optimizer and dataloader state, resume.
pub const Trainer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    backend: *mod.Backend,
    storage: mod.Storage,
    tokenizer: *const mod.Tokenizer,
    options: mod.TrainOptions,
    plan: mod.TrainPlan,
    data: TrainData,
    checkpoint: mod.Checkpoint,
    metrics_path: []const u8,
    observer: ?TrainObserver = null,
    out: *std.Io.Writer,

    model: mod.Gpt,
    grads: mod.GptWeights,
    acts: mod.GptActivations,
    bufs: mod.GptGradBuffers,
    optimizer: mod.MuonAdamW,
    token_bytes: []i32,

    stream: mod.DocumentStream,
    loader: mod.DataLoader,
    inputs: []i32,
    targets: []i32,
    idx: mod.Tensor,
    target_ids: mod.Tensor,
    loss: mod.Tensor,
    row_losses: mod.Tensor,

    step: usize = 0,
    resume_step: ?usize = null,
    smooth_loss: f64 = 0,
    total_time: f64 = 0,
    val_bpb: ?f64 = null,
    min_val_bpb: f64 = std.math.inf(f64),
    state: mod.DataLoaderState = .{},

    /// Builds the model, optimizer, loaders and checkpoint directory; on
    /// `resume_step`, restores model, optimizer, loop and dataloader state.
    ///
    /// Parameters:
    /// - `allocator`: owns everything.
    /// - `io`: storage and parquet reads.
    /// - `backend`: compute.
    /// - `tokenizer`: must outlive the trainer.
    /// - `options`: the training options.
    /// - `data`: documents (must outlive the trainer).
    /// - `base_dir`: checkpoints go in `<base_dir>/base_checkpoints/<tag>`.
    /// - `tag`: the model tag (nanochat: `d<depth>`).
    /// - `resume_step`: a saved step to continue from, or null.
    /// - `out`: progress output.
    ///
    /// Return: the trainer (keep it at a stable address); setup errors.
    pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io, backend: *mod.Backend, tokenizer: *const mod.Tokenizer, options: mod.TrainOptions, data: TrainData, base_dir: []const u8, tag: []const u8, resume_step: ?usize, out: *std.Io.Writer) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.* = .{
            .allocator = allocator,
            .io = io,
            .backend = backend,
            .storage = mod.Storage.init(allocator, io),
            .tokenizer = tokenizer,
            .options = options,
            .plan = try mod.TrainPlan.init(options, tokenizer.vocabSize()),
            .data = data,
            .checkpoint = undefined,
            .metrics_path = undefined,
            .out = out,
            .model = undefined,
            .grads = undefined,
            .acts = undefined,
            .bufs = undefined,
            .optimizer = undefined,
            .token_bytes = undefined,
            .stream = undefined,
            .loader = undefined,
            .inputs = undefined,
            .targets = undefined,
            .idx = undefined,
            .target_ids = undefined,
            .loss = undefined,
            .row_losses = undefined,
            .resume_step = resume_step,
        };
        const plan = self.plan;
        const b = options.device_batch_size;
        const t = options.max_seq_len;
        self.checkpoint = try mod.Checkpoint.init(allocator, self.storage, base_dir, .base, tag);
        errdefer self.checkpoint.deinit();
        self.metrics_path = try std.fs.path.join(allocator, &.{ self.checkpoint.dir, "metrics.jsonl" });
        errdefer allocator.free(self.metrics_path);

        self.model = try mod.Gpt.init(allocator, backend, plan.model);
        errdefer self.model.deinit();
        var rng = mod.Random.init(42);
        try self.model.initWeights(&rng);
        self.grads = try mod.GptWeights.init(allocator, backend, plan.model);
        errdefer self.grads.deinit();
        self.acts = try mod.GptActivations.init(allocator, backend, plan.model, b, t);
        errdefer self.acts.deinit();
        self.bufs = try mod.GptGradBuffers.init(allocator, backend, plan.model, b, t);
        errdefer self.bufs.deinit();
        self.optimizer = try mod.MuonAdamW.init(allocator, backend, &self.model.weights, plan.model.n_embd, plan.optimizer);
        errdefer self.optimizer.deinit();
        self.token_bytes = try tokenizer.tokenByteCounts(allocator);
        errdefer allocator.free(self.token_bytes);

        self.inputs = try allocator.alloc(i32, b * t);
        errdefer allocator.free(self.inputs);
        self.targets = try allocator.alloc(i32, b * t);
        errdefer allocator.free(self.targets);
        self.idx = try backend.alloc(.i32, &.{ b, t });
        errdefer backend.free(self.idx);
        self.target_ids = try backend.alloc(.i32, &.{ b, t });
        errdefer backend.free(self.target_ids);
        self.loss = try backend.alloc(.f32, &.{1});
        errdefer backend.free(self.loss);
        self.row_losses = try backend.alloc(.f32, &.{b * t});
        errdefer backend.free(self.row_losses);

        var resume_state: ?mod.DataLoaderState = null;
        if (resume_step) |s| resume_state = try self.restore(s);
        self.stream = try mod.DocumentStream.init(allocator, io, self.trainSource(), 128, resume_state);
        errdefer self.stream.deinit();
        self.loader = try mod.DataLoader.init(allocator, tokenizer, &self.stream, b, t, 1000);
    }

    /// Frees everything.
    ///
    /// Parameters:
    /// - `self`: the trainer.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.loader.deinit();
        self.stream.deinit();
        self.backend.free(self.row_losses);
        self.backend.free(self.loss);
        self.backend.free(self.target_ids);
        self.backend.free(self.idx);
        self.allocator.free(self.targets);
        self.allocator.free(self.inputs);
        self.allocator.free(self.token_bytes);
        self.optimizer.deinit();
        self.bufs.deinit();
        self.acts.deinit();
        self.grads.deinit();
        self.model.deinit();
        self.allocator.free(self.metrics_path);
        self.checkpoint.deinit();
    }

    /// Prints the plan, as `base_train.py` does before training.
    ///
    /// Parameters:
    /// - `self`: the trainer.
    ///
    /// Return: nothing; write errors.
    pub fn describe(self: *Self) !void {
        const p = self.plan;
        const o = self.options;
        try self.out.print("Model: depth {d}, n_embd {d}, heads {d}, seq {d}, window {s}, vocab {d}\n", .{ p.model.n_layer, p.model.n_embd, p.model.n_head, p.model.sequence_len, p.model.window_pattern, p.model.vocab_size });
        try self.out.print("Parameters: {d} total, {d} scaling\n", .{ p.model.numParams(), p.scaling_params });
        try self.out.print("Estimated FLOPs per token: {e}\n", .{@as(f64, @floatFromInt(p.flops_per_token))});
        try self.out.print("Total batch size {d} => gradient accumulation steps: {d} ({d} x {d} per micro-batch)\n", .{ p.total_batch_size, p.grad_accum_steps, o.device_batch_size, o.max_seq_len });
        try self.out.print("LR scale {d:.4}, Muon weight decay {d:.6}, iterations {d}, tokens {d}\n", .{ p.lr_scale, p.weight_decay, p.num_iterations, p.num_iterations * p.total_batch_size });
        try self.out.print("Checkpoints: {s}\n", .{self.checkpoint.dir});
        try self.out.flush();
    }

    /// Trains to `num_iterations`, evaluating, sampling and saving on schedule.
    ///
    /// Parameters:
    /// - `self`: the trainer.
    ///
    /// Return: nothing; backend, data and storage errors.
    pub fn run(self: *Self) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const o = self.options;
        const iterations = self.plan.num_iterations;
        self.state = try self.loader.next(self.inputs, self.targets); // first batch
        while (true) {
            if (self.observer) |obs| {
                if (obs.shouldStop(obs.context)) {
                    try self.out.print("Stopping at step {d} on request\n", .{self.step});
                    if (self.step > 0 and (self.resume_step == null or self.step != self.resume_step.?)) try self.save();
                    break;
                }
            }
            const last = self.step == iterations;
            if (o.eval_every > 0 and (last or self.step % o.eval_every == 0)) {
                const bpb = try self.evaluate();
                self.val_bpb = bpb;
                self.min_val_bpb = @min(self.min_val_bpb, bpb);
                try self.out.print("Step {d:0>5} | Validation bpb: {d:.6}\n", .{ self.step, bpb });
                try self.metric("{{\"step\":{d},\"val_bpb\":{d:.6}}}\n", .{ self.step, bpb });
                if (self.observer) |obs| obs.onEval(obs.context, self.step, bpb);
            }
            if (o.sample_every > 0 and (last or (self.step > 0 and self.step % o.sample_every == 0))) try self.sample();
            const resumed_here = self.resume_step != null and self.step == self.resume_step.?;
            if (last or (self.step > 0 and !resumed_here and o.save_every > 0 and self.step % o.save_every == 0)) try self.save();
            if (last) break;
            try self.trainStep();
            self.step += 1;
        }
        try self.out.print("Total training time: {d:.2}m\n", .{self.total_time / 60});
        if (self.val_bpb != null) try self.out.print("Minimum validation bpb: {d:.6}\n", .{self.min_val_bpb});
        try self.out.flush();
    }

    /// One optimizer step over `grad_accum_steps` micro-batches.
    fn trainStep(self: *Self) !void {
        const p = self.plan;
        const start = std.Io.Clock.awake.now(self.io);
        try self.grads.zero();
        const scale = 1 / @as(f32, @floatFromInt(p.grad_accum_steps));
        for (0..p.grad_accum_steps) |_| {
            try self.backend.upload(self.idx, i32, self.inputs);
            try self.backend.upload(self.target_ids, i32, self.targets);
            try self.model.forward(&self.acts, self.idx);
            try self.model.loss(&self.acts, self.target_ids, self.loss);
            try self.model.backward(&self.acts, &self.bufs, &self.grads, self.idx, self.target_ids, scale);
            self.state = try self.loader.next(self.inputs, self.targets); // prefetch, as base_train does
        }
        p.schedule.apply(&self.optimizer, self.step);
        try self.optimizer.step(&self.model.weights, &self.grads);
        var train_loss: [1]f32 = undefined;
        try self.backend.download(self.loss, f32, &train_loss); // sync point
        const dt = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds)) / 1e9;

        const ema = 0.9;
        self.smooth_loss = ema * self.smooth_loss + (1 - ema) * train_loss[0];
        const debiased = self.smooth_loss / (1 - std.math.pow(f64, ema, @floatFromInt(self.step + 1)));
        if (self.step > 10) self.total_time += dt;
        const tokens: f64 = @floatFromInt(p.total_batch_size);
        const report = StepReport{
            .step = self.step,
            .num_iterations = p.num_iterations,
            .loss = debiased,
            .lrm = p.schedule.lrMultiplier(self.step),
            .dt = dt,
            .tok_per_sec = tokens / dt,
            .tflops = @as(f64, @floatFromInt(p.flops_per_token)) * tokens / dt / 1e12,
            .total_time = self.total_time,
            .state = self.state,
        };
        const pct = 100 * @as(f64, @floatFromInt(self.step)) / @as(f64, @floatFromInt(p.num_iterations));
        try self.out.print("step {d:0>5}/{d:0>5} ({d:.2}%) | loss: {d:.6} | lrm: {d:.2} | dt: {d:.2}ms | tok/sec: {d:.0} | tflops: {d:.3} | epoch: {d} pq: {d} rg: {d} | total time: {d:.2}m", .{
            self.step, p.num_iterations, pct, debiased, report.lrm, dt * 1000, report.tok_per_sec, report.tflops, self.state.epoch, self.state.pq_idx, self.state.rg_idx, self.total_time / 60,
        });
        if (self.step > 10) {
            const avg = self.total_time / @as(f64, @floatFromInt(self.step - 10));
            try self.out.print(" | eta: {d:.1}m", .{avg * @as(f64, @floatFromInt(p.num_iterations - self.step)) / 60});
        }
        try self.out.writeByte('\n');
        try self.out.flush();
        try self.metric("{{\"step\":{d},\"loss\":{d:.6},\"lrm\":{d:.4},\"dt\":{d:.4},\"tok_per_sec\":{d:.0},\"tflops\":{d:.4},\"epoch\":{d}}}\n", .{ self.step, debiased, report.lrm, dt, report.tok_per_sec, report.tflops, self.state.epoch });
        if (self.observer) |obs| obs.onStep(obs.context, report);
    }

    /// Validation bits per byte over `eval_tokens` (nanochat's `evaluate_bpb`).
    ///
    /// Parameters:
    /// - `self`: the trainer.
    ///
    /// Return: the bpb (+inf without counted bytes); data and backend errors.
    pub fn evaluate(self: *Self) !f64 {
        const o = self.options;
        var stream = try mod.DocumentStream.init(self.allocator, self.io, self.valSource(), 128, null);
        defer stream.deinit();
        var loader = try mod.DataLoader.init(self.allocator, self.tokenizer, &stream, o.device_batch_size, o.max_seq_len, 1000);
        defer loader.deinit();
        const inputs = try self.allocator.alloc(i32, self.inputs.len);
        defer self.allocator.free(inputs);
        const targets = try self.allocator.alloc(i32, self.targets.len);
        defer self.allocator.free(targets);
        const losses = try self.allocator.alloc(f32, self.targets.len);
        defer self.allocator.free(losses);
        const steps = o.eval_tokens / (o.device_batch_size * o.max_seq_len);
        var nats: f64 = 0;
        var bytes: u64 = 0;
        const logits = try self.acts.logits.reshape(&.{ self.targets.len, self.plan.model.vocab_size });
        for (0..steps) |i| {
            mod.TrainObserver.progress(self.observer, "validation bpb", i, steps);
            _ = try loader.next(inputs, targets);
            try self.backend.upload(self.idx, i32, inputs);
            try self.backend.upload(self.target_ids, i32, targets);
            try self.model.forward(&self.acts, self.idx);
            try self.backend.crossEntropyRows(self.row_losses, logits, self.target_ids);
            try self.backend.download(self.row_losses, f32, losses);
            for (losses, targets) |l, t| {
                if (t < 0) continue;
                const n = self.token_bytes[@intCast(t)];
                if (n > 0) nats += l;
                bytes += @intCast(n);
            }
        }
        if (bytes == 0) return std.math.inf(f64);
        return nats / (@log(2.0) * @as(f64, @floatFromInt(bytes)));
    }

    /// Greedy completions of the sample prompts (16 tokens each).
    fn sample(self: *Self) !void {
        const bos = try self.tokenizer.bos();
        for (sample_prompts, 0..) |prompt, i| {
            mod.TrainObserver.progress(self.observer, "samples", i, sample_prompts.len);
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(self.allocator);
            try ids.append(self.allocator, bos);
            try self.tokenizer.encodeAppend(self.allocator, &ids, prompt);
            const generated = try self.model.greedy(self.allocator, ids.items, 16, bos);
            defer self.allocator.free(generated);
            try ids.appendSlice(self.allocator, generated);
            const text = try self.tokenizer.decode(self.allocator, ids.items[1..]);
            defer self.allocator.free(text);
            try self.out.print("{s}\n", .{text});
            const line = try std.json.Stringify.valueAlloc(self.allocator, .{ .step = self.step, .sample = text }, .{});
            defer self.allocator.free(line);
            try self.metric("{s}\n", .{line});
            if (self.observer) |obs| obs.onSample(obs.context, self.step, text);
        }
        try self.out.flush();
    }

    /// Saves model, optimizer and meta for the current step.
    fn save(self: *Self) !void {
        try self.checkpoint.saveModel(self.backend, self.step, &self.model.weights);
        try self.checkpoint.saveOptimizer(self.backend, self.step, &self.optimizer, &self.model.weights);
        const Meta = struct {
            step: usize,
            val_bpb: ?f64,
            model_config: mod.GptConfig,
            user_config: mod.TrainOptions,
            device_batch_size: usize,
            max_seq_len: usize,
            total_batch_size: usize,
            dataloader_state_dict: mod.DataLoaderState,
            loop_state: struct { min_val_bpb: ?f64, smooth_train_loss: f64, total_training_time: f64 },
        };
        const meta = Meta{
            .step = self.step,
            .val_bpb = self.val_bpb,
            .model_config = self.plan.model,
            .user_config = self.options,
            .device_batch_size = self.options.device_batch_size,
            .max_seq_len = self.options.max_seq_len,
            .total_batch_size = self.plan.total_batch_size,
            .dataloader_state_dict = self.state,
            .loop_state = .{
                .min_val_bpb = if (std.math.isInf(self.min_val_bpb)) null else self.min_val_bpb,
                .smooth_train_loss = self.smooth_loss,
                .total_training_time = self.total_time,
            },
        };
        const json = try std.json.Stringify.valueAlloc(self.allocator, meta, .{ .whitespace = .indent_2 });
        defer self.allocator.free(json);
        try self.checkpoint.saveMeta(self.step, json);
        try self.out.print("Saved checkpoint for step {d} to {s}\n", .{ self.step, self.checkpoint.dir });
        try self.out.flush();
    }

    /// Restores a saved step; returns the dataloader state to resume from.
    fn restore(self: *Self, step: usize) !mod.DataLoaderState {
        try self.checkpoint.loadModel(step, &self.model.weights);
        try self.checkpoint.loadOptimizer(step, &self.optimizer, &self.model.weights);
        var meta = try self.checkpoint.loadMeta(step);
        defer meta.deinit();
        const root = meta.value.object;
        const loop = root.get("loop_state").?.object;
        self.step = step;
        self.val_bpb = jsonFloat(root.get("val_bpb"));
        self.min_val_bpb = jsonFloat(loop.get("min_val_bpb")) orelse std.math.inf(f64);
        self.smooth_loss = jsonFloat(loop.get("smooth_train_loss")) orelse 0;
        self.total_time = jsonFloat(loop.get("total_training_time")) orelse 0;
        const dl = root.get("dataloader_state_dict").?.object;
        try self.out.print("Resuming from step {d}\n", .{step});
        return .{
            .pq_idx = @intCast(dl.get("pq_idx").?.integer),
            .rg_idx = @intCast(dl.get("rg_idx").?.integer),
            .epoch = @intCast(dl.get("epoch").?.integer),
        };
    }

    fn jsonFloat(value: ?std.json.Value) ?f64 {
        const v = value orelse return null;
        return switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => null,
        };
    }

    fn trainSource(self: *const Self) mod.DocumentSource {
        return switch (self.data) {
            .shards => |paths| .{ .parquet = paths[0 .. paths.len - 1] },
            .text => |t| .{ .text = t.train },
        };
    }

    fn valSource(self: *const Self) mod.DocumentSource {
        return switch (self.data) {
            .shards => |paths| .{ .parquet = paths[paths.len - 1 ..] },
            .text => |t| .{ .text = t.val },
        };
    }

    /// Appends a line to `metrics.jsonl`.
    fn metric(self: *Self, comptime fmt: []const u8, args: anytype) !void {
        const line = try std.fmt.allocPrint(self.allocator, fmt, args);
        defer self.allocator.free(line);
        try self.storage.append(self.metrics_path, line);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "trainer trains, evaluates, saves and resumes a tiny model" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];

    // A byte-level tokenizer: 256 single-byte tokens, no merges.
    var byte_storage: [256][1]u8 = undefined;
    var byte_tokens: [256][]const u8 = undefined;
    for (&byte_tokens, &byte_storage, 0..) |*t, *b, i| {
        b[0] = @intCast(i);
        t.* = b;
    }
    var tok = try mod.Tokenizer.init(allocator, &byte_tokens, &mod.Tokenizer.special_tokens, mod.Tokenizer.nanochat_max_digits);
    defer tok.deinit();
    const docs = [_][]const u8{ "the cat sat on the mat", "a dog ran in the park", "the sun is hot", "snow is cold and white" };
    const options = mod.TrainOptions{
        .depth = 2,
        .aspect_ratio = 16,
        .head_dim = 16,
        .max_seq_len = 32,
        .window_pattern = "L",
        .num_iterations = 4,
        .device_batch_size = 2,
        .total_batch_size = 128,
        .warmup_steps = 1,
        .eval_every = 2,
        .eval_tokens = 128,
        .sample_every = 4,
        .save_every = 2,
    };
    var discard: std.Io.Writer.Discarding = .init(&.{});
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();

    var first: mod.Trainer = undefined;
    try first.init(allocator, std.testing.io, &backend, &tok, options, .{ .text = .{ .train = &docs, .val = docs[2..] } }, root, "d2", null, &discard.writer);
    defer first.deinit();
    try std.testing.expectEqual(@as(usize, 2), first.plan.grad_accum_steps);
    try first.run();
    try std.testing.expectEqual(@as(usize, 4), first.step);
    try std.testing.expectEqual(@as(usize, 4), try first.checkpoint.lastStep("safetensors"));

    // Resume from the step-2 checkpoint (nanochat resumes data approximately:
    // from the row group after the saved one, so the runs need not match).
    var second: mod.Trainer = undefined;
    try second.init(allocator, std.testing.io, &backend, &tok, options, .{ .text = .{ .train = &docs, .val = docs[2..] } }, root, "d2", 2, &discard.writer);
    defer second.deinit();
    try std.testing.expectEqual(@as(usize, 2), second.step);
    try std.testing.expect(second.val_bpb != null);
    try second.run();
    try std.testing.expectEqual(@as(usize, 4), second.step);
    try std.testing.expect(std.math.isFinite(second.val_bpb.?));
    const metrics = try second.storage.read(second.metrics_path);
    defer allocator.free(metrics);
    try std.testing.expect(std.mem.count(u8, metrics, "\"loss\"") == 6); // 4 + 2 resumed steps
}
