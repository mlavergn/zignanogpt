const std = @import("std");
const log = std.log.scoped(.zignanogpt_sft_trainer);
const mod = @import("module.zig");

/// `chat_sft.py`'s options. Optional fields left null inherit the base
/// checkpoint's values (its meta, then nanochat's fallbacks); 0 for a count
/// means Python's -1 (disabled / unlimited / full epoch).
pub const SftOptions = struct {
    model_tag: ?[]const u8 = null,
    model_step: ?usize = null,
    /// Warm-start AdamW/Muon state from the base checkpoint (1) or not (0).
    load_optimizer: usize = 1,
    num_iterations: usize = 0,
    max_seq_len: ?usize = null,
    device_batch_size: ?usize = null,
    total_batch_size: ?usize = null,
    embedding_lr: ?f64 = null,
    unembedding_lr: ?f64 = null,
    matrix_lr: ?f64 = null,
    init_lr_frac: f64 = 0.8,
    warmup_ratio: f64 = 0.0,
    warmdown_ratio: f64 = 0.5,
    final_lr_frac: f64 = 0.0,
    eval_every: usize = 200,
    eval_tokens: usize = 40 * 524288,
    chatcore_every: usize = 200,
    /// Problems per categorical ChatCORE task (0: all).
    chatcore_max_cat: usize = 0,
    /// Problems per generative ChatCORE task (0: all).
    chatcore_max_sample: usize = 24,
    mmlu_epochs: usize = 3,
    gsm8k_epochs: usize = 4,
};

/// nanochat's supervised fine-tuning (`chat_sft.py`) on one process: a base
/// checkpoint, the task mixture packed by `SftLoader`, MuonAdamW at the base
/// run's learning rates times `init_lr_frac`, a progress-based LR schedule,
/// val bpb and ChatCORE on schedule, and a checkpoint under
/// `chatsft_checkpoints/<tag>` at the end.
pub const SftTrainer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    backend: *mod.Backend,
    storage: mod.Storage,
    options: SftOptions,
    data: *mod.SftData,
    out: *std.Io.Writer,
    observer: ?mod.TrainObserver = null,

    base: *mod.LoadedModel,
    config: mod.GptConfig,
    batch: usize,
    seq: usize,
    total_batch_size: usize,
    grad_accum_steps: usize,
    checkpoint: mod.Checkpoint,
    lrs: mod.OptimizerConfig,

    grads: mod.GptWeights,
    acts: mod.GptActivations,
    bufs: mod.GptGradBuffers,
    optimizer: mod.MuonAdamW,
    token_bytes: []i32,
    loader: mod.SftLoader,
    inputs: []i32,
    targets: []i32,
    idx: mod.Tensor,
    target_ids: mod.Tensor,
    loss: mod.Tensor,
    row_losses: mod.Tensor,

    step: usize = 0,
    progress: f64 = 0,
    smooth_loss: f64 = 0,
    total_time: f64 = 0,
    val_bpb: ?f64 = null,
    min_val_bpb: f64 = std.math.inf(f64),
    /// The last ChatCORE results (for tests and reports).
    last_chatcore: ?f64 = null,

    /// Loads the base model, resolves the inherited settings, builds the optimizer and loader.
    ///
    /// Parameters:
    /// - `self`: the storage (keep it at this address).
    /// - `allocator`: owns everything.
    /// - `io`: file access.
    /// - `backend`: compute.
    /// - `base_dir`: this port's base directory (base checkpoints in, SFT checkpoints out).
    /// - `options`: the SFT options (strings outlive the trainer).
    /// - `data`: the mixtures (outlives the trainer).
    /// - `out`: progress output.
    ///
    /// Return: nothing; loading and setup errors.
    pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io, backend: *mod.Backend, base_dir: []const u8, options: SftOptions, data: *mod.SftData, out: *std.Io.Writer) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const storage = mod.Storage.init(allocator, io);
        const base = try allocator.create(mod.LoadedModel);
        errdefer allocator.destroy(base);
        try base.init(allocator, backend, storage, base_dir, .{ .kind = .base, .tag = options.model_tag, .step = options.model_step });
        errdefer base.deinit();
        try out.print("Loaded base model {s} step {d}\n", .{ base.tag, base.step });

        // Inherit the batch shape and learning rates from the base run.
        var base_ckpt = try mod.Checkpoint.init(allocator, storage, base_dir, .base, base.tag);
        defer base_ckpt.deinit();
        var meta = try base_ckpt.loadMeta(base.step);
        defer meta.deinit();
        const root = meta.value.object;
        const user = if (root.get("user_config")) |u| (if (u == .object) u.object else null) else null;
        const seq = try inheritInt(out, "max_seq_len", options.max_seq_len, root.get("max_seq_len"), 2048);
        const batch = try inheritInt(out, "device_batch_size", options.device_batch_size, root.get("device_batch_size"), 32);
        const total = try inheritInt(out, "total_batch_size", options.total_batch_size, root.get("total_batch_size"), 524288);
        const lrs = mod.OptimizerConfig{
            .embedding_lr = try inheritFloat(out, "embedding_lr", options.embedding_lr, if (user) |u| u.get("embedding_lr") else null, 0.3),
            .unembedding_lr = try inheritFloat(out, "unembedding_lr", options.unembedding_lr, if (user) |u| u.get("unembedding_lr") else null, 0.004),
            .matrix_lr = try inheritFloat(out, "matrix_lr", options.matrix_lr, if (user) |u| u.get("matrix_lr") else null, 0.02),
            .weight_decay = 0,
        };
        const config = base.model.config;
        if (total % (batch * seq) != 0) {
            log.warn("total_batch_size {d} must be a multiple of {d} x {d}", .{ total, batch, seq });
            return error.InvalidBatchSize;
        }
        var tag_buf: [32]u8 = undefined;
        const tag = options.model_tag orelse try std.fmt.bufPrint(&tag_buf, "d{d}", .{config.n_layer});

        self.* = .{
            .allocator = allocator,
            .io = io,
            .backend = backend,
            .storage = storage,
            .options = options,
            .data = data,
            .out = out,
            .base = base,
            .config = config,
            .batch = batch,
            .seq = seq,
            .total_batch_size = total,
            .grad_accum_steps = total / (batch * seq),
            .checkpoint = try mod.Checkpoint.init(allocator, storage, base_dir, .sft, tag),
            .lrs = lrs,
            .grads = undefined,
            .acts = undefined,
            .bufs = undefined,
            .optimizer = undefined,
            .token_bytes = undefined,
            .loader = undefined,
            .inputs = undefined,
            .targets = undefined,
            .idx = undefined,
            .target_ids = undefined,
            .loss = undefined,
            .row_losses = undefined,
        };
        errdefer self.checkpoint.deinit();
        self.grads = try mod.GptWeights.init(allocator, backend, config);
        errdefer self.grads.deinit();
        self.acts = try mod.GptActivations.init(allocator, backend, config, batch, seq);
        errdefer self.acts.deinit();
        self.bufs = try mod.GptGradBuffers.init(allocator, backend, config, batch, seq);
        errdefer self.bufs.deinit();
        self.optimizer = try mod.MuonAdamW.init(allocator, backend, &base.model.weights, config.n_embd, lrs);
        errdefer self.optimizer.deinit();
        if (options.load_optimizer != 0) {
            const optim_path = try base_ckpt.path("optim", base.step, "safetensors");
            defer allocator.free(optim_path);
            if (try storage.exists(optim_path)) {
                try base_ckpt.loadOptimizer(base.step, &self.optimizer, &base.model.weights);
                try out.writeAll("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)\n");
            } else try out.writeAll("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)\n");
        }
        self.optimizer.scaleLearningRates(options.init_lr_frac);
        self.token_bytes = try base.tokenizer.tokenByteCounts(allocator);
        errdefer allocator.free(self.token_bytes);
        self.inputs = try allocator.alloc(i32, batch * seq);
        errdefer allocator.free(self.inputs);
        self.targets = try allocator.alloc(i32, batch * seq);
        errdefer allocator.free(self.targets);
        self.idx = try backend.alloc(.i32, &.{ batch, seq });
        errdefer backend.free(self.idx);
        self.target_ids = try backend.alloc(.i32, &.{ batch, seq });
        errdefer backend.free(self.target_ids);
        self.loss = try backend.alloc(.f32, &.{1});
        errdefer backend.free(self.loss);
        self.row_losses = try backend.alloc(.f32, &.{batch * seq});
        errdefer backend.free(self.row_losses);
        self.loader = try mod.SftLoader.init(allocator, &base.tokenizer, &data.train, batch, seq, options.num_iterations, true);
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.loader.deinit();
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
        self.checkpoint.deinit();
        self.base.deinit();
        self.allocator.destroy(self.base);
    }

    /// Prints the setup, as `chat_sft.py` does before training.
    pub fn describe(self: *Self) !void {
        try self.out.print("Tokens / micro-batch: {d} x {d} = {d}\n", .{ self.batch, self.seq, self.batch * self.seq });
        try self.out.print("Total batch size {d} => gradient accumulation steps: {d}\n", .{ self.total_batch_size, self.grad_accum_steps });
        try self.out.print("Training mixture: {d} rows (MMLU x{d}, GSM8K x{d}); validation {d} rows\n", .{ self.data.train.len(), self.options.mmlu_epochs, self.options.gsm8k_epochs, self.data.val.len() });
        try self.out.print("Checkpoint: {s}\n", .{self.checkpoint.dir});
        try self.out.flush();
    }

    /// Fine-tunes until the data (or `num_iterations`) runs out.
    ///
    /// Parameters:
    /// - `self`: the trainer.
    ///
    /// Return: nothing; backend, data and storage errors.
    pub fn run(self: *Self) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const o = self.options;
        try self.loader.next(self.inputs, self.targets); // prefetch the first batch
        while (true) {
            if (self.observer) |obs| {
                if (obs.shouldStop(obs.context)) {
                    try self.out.print("Stopping at step {d} on request\n", .{self.step});
                    try self.save();
                    break;
                }
            }
            const last = self.loader.last_step;
            if (last or (o.eval_every > 0 and self.step % o.eval_every == 0)) {
                const bpb = try self.evaluate();
                self.val_bpb = bpb;
                self.min_val_bpb = @min(self.min_val_bpb, bpb);
                try self.out.print("Step {d:0>5} | Validation bpb: {d:.4}\n", .{ self.step, bpb });
                try self.out.flush();
                if (self.observer) |obs| obs.onEval(obs.context, self.step, bpb);
            }
            if (o.chatcore_every > 0 and (last or (self.step > 0 and self.step % o.chatcore_every == 0))) try self.chatcore();
            if (last) {
                try self.save();
                break;
            }
            try self.trainStep();
        }
        try self.out.print("Total training time: {d:.2}m\n", .{self.total_time / 60});
        try self.out.print("Minimum validation bpb: {d:.4}\n", .{self.min_val_bpb});
        try self.out.flush();
    }

    /// `get_lr_multiplier(progress)`: warmup, constant, linear warmdown to `final_lr_frac`.
    pub fn lrMultiplier(self: *const Self, progress: f64) f64 {
        const o = self.options;
        if (progress < o.warmup_ratio) return (progress + 1e-8) / o.warmup_ratio;
        if (progress <= 1.0 - o.warmdown_ratio) return 1.0;
        const decay = (progress - (1.0 - o.warmdown_ratio)) / o.warmdown_ratio;
        return (1 - decay) * 1.0 + decay * o.final_lr_frac;
    }

    /// `get_muon_momentum(it)`: 0.85 -> 0.95 over 300 steps.
    pub fn muonMomentum(it: usize) f64 {
        const frac = @min(@as(f64, @floatFromInt(it)) / 300, 1);
        return (1 - frac) * 0.85 + frac * 0.95;
    }

    fn trainStep(self: *Self) !void {
        const start = std.Io.Clock.awake.now(self.io);
        try self.grads.zero();
        const scale = 1 / @as(f32, @floatFromInt(self.grad_accum_steps));
        const model = &self.base.model;
        for (0..self.grad_accum_steps) |_| {
            try self.backend.upload(self.idx, i32, self.inputs);
            try self.backend.upload(self.target_ids, i32, self.targets);
            try model.forward(&self.acts, self.idx);
            try model.loss(&self.acts, self.target_ids, self.loss);
            try model.backward(&self.acts, &self.bufs, &self.grads, self.idx, self.target_ids, scale);
            try self.loader.next(self.inputs, self.targets); // prefetch, as chat_sft does
            self.progress = @max(self.progress, self.loader.progress);
        }
        const lrm = self.lrMultiplier(self.progress);
        // Muon weight decay stays 0: SFT passes weight_decay=0.
        self.optimizer.setSchedule(lrm, muonMomentum(self.step), 0);
        try self.optimizer.step(&model.weights, &self.grads);
        var train_loss: [1]f32 = undefined;
        try self.backend.download(self.loss, f32, &train_loss);
        const dt = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds)) / 1e9;
        self.step += 1;

        const ema = 0.9;
        self.smooth_loss = ema * self.smooth_loss + (1 - ema) * train_loss[0];
        const debiased = self.smooth_loss / (1 - std.math.pow(f64, ema, @floatFromInt(self.step + 1)));
        if (self.step > 10) self.total_time += dt;
        const tokens: f64 = @floatFromInt(self.total_batch_size);
        const flops: f64 = @floatFromInt(self.config.flopsPerToken());
        try self.out.print("step {d:0>5} ({d:.2}%) | loss: {d:.6} | lrm: {d:.2} | dt: {d:.2}ms | tok/sec: {d:.0} | tflops: {d:.3} | epoch: {d} | total time: {d:.2}m\n", .{
            self.step, 100 * self.progress, debiased, lrm, dt * 1000, tokens / dt, flops * tokens / dt / 1e12, self.loader.current_epoch, self.total_time / 60,
        });
        try self.out.flush();
        if (self.observer) |obs| {
            // Without num_iterations the length is known only through the progress.
            const iterations = if (self.options.num_iterations > 0)
                self.options.num_iterations
            else if (self.progress > 0)
                @max(self.step, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(self.step)) / self.progress))))
            else
                self.step;
            obs.onStep(obs.context, .{
                .step = self.step - 1,
                .num_iterations = iterations,
                .loss = debiased,
                .lrm = lrm,
                .dt = dt,
                .tok_per_sec = tokens / dt,
                .tflops = flops * tokens / dt / 1e12,
                .total_time = self.total_time,
                .state = .{ .epoch = self.loader.current_epoch },
            });
        }
    }

    /// Val bpb over `eval_tokens` of the validation mixture (a fresh loader each time).
    pub fn evaluate(self: *Self) !f64 {
        var loader = try mod.SftLoader.init(self.allocator, &self.base.tokenizer, &self.data.val, self.batch, self.seq, 0, false);
        defer loader.deinit();
        const inputs = try self.allocator.alloc(i32, self.inputs.len);
        defer self.allocator.free(inputs);
        const targets = try self.allocator.alloc(i32, self.targets.len);
        defer self.allocator.free(targets);
        const losses = try self.allocator.alloc(f32, self.targets.len);
        defer self.allocator.free(losses);
        const steps = self.options.eval_tokens / (self.batch * self.seq);
        const logits = try self.acts.logits.reshape(&.{ self.targets.len, self.config.vocab_size });
        var nats: f64 = 0;
        var bytes: u64 = 0;
        for (0..steps) |_| {
            try loader.next(inputs, targets);
            try self.backend.upload(self.idx, i32, inputs);
            try self.backend.upload(self.target_ids, i32, targets);
            try self.base.model.forward(&self.acts, self.idx);
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

    /// ChatCORE over the chat-eval tasks (HumanEval excluded), plus the categorical-only mean.
    fn chatcore(self: *Self) !void {
        const tasks = try self.data.chatcoreTasks(self.out);
        var eval = mod.ChatEval.init(self.allocator, &self.base.model, &self.base.tokenizer);
        var results: std.ArrayList(mod.EvalResult) = .empty;
        defer results.deinit(self.allocator);
        var categorical: std.ArrayList(mod.EvalResult) = .empty;
        defer categorical.deinit(self.allocator);
        for (tasks) |task| {
            const is_cat = task.evalType() == .categorical;
            const limit = if (is_cat) self.options.chatcore_max_cat else self.options.chatcore_max_sample;
            const r = try eval.run(task, .{}, if (limit == 0) null else limit);
            try self.out.print("  {s}: {d:.2}%\n", .{ task.kind.name(), 100 * r.accuracy() });
            try results.append(self.allocator, r);
            if (is_cat) try categorical.append(self.allocator, r);
        }
        const core = mod.ChatEval.chatCore(results.items);
        self.last_chatcore = core;
        try self.out.print("Step {d:0>5} | ChatCORE: {d:.4} | ChatCORE_cat: {d:.4}\n", .{ self.step, core, mod.ChatEval.chatCore(categorical.items) });
        try self.out.flush();
    }

    /// Saves model, optimizer and meta under `chatsft_checkpoints/<tag>`.
    fn save(self: *Self) !void {
        const model = &self.base.model;
        try self.checkpoint.saveModel(self.backend, self.step, &model.weights);
        try self.checkpoint.saveOptimizer(self.backend, self.step, &self.optimizer, &model.weights);
        var model_config = self.config;
        model_config.sequence_len = self.seq;
        const Meta = struct { step: usize, val_bpb: ?f64, model_config: mod.GptConfig, user_config: SftOptions };
        const json = try std.json.Stringify.valueAlloc(self.allocator, Meta{ .step = self.step, .val_bpb = self.val_bpb, .model_config = model_config, .user_config = self.options }, .{ .whitespace = .indent_2 });
        defer self.allocator.free(json);
        try self.checkpoint.saveMeta(self.step, json);
        try self.out.print("Saved checkpoint for step {d} to {s}\n", .{ self.step, self.checkpoint.dir });
        try self.out.flush();
    }

    fn inheritInt(out: *std.Io.Writer, name: []const u8, arg: ?usize, base: ?std.json.Value, fallback: usize) !usize {
        const from_base: ?usize = if (base) |v| switch (v) {
            .integer => |i| std.math.cast(usize, i),
            else => null,
        } else null;
        return inherit(usize, out, name, arg, from_base, fallback);
    }

    fn inheritFloat(out: *std.Io.Writer, name: []const u8, arg: ?f64, base: ?std.json.Value, fallback: f64) !f64 {
        const from_base: ?f64 = if (base) |v| switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => null,
        } else null;
        return inherit(f64, out, name, arg, from_base, fallback);
    }

    fn inherit(comptime T: type, out: *std.Io.Writer, name: []const u8, arg: ?T, base: ?T, fallback: T) !T {
        if (arg) |v| {
            if (base != null and base.? != v) {
                try out.print("NOTE: --{s}={any} overrides pretrained value of {any}\n", .{ name, v, base.? });
            } else try out.print("Using {s}={any}\n", .{ name, v });
            return v;
        }
        const resolved = base orelse fallback;
        try out.print("Inherited {s}={any} from pretrained checkpoint\n", .{ name, resolved });
        return resolved;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "sft trainer reproduces chat_sft.py's losses, val bpb and weights" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    const imported = try mod.TorchImport.importCheckpoint(allocator, &backend, storage, root ++ "/nanochat_base", base, .base, "d2", null);
    allocator.free(imported.tag);

    var expected = try mod.SafeTensors.load(allocator, std.testing.io, root ++ "/sft.safetensors");
    defer expected.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const want_losses = try std.json.parseFromSliceLeaky([]const f64, a, expected.metadata("losses").?, .{});
    const want_bpbs = try std.json.parseFromSliceLeaky([]const [2]f64, a, expected.metadata("bpbs").?, .{});

    // The tasks come from the fixture slices (nanochat's dir is the fallback location).
    const config = mod.Config{ .allocator = allocator, .base_dir = base, .nanochat_dir = root ++ "/task_base", .data_url = "http://unused" };
    const data = try mod.SftData.openStandard(allocator, std.testing.io, &config, 1, 2, null);
    defer data.destroy();
    var log_text: std.Io.Writer.Allocating = .init(allocator);
    defer log_text.deinit();
    var trainer: SftTrainer = undefined;
    try trainer.init(allocator, std.testing.io, &backend, base, .{
        .num_iterations = 4,
        .max_seq_len = 256,
        .device_batch_size = 2,
        .total_batch_size = 512,
        .eval_every = 2,
        .eval_tokens = 512,
        .chatcore_every = 0,
        .mmlu_epochs = 1,
        .gsm8k_epochs = 2,
    }, data, &log_text.writer);
    defer trainer.deinit();
    try trainer.describe();
    try trainer.run();

    var losses: std.ArrayList(f64) = .empty;
    var bpbs: std.ArrayList([2]f64) = .empty;
    var lines = std.mem.splitScalar(u8, log_text.written(), '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "step ")) {
            const at = std.mem.indexOf(u8, line, "loss: ").? + 6;
            const end = std.mem.indexOfScalarPos(u8, line, at, ' ').?;
            try losses.append(a, try std.fmt.parseFloat(f64, line[at..end]));
        } else if (std.mem.startsWith(u8, line, "Step ") and std.mem.indexOf(u8, line, "Validation bpb: ") != null) {
            const step = try std.fmt.parseFloat(f64, line[5..10]);
            const at = std.mem.indexOf(u8, line, "bpb: ").? + 5;
            try bpbs.append(a, .{ step, try std.fmt.parseFloat(f64, line[at..]) });
        }
    }
    try std.testing.expectEqual(want_losses.len, losses.items.len);
    for (want_losses, losses.items) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-4);
    try std.testing.expectEqual(want_bpbs.len, bpbs.items.len);
    for (want_bpbs, bpbs.items) |w, g| {
        try std.testing.expectEqual(w[0], g[0]);
        try std.testing.expectApproxEqAbs(w[1], g[1], 2e-4);
    }

    // The saved SFT checkpoint gives PyTorch's logits.
    var sft: mod.LoadedModel = undefined;
    try sft.init(allocator, &backend, storage, base, .{ .kind = .sft });
    defer sft.deinit();
    try std.testing.expectEqual(try std.fmt.parseInt(usize, expected.metadata("step").?, 10), sft.step);
    var acts = try mod.GptActivations.init(allocator, &backend, sft.model.config, 1, 24);
    defer acts.deinit();
    const ids = try expected.readAlloc(allocator, "idx", i32);
    defer allocator.free(ids);
    const idx = try backend.alloc(.i32, &.{ 1, 24 });
    defer backend.free(idx);
    try backend.upload(idx, i32, ids);
    try sft.model.forward(&acts, idx);
    const got = try allocator.alloc(f32, acts.logits.numel());
    defer allocator.free(got);
    try backend.download(acts.logits, f32, got);
    const want = try expected.readAlloc(allocator, "logits", f32);
    defer allocator.free(want);
    var worst: f32 = 0;
    for (want, got) |w, g| worst = @max(worst, @abs(w - g));
    if (worst > 2e-4) {
        std.debug.print("sft logits max |diff| {e}\n", .{worst});
        return error.ParityMismatch;
    }
}
