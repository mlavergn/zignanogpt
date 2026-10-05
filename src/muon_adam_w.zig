const std = @import("std");
const log = std.log.scoped(.zignanogpt_muon_adam_w);
const mod = @import("module.zig");

/// `GPT.setup_optimizer`'s arguments (its defaults; `base_train.py` passes its own).
pub const OptimizerConfig = struct {
    unembedding_lr: f64 = 0.004,
    embedding_lr: f64 = 0.2,
    matrix_lr: f64 = 0.02,
    weight_decay: f64 = 0.0,
    scalar_lr: f64 = 0.5,
};

const GroupKind = enum { adamw, muon };

/// One parameter group, with nanochat's hyperparameters (kept in f64, rounded
/// to f32 at each step as PyTorch's 0-D tensors are).
const ParamGroup = struct {
    kind: GroupKind,
    /// Indices into `GptWeights.params`.
    indices: []usize,
    initial_lr: f64,
    lr: f64,
    beta1: f64 = 0,
    beta2: f64,
    eps: f64 = 0,
    weight_decay: f64,
    momentum: f64 = 0,
};

/// Per-parameter optimizer state.
/// The Muon parameters of one shape, orthogonalized together: their
/// momentum-applied gradients are copied into `x` (`[count * rows, cols]`)
/// and Polar Express runs as batched matmuls over all of them.
const MuonStack = struct {
    indices: []usize,
    rows: usize,
    cols: usize,
    x: mod.Tensor,
    gram: mod.Tensor,
    poly: mod.Tensor,
    product: mod.Tensor,
};

const ParamState = struct {
    /// AdamW moments and update count.
    m: ?mod.Tensor = null,
    v: ?mod.Tensor = null,
    step: u32 = 0,
    /// Muon momentum and factored second moment.
    momentum: ?mod.Tensor = null,
    second: ?mod.Tensor = null,
};

/// nanochat's `MuonAdamW` (single process): Muon for the transformer matrices,
/// AdamW for embeddings, `lm_head` and scalars, grouped as `setup_optimizer`
/// does. Muon per matrix: Nesterov momentum, MuonEq, 5 Polar Express steps,
/// Muon+, NorMuon variance reduction, cautious weight decay.
pub const MuonAdamW = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    backend: *mod.Backend,
    arena: std.heap.ArenaAllocator,
    groups: []ParamGroup,
    states: []ParamState,
    /// The Muon group's parameters by shape, in parameter order.
    stacks: []MuonStack,

    /// Builds the groups and zeroed state for `weights`.
    ///
    /// Parameters:
    /// - `allocator`: backs the bookkeeping.
    /// - `backend`: allocates the state.
    /// - `weights`: the parameters, as named by `GptWeights`.
    /// - `model_dim`: `n_embd`, for the AdamW `1/sqrt(dmodel / 768)` LR scale.
    /// - `config`: the learning rates and Muon weight decay.
    ///
    /// Return: the optimizer; allocation errors, `error.UnknownParam`.
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, weights: *const mod.GptWeights, model_dim: usize, config: OptimizerConfig) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = undefined;
        self.allocator = allocator;
        self.backend = backend;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();

        const scale = std.math.pow(f64, @as(f64, @floatFromInt(model_dim)) / 768.0, -0.5);
        log.info("scaling the AdamW LRs by 1/sqrt({d}/768) = {d:.6}", .{ model_dim, scale });
        const Bucket = enum { lm_head, embedding, value_embeds, resid, x0, smear, matrix };
        var buckets: [@typeInfo(Bucket).@"enum".fields.len]std.ArrayList(usize) = @splat(.empty);
        for (weights.params, 0..) |p, i| {
            const bucket: Bucket = if (std.mem.eql(u8, p.name, "lm_head.weight"))
                .lm_head
            else if (std.mem.eql(u8, p.name, "transformer.wte.weight"))
                .embedding
            else if (std.mem.startsWith(u8, p.name, "value_embeds."))
                .value_embeds
            else if (std.mem.eql(u8, p.name, "resid_lambdas"))
                .resid
            else if (std.mem.eql(u8, p.name, "x0_lambdas"))
                .x0
            else if (std.mem.eql(u8, p.name, "smear_gate.weight") or std.mem.eql(u8, p.name, "smear_lambda") or std.mem.eql(u8, p.name, "backout_lambda"))
                .smear
            else if (std.mem.startsWith(u8, p.name, "transformer.h."))
                .matrix
            else {
                log.err("no optimizer group for parameter {s}", .{p.name});
                return error.UnknownParam;
            };
            try buckets[@intFromEnum(bucket)].append(arena, i);
        }

        var groups: std.ArrayList(ParamGroup) = .empty;
        const adamw = struct {
            fn group(indices: []usize, lr: f64, beta1: f64, beta2: f64, wd: f64) ParamGroup {
                return .{ .kind = .adamw, .indices = indices, .initial_lr = lr, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = 1e-10, .weight_decay = wd };
            }
        };
        try groups.append(arena, adamw.group(buckets[@intFromEnum(Bucket.lm_head)].items, config.unembedding_lr * scale, 0.8, 0.96, 0.01));
        try groups.append(arena, adamw.group(buckets[@intFromEnum(Bucket.embedding)].items, config.embedding_lr * scale, 0.8, 0.995, 0.001));
        try groups.append(arena, adamw.group(buckets[@intFromEnum(Bucket.value_embeds)].items, config.embedding_lr * scale * 0.5, 0.8, 0.995, 0.01));
        try groups.append(arena, adamw.group(buckets[@intFromEnum(Bucket.resid)].items, config.scalar_lr * 0.01, 0.8, 0.95, 0.05));
        try groups.append(arena, adamw.group(buckets[@intFromEnum(Bucket.x0)].items, config.scalar_lr, 0.96, 0.95, 0.0));
        try groups.append(arena, adamw.group(buckets[@intFromEnum(Bucket.smear)].items, 0.2, 0.8, 0.95, 0.0));
        // Muon groups are per shape in nanochat only so params can be stacked;
        // the update is per matrix, so one group serves.
        try groups.append(arena, .{
            .kind = .muon,
            .indices = buckets[@intFromEnum(Bucket.matrix)].items,
            .initial_lr = config.matrix_lr,
            .lr = config.matrix_lr,
            .beta2 = 0.9,
            .weight_decay = config.weight_decay,
            .momentum = 0.95,
        });
        self.groups = try groups.toOwnedSlice(arena);

        self.states = try arena.alloc(ParamState, weights.params.len);
        @memset(self.states, .{});
        self.stacks = &.{};
        errdefer self.freeStates();
        var stacks: std.ArrayList(MuonStack) = .empty;
        for (self.groups) |group| {
            for (group.indices) |i| {
                const t = weights.params[i].tensor;
                const state = &self.states[i];
                switch (group.kind) {
                    .adamw => {
                        state.m = try backend.alloc(.f32, t.shape.slice());
                        state.v = try backend.alloc(.f32, t.shape.slice());
                    },
                    .muon => {
                        const rows = t.shape.rows();
                        const cols = t.shape.cols();
                        state.momentum = try backend.alloc(.f32, t.shape.slice());
                        state.second = try backend.alloc(.f32, &mod.MuonParams.secondShape(rows, cols));
                        for (stacks.items) |stack| {
                            if (stack.rows == rows and stack.cols == cols) break;
                        } else try stacks.append(arena, .{ .indices = &.{}, .rows = rows, .cols = cols, .x = undefined, .gram = undefined, .poly = undefined, .product = undefined });
                    },
                }
            }
        }
        // Each stack lists its parameters, in parameter order.
        for (stacks.items) |*stack| {
            var indices: std.ArrayList(usize) = .empty;
            for (self.groups) |group| {
                if (group.kind != .muon) continue;
                for (group.indices) |i| {
                    const shape = weights.params[i].tensor.shape;
                    if (shape.rows() == stack.rows and shape.cols() == stack.cols) try indices.append(arena, i);
                }
            }
            stack.indices = indices.items;
        }
        self.stacks = stacks.items;
        for (self.stacks, 0..) |*stack, made| {
            errdefer for (self.stacks[0..made]) |done| freeStack(backend, done);
            const n = stack.indices.len;
            const k = @min(stack.rows, stack.cols);
            stack.x = try backend.alloc(.f32, &.{ n * stack.rows, stack.cols });
            errdefer backend.free(stack.x);
            stack.product = try backend.alloc(.f32, &.{ n * stack.rows, stack.cols });
            errdefer backend.free(stack.product);
            stack.gram = try backend.alloc(.f32, &.{ n * k, k });
            errdefer backend.free(stack.gram);
            stack.poly = try backend.alloc(.f32, &.{ n * k, k });
        }
        return self;
    }

    /// Frees the state.
    ///
    /// Parameters:
    /// - `self`: the optimizer.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (self.stacks) |stack| freeStack(self.backend, stack);
        self.freeStates();
        self.arena.deinit();
    }

    /// Applies the step's schedule, as `base_train.py` does before each step.
    ///
    /// Parameters:
    /// - `self`: the optimizer.
    /// - `lr_multiplier`: every group's `lr = initial_lr * lr_multiplier`.
    /// - `muon_momentum`: the Muon momentum.
    /// - `muon_weight_decay`: the Muon weight decay.
    ///
    /// Return: nothing.
    pub fn setSchedule(self: *Self, lr_multiplier: f64, muon_momentum: f64, muon_weight_decay: f64) void {
        for (self.groups) |*group| {
            group.lr = group.initial_lr * lr_multiplier;
            if (group.kind == .muon) {
                group.momentum = muon_momentum;
                group.weight_decay = muon_weight_decay;
            }
        }
    }

    /// Multiplies every group's learning rate, and makes the result its new
    /// initial LR (`chat_sft.py`'s `init_lr_frac`).
    ///
    /// Parameters:
    /// - `self`: the optimizer.
    /// - `factor`: the multiplier.
    ///
    /// Return: nothing.
    pub fn scaleLearningRates(self: *Self, factor: f64) void {
        for (self.groups) |*group| {
            group.lr *= factor;
            group.initial_lr = group.lr;
        }
    }

    /// One optimizer step. Muon groups use `grads` as scratch (they hold the
    /// orthogonalized update afterwards); zero them before the next step.
    ///
    /// Parameters:
    /// - `self`: the optimizer.
    /// - `weights`: updated in place.
    /// - `grads`: the gradients, same layout as `weights`.
    ///
    /// Return: nothing; backend errors.
    pub fn step(self: *Self, weights: *mod.GptWeights, grads: *mod.GptWeights) !void {
        for (self.groups) |group| {
            if (group.kind == .muon) {
                for (self.stacks) |stack| try self.muonStack(group, stack, weights, grads);
                continue;
            }
            for (group.indices) |i| {
                const p = weights.params[i].tensor;
                const g = grads.params[i].tensor;
                const state = &self.states[i];
                switch (group.kind) {
                    .adamw => {
                        state.step += 1;
                        try self.backend.adamwStep(p, g, state.m.?, state.v.?, .{
                            .lr = @floatCast(group.lr),
                            .beta1 = @floatCast(group.beta1),
                            .beta2 = @floatCast(group.beta2),
                            .eps = @floatCast(group.eps),
                            .weight_decay = @floatCast(group.weight_decay),
                            .step = state.step,
                        });
                    },
                    .muon => unreachable,
                }
            }
        }
    }

    /// Queues the optimizer state for a checkpoint: per parameter
    /// `adamw.{m,v,step}.<name>` or `muon.{momentum,second}.<name>`.
    ///
    /// Parameters:
    /// - `self`: the optimizer.
    /// - `writer`: receives the tensors.
    /// - `weights`: names the parameters (the layout the optimizer was built for).
    ///
    /// Return: nothing; allocation errors.
    pub fn addState(self: *const Self, writer: *mod.SafeTensorsWriter, weights: *const mod.GptWeights) !void {
        var name_buf: [128]u8 = undefined;
        for (self.states, weights.params) |state, p| {
            if (state.m) |m| {
                try writer.addTensor(try std.fmt.bufPrint(&name_buf, "adamw.m.{s}", .{p.name}), m);
                try writer.addTensor(try std.fmt.bufPrint(&name_buf, "adamw.v.{s}", .{p.name}), state.v.?);
                try writer.addHost(try std.fmt.bufPrint(&name_buf, "adamw.step.{s}", .{p.name}), i32, &.{1}, &.{@intCast(state.step)});
            }
            if (state.momentum) |b| {
                try writer.addTensor(try std.fmt.bufPrint(&name_buf, "muon.momentum.{s}", .{p.name}), b);
                try writer.addTensor(try std.fmt.bufPrint(&name_buf, "muon.second.{s}", .{p.name}), state.second.?);
            }
        }
    }

    /// Restores state written by `addState`.
    ///
    /// Parameters:
    /// - `self`: the optimizer, built for the same weights.
    /// - `file`: the optimizer checkpoint.
    /// - `weights`: names the parameters.
    ///
    /// Return: nothing; `error.MissingTensor`, shape errors.
    pub fn loadState(self: *Self, file: *const mod.SafeTensors, weights: *const mod.GptWeights) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var name_buf: [128]u8 = undefined;
        for (self.states, weights.params) |*state, p| {
            if (state.m) |m| {
                try self.loadTensor(file, try std.fmt.bufPrint(&name_buf, "adamw.m.{s}", .{p.name}), m);
                try self.loadTensor(file, try std.fmt.bufPrint(&name_buf, "adamw.v.{s}", .{p.name}), state.v.?);
                var step_value: [1]i32 = undefined;
                try file.read(try std.fmt.bufPrint(&name_buf, "adamw.step.{s}", .{p.name}), i32, &step_value);
                state.step = std.math.cast(u32, step_value[0]) orelse return error.InvalidCheckpoint;
            }
            if (state.momentum) |b| {
                try self.loadTensor(file, try std.fmt.bufPrint(&name_buf, "muon.momentum.{s}", .{p.name}), b);
                try self.loadTensor(file, try std.fmt.bufPrint(&name_buf, "muon.second.{s}", .{p.name}), state.second.?);
            }
        }
    }

    fn loadTensor(self: *Self, file: *const mod.SafeTensors, name: []const u8, tensor: mod.Tensor) !void {
        const host = try file.readAlloc(self.allocator, name, f32);
        defer self.allocator.free(host);
        try self.backend.upload(tensor, f32, host);
    }

    /// Muon on one matrix.
    /// Muon on one shape stack: momentum and MuonEq per matrix, Polar Express
    /// as batched matmuls over the stack, then each matrix's update.
    fn muonStack(self: *Self, group: ParamGroup, stack: MuonStack, weights: *mod.GptWeights, grads: *mod.GptWeights) !void {
        const be = self.backend;
        const rows = stack.rows;
        const cols = stack.cols;
        const n = stack.indices.len;
        for (stack.indices, 0..) |i, slot| {
            const g = try grads.params[i].tensor.reshape(&.{ rows, cols });
            try be.muonMomentum(g, self.states[i].momentum.?, @floatCast(group.momentum));
            const x = try stack.x.rows(slot * rows, rows);
            try be.copy(x, g);
            try be.muonPrepare(x);
        }

        // Polar Express: X <- a X + (b A + c A^2) applied from the short side.
        const tall = rows > cols;
        for (mod.polar_express_coeffs) |coeffs| {
            const a, const b, const c = coeffs;
            if (tall) {
                try be.matmul(stack.gram, stack.x, stack.x, .{ .transpose_a = true, .batch = n }); // X^T X
            } else {
                try be.matmul(stack.gram, stack.x, stack.x, .{ .transpose_b = true, .batch = n }); // X X^T
            }
            try be.matmul(stack.poly, stack.gram, stack.gram, .{ .alpha = c, .batch = n });
            try be.combine(stack.poly, stack.gram, mod.Scalar.constant(b), stack.poly, mod.Scalar.constant(1));
            if (tall) {
                try be.matmul(stack.product, stack.x, stack.poly, .{ .batch = n });
            } else {
                try be.matmul(stack.product, stack.poly, stack.x, .{ .batch = n });
            }
            try be.combine(stack.x, stack.x, mod.Scalar.constant(a), stack.product, mod.Scalar.constant(1));
        }

        const aspect = @max(1.0, @as(f64, @floatFromInt(rows)) / @as(f64, @floatFromInt(cols)));
        for (stack.indices, 0..) |i, slot| {
            try be.muonFinish(try weights.params[i].tensor.reshape(&.{ rows, cols }), try stack.x.rows(slot * rows, rows), self.states[i].second.?, .{
                .lr = @floatCast(group.lr * @sqrt(aspect)),
                .weight_decay = @floatCast(group.weight_decay),
                .beta2 = @floatCast(group.beta2),
            });
        }
    }

    fn freeStack(backend: *mod.Backend, stack: MuonStack) void {
        backend.free(stack.poly);
        backend.free(stack.gram);
        backend.free(stack.product);
        backend.free(stack.x);
    }

    fn freeStates(self: *Self) void {
        for (self.states) |state| {
            inline for (.{ state.m, state.v, state.momentum, state.second }) |t| {
                if (t) |tensor| self.backend.free(tensor);
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "muon adamw groups parameters like setup_optimizer" {
    var backend = try mod.Backend.init(std.testing.allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 32, .vocab_size = 70, .n_layer = 2, .n_head = 2, .n_kv_head = 1, .n_embd = 32 };
    var weights = try mod.GptWeights.init(std.testing.allocator, &backend, config);
    defer weights.deinit();
    var optimizer = try mod.MuonAdamW.init(std.testing.allocator, &backend, &weights, config.n_embd, .{});
    defer optimizer.deinit();
    var total: usize = 0;
    for (optimizer.groups) |g| total += g.indices.len;
    try std.testing.expectEqual(weights.params.len, total);
    // 2 layers x (q, k, v, proj, fc, mlp proj) + 1 ve_gate (layer 1)
    try std.testing.expectEqual(@as(usize, 13), optimizer.groups[6].indices.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.004 * @sqrt(24.0)), optimizer.groups[0].initial_lr, 1e-12);
    optimizer.setSchedule(0.5, 0.9, 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), optimizer.groups[6].lr, 1e-12);
    try std.testing.expectEqual(@as(f64, 0.9), optimizer.groups[6].momentum);
}

test "muon adamw steps match pytorch on the fixture" {
    const allocator = std.testing.allocator;
    var file = try mod.SafeTensors.load(allocator, std.testing.io, mod.build_options.source_root ++ "/testdata/optim.safetensors");
    defer file.deinit();
    const config = mod.GptConfig{
        .sequence_len = try file.metadataInt(usize, "sequence_len"),
        .vocab_size = try file.metadataInt(usize, "vocab_size"),
        .n_layer = try file.metadataInt(usize, "n_layer"),
        .n_head = try file.metadataInt(usize, "n_head"),
        .n_kv_head = try file.metadataInt(usize, "n_kv_head"),
        .n_embd = try file.metadataInt(usize, "n_embd"),
    };
    const meta = struct {
        fn float(f: *const mod.SafeTensors, key: []const u8) !f64 {
            return std.fmt.parseFloat(f64, f.metadata(key) orelse return error.MissingMetadata);
        }
    };
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var weights = try mod.GptWeights.init(allocator, &backend, config);
    defer weights.deinit();
    var grads = try mod.GptWeights.init(allocator, &backend, config);
    defer grads.deinit();
    try weights.load(allocator, &file, "init.");
    var optimizer = try mod.MuonAdamW.init(allocator, &backend, &weights, config.n_embd, .{
        .unembedding_lr = try meta.float(&file, "unembedding_lr"),
        .embedding_lr = try meta.float(&file, "embedding_lr"),
        .matrix_lr = try meta.float(&file, "matrix_lr"),
        .weight_decay = try meta.float(&file, "weight_decay"),
        .scalar_lr = try meta.float(&file, "scalar_lr"),
    });
    defer optimizer.deinit();

    var schedule = std.mem.splitScalar(u8, file.metadata("schedule") orelse return error.MissingMetadata, ';');
    var step: usize = 0;
    while (schedule.next()) |entry| : (step += 1) {
        var fields = std.mem.splitScalar(u8, entry, ',');
        const lrm = try std.fmt.parseFloat(f64, fields.next().?);
        const momentum = try std.fmt.parseFloat(f64, fields.next().?);
        const wd = try std.fmt.parseFloat(f64, fields.next().?);
        var prefix_buf: [16]u8 = undefined;
        try grads.load(allocator, &file, try std.fmt.bufPrint(&prefix_buf, "grad{d}.", .{step}));
        optimizer.setSchedule(lrm, momentum, wd);
        try optimizer.step(&weights, &grads);

        for (weights.params) |p| {
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "step{d}.{s}", .{ step, p.name });
            const want = try file.readAlloc(allocator, name, f32);
            defer allocator.free(want);
            const got = try allocator.alloc(f32, p.tensor.numel());
            defer allocator.free(got);
            try backend.download(p.tensor, f32, got);
            var worst: f32 = 0;
            var peak: f32 = 0;
            for (want, got) |w, g| {
                worst = @max(worst, @abs(w - g));
                peak = @max(peak, @abs(w));
            }
            if (worst > 1e-5 * peak + 1e-7) {
                std.debug.print("{s}: max |diff| {e} (peak {e})\n", .{ name, worst, peak });
                return error.ParityMismatch;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 3), step);
}

test "muon adamw state survives a checkpoint round trip" {
    const allocator = std.testing.allocator;
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 16, .vocab_size = 40, .n_layer = 2, .n_head = 2, .n_kv_head = 1, .n_embd = 32 };
    var rng = mod.Random.init(3);
    var a = try mod.GptWeights.init(allocator, &backend, config);
    defer a.deinit();
    var grads = try mod.GptWeights.init(allocator, &backend, config);
    defer grads.deinit();
    const randomize = struct {
        fn run(be: *mod.Backend, w: *mod.GptWeights, r: *mod.Random, alloc: std.mem.Allocator) !void {
            for (w.params) |p| {
                const host = try alloc.alloc(f32, p.tensor.numel());
                defer alloc.free(host);
                r.fillUniform(host, -0.1, 0.1);
                try be.upload(p.tensor, f32, host);
            }
        }
    }.run;
    try randomize(&backend, &a, &rng, allocator);
    var opt_a = try mod.MuonAdamW.init(allocator, &backend, &a, config.n_embd, .{ .weight_decay = 0.1 });
    defer opt_a.deinit();
    try randomize(&backend, &grads, &rng, allocator);
    try opt_a.step(&a, &grads);

    // Save weights + optimizer, restore into fresh copies.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    var ckpt = try mod.Checkpoint.init(allocator, storage, root, .base, "d2");
    defer ckpt.deinit();
    try ckpt.saveModel(&backend, 1, &a);
    try ckpt.saveOptimizer(&backend, 1, &opt_a, &a);
    var b = try mod.GptWeights.init(allocator, &backend, config);
    defer b.deinit();
    try ckpt.loadModel(1, &b);
    var opt_b = try mod.MuonAdamW.init(allocator, &backend, &b, config.n_embd, .{ .weight_decay = 0.1 });
    defer opt_b.deinit();
    try ckpt.loadOptimizer(1, &opt_b, &b);

    // The same next step from both gives the same weights.
    var g2 = try mod.GptWeights.init(allocator, &backend, config);
    defer g2.deinit();
    try randomize(&backend, &grads, &rng, allocator);
    for (grads.params, g2.params) |src, dst| try backend.copy(dst.tensor, src.tensor);
    try opt_a.step(&a, &grads);
    try opt_b.step(&b, &g2);
    for (a.params, b.params) |pa, pb| {
        const ha = try allocator.alloc(f32, pa.tensor.numel());
        defer allocator.free(ha);
        const hb = try allocator.alloc(f32, pb.tensor.numel());
        defer allocator.free(hb);
        try backend.download(pa.tensor, f32, ha);
        try backend.download(pb.tensor, f32, hb);
        try std.testing.expectEqualSlices(f32, ha, hb);
    }
}

test "muon adamw training overfits one batch" {
    const allocator = std.testing.allocator;
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 16, .vocab_size = 40, .n_layer = 2, .n_head = 2, .n_kv_head = 1, .n_embd = 32, .window_pattern = "L" };
    var model = try mod.Gpt.init(allocator, &backend, config);
    defer model.deinit();
    var rng = mod.Random.init(5);
    try model.initWeights(&rng);
    var grads = try mod.GptWeights.init(allocator, &backend, config);
    defer grads.deinit();
    var acts = try mod.GptActivations.init(allocator, &backend, config, 4, 16);
    defer acts.deinit();
    var bufs = try mod.GptGradBuffers.init(allocator, &backend, config, 4, 16);
    defer bufs.deinit();
    var optimizer = try mod.MuonAdamW.init(allocator, &backend, &model.weights, config.n_embd, .{ .weight_decay = 0.0 });
    defer optimizer.deinit();
    const schedule = mod.TrainSchedule{ .num_iterations = 60, .warmup_steps = 5 };

    var ids: [64]i32 = undefined;
    var next: [64]i32 = undefined;
    for (&ids, &next) |*a, *b| {
        a.* = @intCast(rng.below(40));
        b.* = @intCast(rng.below(40));
    }
    const idx = try backend.alloc(.i32, &.{ 4, 16 });
    defer backend.free(idx);
    const targets = try backend.alloc(.i32, &.{ 4, 16 });
    defer backend.free(targets);
    try backend.upload(idx, i32, &ids);
    try backend.upload(targets, i32, &next);
    const loss = try backend.alloc(.f32, &.{1});
    defer backend.free(loss);

    var first: [1]f32 = undefined;
    var last: [1]f32 = undefined;
    for (0..schedule.num_iterations) |it| {
        try model.forward(&acts, idx);
        try model.loss(&acts, targets, loss);
        try backend.download(loss, f32, if (it == 0) &first else &last);
        try grads.zero();
        try model.backward(&acts, &bufs, &grads, idx, targets, 1);
        schedule.apply(&optimizer, it);
        try optimizer.step(&model.weights, &grads);
    }
    // ln(40) ~ 3.7 at the start; memorizing 64 random pairs drives it well down.
    try std.testing.expect(first[0] > 3.0);
    try std.testing.expect(last[0] < 0.5 * first[0]);
}
