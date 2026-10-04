const std = @import("std");
const log = std.log.scoped(.zignanogpt_gpt);
const mod = @import("module.zig");

/// nanochat's GPT (`nanochat/gpt.py`): rotary embeddings, QK norm, untied
/// embedding and head, ReLU^2 MLP, weightless RMS norm, no biases, GQA,
/// sliding windows, value embeddings, smear, backout, per-layer residual
/// scalars, logit soft cap. Forward, loss and a hand-written backward pass.
pub const Gpt = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    backend: *mod.Backend,
    config: mod.GptConfig,
    weights: mod.GptWeights,
    /// Rotary tables `[rotary_seq_len, head_dim / 2]`; not parameters.
    cos: mod.Tensor,
    sin: mod.Tensor,

    /// Allocates a zero-initialized model and computes its rotary tables.
    ///
    /// Parameters:
    /// - `allocator`: owns the bookkeeping.
    /// - `backend`: allocates the tensors.
    /// - `config`: the model shape; validated here.
    ///
    /// Return: the model (weights all zero until `initWeights` or a load).
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, config: mod.GptConfig) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        try config.validate();
        var self: Self = undefined;
        self.allocator = allocator;
        self.backend = backend;
        self.config = config;
        self.weights = try mod.GptWeights.init(allocator, backend, config);
        errdefer self.weights.deinit();
        try self.initRotary();
        return self;
    }

    /// Frees every tensor.
    ///
    /// Parameters:
    /// - `self`: the model.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.backend.free(self.sin);
        self.backend.free(self.cos);
        self.weights.deinit();
    }

    /// nanochat's `init_weights`: random matrices, zeroed output projections,
    /// depth-dependent residual scalars. Draws differ from PyTorch's generator.
    ///
    /// Parameters:
    /// - `self`: the model.
    /// - `rng`: the random source.
    ///
    /// Return: nothing; allocation or upload errors.
    pub fn initWeights(self: *Self, rng: *mod.Random) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const cfg = self.config;
        const w = &self.weights;
        const s: f32 = @floatCast(@sqrt(3.0) / @sqrt(@as(f64, @floatFromInt(cfg.n_embd))));
        try self.fillNormal(rng, w.wte, 0.8);
        try self.fillNormal(rng, w.lm_head, 0.001);
        for (w.layers) |layer| {
            try self.fillUniform(rng, layer.c_q, -s, s);
            try self.fillUniform(rng, layer.c_k, -s, s);
            try self.fillUniform(rng, layer.c_v, -s, s);
            try self.backend.fill(layer.c_proj, 0);
            try self.fillUniform(rng, layer.c_fc, -s * 0.4, s * 0.4);
            try self.backend.fill(layer.mlp_proj, 0);
            if (layer.value_embed) |t| try self.fillUniform(rng, t, -s, s);
            if (layer.ve_gate) |t| try self.fillUniform(rng, t, 0, 0.02);
        }
        const resid = try self.allocator.alloc(f32, cfg.n_layer);
        defer self.allocator.free(resid);
        const x0 = try self.allocator.alloc(f32, cfg.n_layer);
        defer self.allocator.free(x0);
        const span: f32 = @floatFromInt(@max(cfg.n_layer - 1, 1));
        for (resid, x0, 0..) |*r, *z, i| {
            const depth = @as(f32, @floatFromInt(i)) / span;
            r.* = 1.15 - 0.10 * depth;
            z.* = 0.20 - 0.15 * depth;
        }
        try self.backend.upload(w.resid_lambdas, f32, resid);
        try self.backend.upload(w.x0_lambdas, f32, x0);
        try self.backend.fill(w.smear_lambda, 0);
        try self.backend.fill(w.backout_lambda, 0.2);
        try self.fillUniform(rng, w.smear_gate, 0, 0.02);
    }

    /// The forward pass over a batch of token ids, leaving the logits (and
    /// every intermediate the backward pass needs) in `acts`.
    ///
    /// Parameters:
    /// - `self`: the model.
    /// - `acts`: sized for `idx`'s `[B, T]`; `acts.logits` receives `[B, T, V]`.
    /// - `idx`: `[B, T]` i32 token ids.
    ///
    /// Return: nothing; backend errors.
    pub fn forward(self: *Self, acts: *mod.GptActivations, idx: mod.Tensor) !void {
        const be = self.backend;
        const cfg = self.config;
        const w = &self.weights;
        const eps = mod.GptConfig.rms_eps;
        const bt = acts.batch * acts.seq;
        const c = cfg.n_embd;
        if (idx.numel() != bt) return error.ShapeMismatch;

        // Embed, normalize, smear the previous token in.
        try be.embedding(acts.emb, w.wte, idx);
        try be.rmsnorm(acts.emb_norm, acts.emb, eps);
        try be.gateLinear(acts.smear_gate, try acts.emb_norm.reshape(&.{ bt, c }), w.smear_gate);
        try be.smear(acts.x0, acts.emb_norm, acts.smear_gate, mod.Scalar.of(w.smear_lambda, 1));

        var x = acts.x0;
        for (w.layers, acts.layers, 0..) |layer, la, i| {
            try be.combine(la.x_in, x, mod.Scalar.of(try w.resid_lambdas.rows(i, 1), 1), acts.x0, mod.Scalar.of(try w.x0_lambdas.rows(i, 1), 1));
            try self.attention(layer, la, idx, cfg.windowSize(i), acts.tmp);
            try be.add(la.x_mid, la.x_in, acts.tmp);
            try be.rmsnorm(la.xn2, la.x_mid, eps);
            try be.matmul(la.h, la.xn2, layer.c_fc, .{ .transpose_b = true });
            try be.reluSquare(la.a, la.h);
            try be.matmul(acts.tmp, la.a, layer.mlp_proj, .{ .transpose_b = true });
            try be.add(la.x_out, la.x_mid, acts.tmp);
            x = la.x_out;
        }

        // Backout: subtract the mid-depth residual before the final norm.
        const mid = acts.layers[cfg.backoutLayer()].x_out;
        try be.combine(acts.x_final, x, mod.Scalar.constant(1), mid, mod.Scalar.of(w.backout_lambda, -1));
        try be.rmsnorm(acts.xn_final, acts.x_final, eps);
        try be.matmul(acts.logits_pad, acts.xn_final, w.lm_head, .{ .transpose_b = true });
        try be.softcap(try acts.logits.reshape(&.{ bt, cfg.vocab_size }), acts.logits_pad, mod.GptConfig.softcap);
    }

    /// Mean cross-entropy of the logits `forward` left in `acts`.
    ///
    /// Parameters:
    /// - `self`: the model.
    /// - `acts`: after `forward`.
    /// - `targets`: `[B, T]` i32 next tokens, -1 where ignored.
    /// - `loss`: element 0 receives the loss.
    ///
    /// Return: nothing; backend errors.
    pub fn loss(self: *Self, acts: *const mod.GptActivations, targets: mod.Tensor, out: mod.Tensor) !void {
        const bt = acts.batch * acts.seq;
        try self.backend.crossEntropy(out, try acts.logits.reshape(&.{ bt, self.config.vocab_size }), targets);
    }

    /// Backpropagates `scale * loss` from the activations of the last
    /// `forward`, adding every parameter's gradient into `grads`.
    ///
    /// Parameters:
    /// - `self`: the model.
    /// - `acts`: after `forward` on `idx`.
    /// - `bufs`: activation-gradient scratch for the same `[B, T]`.
    /// - `grads`: accumulated into (zero it to start a step).
    /// - `idx`: the `[B, T]` ids `forward` saw.
    /// - `targets`: `[B, T]` i32 next tokens, -1 where ignored.
    /// - `scale`: multiplies the gradient (`1 / grad_accum_steps`).
    ///
    /// Return: nothing; backend errors.
    pub fn backward(self: *Self, acts: *const mod.GptActivations, bufs: *mod.GptGradBuffers, grads: *mod.GptWeights, idx: mod.Tensor, targets: mod.Tensor, scale: f32) !void {
        const be = self.backend;
        const cfg = self.config;
        const w = &self.weights;
        const g = bufs;
        const eps = mod.GptConfig.rms_eps;
        const bt = acts.batch * acts.seq;
        const c = cfg.n_embd;
        const acc = mod.MatmulOptions{ .transpose_a = true, .accumulate = true };

        // Loss, soft cap and lm_head.
        try be.crossEntropyBackward(g.dlogits_pad, try acts.logits.reshape(&.{ bt, cfg.vocab_size }), targets, mod.GptConfig.softcap, scale);
        try be.matmul(g.dxn, g.dlogits_pad, w.lm_head, .{});
        try be.matmul(grads.lm_head, g.dlogits_pad, acts.xn_final, acc);
        try be.rmsnormBackward(g.dx, g.dxn, acts.x_final, eps, false);

        // Backout: x_final = x_last - lambda * mid.
        const mid_layer = cfg.backoutLayer();
        try be.dot(grads.backout_lambda, g.dx, acts.layers[mid_layer].x_out, -1, true);
        try be.combine(g.dmid, g.dx, mod.Scalar.of(w.backout_lambda, -1), g.dx, mod.Scalar.constant(0));

        try be.fill(g.dx0, 0);
        var i = cfg.n_layer;
        while (i > 0) {
            i -= 1;
            // g.dx is the gradient of this block's output.
            if (i == mid_layer) try be.add(g.dx, g.dx, g.dmid);
            try self.blockBackward(w.layers[i], &grads.layers[i], acts.layers[i], g, idx, cfg.windowSize(i));

            // x_in = resid[i] * x_prev + x0l[i] * x0.
            const x_prev = if (i == 0) acts.x0 else acts.layers[i - 1].x_out;
            const resid = try w.resid_lambdas.rows(i, 1);
            const x0l = try w.x0_lambdas.rows(i, 1);
            try be.dot(try grads.resid_lambdas.rows(i, 1), g.dx, x_prev, 1, true);
            try be.dot(try grads.x0_lambdas.rows(i, 1), g.dx, acts.x0, 1, true);
            try be.combine(g.dx0, g.dx0, mod.Scalar.constant(1), g.dx, mod.Scalar.of(x0l, 1));
            try be.combine(g.dx, g.dx, mod.Scalar.of(resid, 1), g.dx, mod.Scalar.constant(0));
        }
        // Layer 0's x_prev is x0 itself.
        try be.add(g.dx0, g.dx0, g.dx);

        // Smear, its gate, the embedding norm, the embedding.
        const emb_norm = try acts.emb_norm.reshape(&.{ bt, c });
        try be.smearBackward(g.demb_norm, g.dsmear_gate, grads.smear_lambda, g.dx0, acts.emb_norm, acts.smear_gate, mod.Scalar.of(w.smear_lambda, 1));
        try be.gateLinearBackward(try g.demb_norm.reshape(&.{ bt, c }), grads.smear_gate, g.dsmear_gate, emb_norm, w.smear_gate);
        try be.rmsnormBackward(g.demb, g.demb_norm, acts.emb, eps, false);
        try be.embeddingBackward(grads.wte, g.demb, idx);
    }

    /// Greedy decoding without a KV cache: each step runs the whole forward
    /// pass over the sequence so far, padded to its final length (causal
    /// attention keeps the padding from mattering), and appends the argmax.
    ///
    /// Parameters:
    /// - `self`: the model.
    /// - `allocator`: allocates the result and scratch.
    /// - `prompt`: the starting tokens (e.g. BOS + prompt), non-empty.
    /// - `max_new`: tokens to generate at most (capped by `sequence_len`).
    /// - `stop`: a token that ends generation (not appended), or null.
    ///
    /// Return: the generated tokens (without the prompt), owned by the caller.
    pub fn greedy(self: *Self, allocator: std.mem.Allocator, prompt: []const u32, max_new: usize, stop: ?u32) ![]u32 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (prompt.len == 0) return error.EmptyPrompt;
        const len = @min(prompt.len + max_new, self.config.sequence_len);
        if (len <= prompt.len) return allocator.alloc(u32, 0);
        var acts = try mod.GptActivations.init(allocator, self.backend, self.config, 1, len);
        defer acts.deinit();
        const ids = try allocator.alloc(i32, len);
        defer allocator.free(ids);
        @memset(ids, 0);
        for (prompt, 0..) |t, i| ids[i] = @intCast(t);
        const idx = try self.backend.alloc(.i32, &.{ 1, len });
        defer self.backend.free(idx);
        const row = try allocator.alloc(f32, self.config.vocab_size);
        defer allocator.free(row);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        var n = prompt.len;
        while (n < len) : (n += 1) {
            try self.backend.upload(idx, i32, ids);
            try self.forward(&acts, idx);
            const logits = try acts.logits.reshape(&.{ len, self.config.vocab_size });
            try self.backend.download(try logits.rows(n - 1, 1), f32, row);
            const next: u32 = @intCast(std.mem.indexOfMax(f32, row));
            if (stop != null and next == stop.?) break;
            try out.append(allocator, next);
            ids[n] = @intCast(next);
        }
        return out.toOwnedSlice(allocator);
    }

    /// One block's backward: on entry `g.dx` is the gradient of the block
    /// output; on exit it is the gradient of the block input `x_in`.
    fn blockBackward(self: *Self, layer: mod.GptLayer, grad: *mod.GptLayer, la: mod.GptLayerActivations, g: *mod.GptGradBuffers, idx: mod.Tensor, window: usize) !void {
        const be = self.backend;
        const cfg = self.config;
        const eps = mod.GptConfig.rms_eps;
        const bt = la.x_in.shape.rows();
        const c = cfg.n_embd;
        const kv = cfg.kvDim();
        const acc = mod.MatmulOptions{ .transpose_a = true, .accumulate = true };

        // MLP: x_out = x_mid + c_proj(relu(c_fc(norm(x_mid)))^2).
        try be.matmul(g.da, g.dx, layer.mlp_proj, .{});
        try be.matmul(grad.mlp_proj, g.dx, la.a, acc);
        try be.reluSquareBackward(g.dh, g.da, la.h);
        try be.matmul(g.dxn, g.dh, layer.c_fc, .{});
        try be.matmul(grad.c_fc, g.dh, la.xn2, acc);
        try be.rmsnormBackward(g.dx, g.dxn, la.x_mid, eps, true);

        // Attention: x_mid = x_in + c_proj(attn(q, k, v)).
        const dy = try g.dy.reshape(&.{ bt, c });
        try be.matmul(dy, g.dx, layer.c_proj, .{});
        try be.matmul(grad.c_proj, g.dx, try la.y.reshape(&.{ bt, c }), acc);
        try be.attentionBackward(g.dq, g.dk, g.dv, g.dy, la.q, la.k, la.v, la.y, la.lse, .{ .window = window });

        // q = 1.2 * norm(rope(c_q(xn))), likewise k.
        try be.scale(g.dq, g.dq, mod.GptConfig.qk_scale);
        try be.rmsnormBackward(g.dq, g.dq, la.q_rot, eps, false);
        try be.ropeBackward(g.dq, g.dq, self.cos, self.sin, 0);
        try be.scale(g.dk, g.dk, mod.GptConfig.qk_scale);
        try be.rmsnormBackward(g.dk, g.dk, la.k_rot, eps, false);
        try be.ropeBackward(g.dk, g.dk, self.cos, self.sin, 0);

        const dq = try g.dq.reshape(&.{ bt, c });
        const dk = try g.dk.reshape(&.{ bt, kv });
        const dv = try g.dv.reshape(&.{ bt, kv });
        try be.matmul(g.dxn, dq, layer.c_q, .{});
        try be.matmul(g.dxn, dk, layer.c_k, .{ .accumulate = true });
        try be.matmul(g.dxn, dv, layer.c_v, .{ .accumulate = true });
        try be.matmul(grad.c_q, dq, la.xn, acc);
        try be.matmul(grad.c_k, dk, la.xn, acc);
        try be.matmul(grad.c_v, dv, la.xn, acc);

        // Value residual: v += 3 * sigmoid(ve_gate(xn[:, :12])) * value_embed(idx).
        if (layer.value_embed != null) {
            try be.valueMixBackward(g.dve, g.dve_gate, dv, la.ve.?, la.ve_gate.?);
            try be.embeddingBackward(grad.value_embed.?, g.dve, idx);
            try be.gateLinearBackward(try g.dxn.reshape(&.{ bt, c }), grad.ve_gate.?, g.dve_gate, try la.xn.reshape(&.{ bt, c }), layer.ve_gate.?);
        }
        try be.rmsnormBackward(g.dx, g.dxn, la.x_in, eps, true);
    }

    /// One block's attention half, `out = c_proj(attn(norm(x_in)))`; the caller
    /// adds `out` into the residual stream.
    fn attention(self: *Self, layer: mod.GptLayer, la: mod.GptLayerActivations, idx: mod.Tensor, window: usize, out: mod.Tensor) !void {
        const be = self.backend;
        const eps = mod.GptConfig.rms_eps;
        const bt = la.x_in.shape.rows();
        const c = self.config.n_embd;
        const kv = self.config.kvDim();
        const scale = mod.GptConfig.qk_scale;
        try be.rmsnorm(la.xn, la.x_in, eps);
        try be.matmul(try la.q_rot.reshape(&.{ bt, c }), la.xn, layer.c_q, .{ .transpose_b = true });
        try be.matmul(try la.k_rot.reshape(&.{ bt, kv }), la.xn, layer.c_k, .{ .transpose_b = true });
        try be.matmul(try la.v.reshape(&.{ bt, kv }), la.xn, layer.c_v, .{ .transpose_b = true });
        if (layer.value_embed) |table| {
            try be.embedding(la.ve.?, table, idx);
            try be.gateLinear(la.ve_gate.?, try la.xn.reshape(&.{ bt, c }), layer.ve_gate.?);
            try be.valueMix(la.v, la.ve.?, la.ve_gate.?);
        }
        try be.rope(la.q_rot, la.q_rot, self.cos, self.sin, 0);
        try be.rope(la.k_rot, la.k_rot, self.cos, self.sin, 0);
        try be.rmsnorm(la.q, la.q_rot, eps);
        try be.rmsnorm(la.k, la.k_rot, eps);
        try be.scale(la.q, la.q, scale);
        try be.scale(la.k, la.k, scale);
        try be.attention(la.y, la.q, la.k, la.v, la.lse, .{ .window = window });
        try be.matmul(out, try la.y.reshape(&.{ bt, c }), layer.c_proj, .{ .transpose_b = true });
    }

    /// Computes the rotary tables on the host, in f32 as PyTorch does, and uploads them.
    fn initRotary(self: *Self) !void {
        const half = self.config.headDim() / 2;
        const rows = self.config.rotarySeqLen();
        const cos = try self.allocator.alloc(f32, rows * half);
        defer self.allocator.free(cos);
        const sin = try self.allocator.alloc(f32, rows * half);
        defer self.allocator.free(sin);
        const head_dim: f32 = @floatFromInt(self.config.headDim());
        for (0..half) |i| {
            // inv_freq = 1 / base^(2i / head_dim), each step rounded to f32 like torch
            const exponent = @as(f32, @floatFromInt(2 * i)) / head_dim;
            const power: f32 = @floatCast(std.math.pow(f64, mod.GptConfig.rope_base, exponent));
            const inv_freq = 1.0 / power;
            for (0..rows) |t| {
                const angle = @as(f32, @floatFromInt(t)) * inv_freq;
                cos[t * half + i] = @floatCast(@cos(@as(f64, angle)));
                sin[t * half + i] = @floatCast(@sin(@as(f64, angle)));
            }
        }
        self.cos = try self.backend.alloc(.f32, &.{ rows, half });
        errdefer self.backend.free(self.cos);
        self.sin = try self.backend.alloc(.f32, &.{ rows, half });
        errdefer self.backend.free(self.sin);
        try self.backend.upload(self.cos, f32, cos);
        try self.backend.upload(self.sin, f32, sin);
    }

    fn fillNormal(self: *Self, rng: *mod.Random, t: mod.Tensor, stddev: f32) !void {
        const host = try self.allocator.alloc(f32, t.numel());
        defer self.allocator.free(host);
        rng.fillNormal(host, 0, stddev);
        try self.backend.upload(t, f32, host);
    }

    fn fillUniform(self: *Self, rng: *mod.Random, t: mod.Tensor, lo: f32, hi: f32) !void {
        const host = try self.allocator.alloc(f32, t.numel());
        defer self.allocator.free(host);
        rng.fillUniform(host, lo, hi);
        try self.backend.upload(t, f32, host);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

/// Largest absolute difference, and where.
fn maxDiff(expected: []const f32, got: []const f32) struct { diff: f32, index: usize } {
    var worst: f32 = 0;
    var at: usize = 0;
    for (expected, got, 0..) |e, g, i| {
        const d = @abs(e - g);
        if (d > worst or std.math.isNan(g)) {
            worst = if (std.math.isNan(g)) std.math.inf(f32) else d;
            at = i;
        }
    }
    return .{ .diff = worst, .index = at };
}

fn expectMatches(label: []const u8, expected: []const f32, got: []const f32, tolerance: f32) !void {
    const result = maxDiff(expected, got);
    if (result.diff > tolerance) {
        std.debug.print("{s}: max |diff| {e} at {d} (expected {d}, got {d})\n", .{ label, result.diff, result.index, expected[result.index], got[result.index] });
        return error.ParityMismatch;
    }
}

/// The fixture's model config, from its metadata.
fn fixtureConfig(file: *const mod.SafeTensors) !mod.GptConfig {
    return mod.GptConfig{
        .sequence_len = try file.metadataInt(usize, "sequence_len"),
        .vocab_size = try file.metadataInt(usize, "vocab_size"),
        .n_layer = try file.metadataInt(usize, "n_layer"),
        .n_head = try file.metadataInt(usize, "n_head"),
        .n_kv_head = try file.metadataInt(usize, "n_kv_head"),
        .n_embd = try file.metadataInt(usize, "n_embd"),
        .window_pattern = file.metadata("window_pattern") orelse return error.MissingMetadata,
    };
}

/// Uploads a fixture's `[batch, seq]` i32 tensor.
fn fixtureIds(allocator: std.mem.Allocator, backend: *mod.Backend, file: *const mod.SafeTensors, name: []const u8, batch: usize, seq: usize) !mod.Tensor {
    const host = try file.readAlloc(allocator, name, i32);
    defer allocator.free(host);
    const t = try backend.alloc(.i32, &.{ batch, seq });
    errdefer backend.free(t);
    try backend.upload(t, i32, host);
    return t;
}

fn downloadAlloc(allocator: std.mem.Allocator, backend: *mod.Backend, t: mod.Tensor) ![]f32 {
    const out = try allocator.alloc(f32, t.numel());
    errdefer allocator.free(out);
    try backend.download(t, f32, out);
    return out;
}

test "gpt forward matches pytorch on the fixture" {
    const allocator = std.testing.allocator;
    var file = try mod.SafeTensors.load(allocator, std.testing.io, mod.build_options.source_root ++ "/testdata/gpt.safetensors");
    defer file.deinit();
    const config = try fixtureConfig(&file);
    try std.testing.expectEqual(try file.metadataInt(usize, "num_params"), config.numParams());
    try std.testing.expectEqual(try file.metadataInt(usize, "flops_per_token"), config.flopsPerToken());

    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var model = try mod.Gpt.init(allocator, &backend, config);
    defer model.deinit();
    try model.weights.load(allocator, &file, "");

    const cos = try downloadAlloc(allocator, &backend, model.cos);
    defer allocator.free(cos);
    const want_cos = try file.readAlloc(allocator, "cos", f32);
    defer allocator.free(want_cos);
    try expectMatches("cos", want_cos, cos, 1e-6);

    const batch = try file.metadataInt(usize, "batch");
    const seq = try file.metadataInt(usize, "seq");
    var acts = try mod.GptActivations.init(allocator, &backend, config, batch, seq);
    defer acts.deinit();
    const idx = try fixtureIds(allocator, &backend, &file, "idx", batch, seq);
    defer backend.free(idx);
    try model.forward(&acts, idx);

    for (acts.layers, 0..) |la, i| {
        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "debug.block.{d}", .{i});
        const want = try file.readAlloc(allocator, name, f32);
        defer allocator.free(want);
        const got = try downloadAlloc(allocator, &backend, la.x_out);
        defer allocator.free(got);
        try expectMatches(name, want, got, 5e-5);
    }
    const want = try file.readAlloc(allocator, "logits", f32);
    defer allocator.free(want);
    const got = try downloadAlloc(allocator, &backend, acts.logits);
    defer allocator.free(got);
    try expectMatches("logits", want, got, 5e-5);
}

test "gpt loss and every parameter gradient match pytorch" {
    const allocator = std.testing.allocator;
    var file = try mod.SafeTensors.load(allocator, std.testing.io, mod.build_options.source_root ++ "/testdata/gpt_grad.safetensors");
    defer file.deinit();
    const config = try fixtureConfig(&file);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var model = try mod.Gpt.init(allocator, &backend, config);
    defer model.deinit();
    try model.weights.load(allocator, &file, "");
    var grads = try mod.GptWeights.init(allocator, &backend, config);
    defer grads.deinit();

    const batch = try file.metadataInt(usize, "batch");
    const seq = try file.metadataInt(usize, "seq");
    var acts = try mod.GptActivations.init(allocator, &backend, config, batch, seq);
    defer acts.deinit();
    var bufs = try mod.GptGradBuffers.init(allocator, &backend, config, batch, seq);
    defer bufs.deinit();
    const idx = try fixtureIds(allocator, &backend, &file, "idx", batch, seq);
    defer backend.free(idx);
    const targets = try fixtureIds(allocator, &backend, &file, "targets", batch, seq);
    defer backend.free(targets);
    const loss = try backend.alloc(.f32, &.{1});
    defer backend.free(loss);

    try model.forward(&acts, idx);
    try model.loss(&acts, targets, loss);
    var got_loss: [1]f32 = undefined;
    try backend.download(loss, f32, &got_loss);
    var want_loss: [1]f32 = undefined;
    try file.read("loss", f32, &want_loss);
    try std.testing.expectApproxEqAbs(want_loss[0], got_loss[0], 1e-5);

    try grads.zero();
    try model.backward(&acts, &bufs, &grads, idx, targets, 1);
    for (grads.params) |p| {
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "grad.{s}", .{p.name});
        const want = try file.readAlloc(allocator, name, f32);
        defer allocator.free(want);
        const got = try downloadAlloc(allocator, &backend, p.tensor);
        defer allocator.free(got);
        // Gradients span orders of magnitude; compare against the tensor's own scale.
        var peak: f32 = 0;
        for (want) |v| peak = @max(peak, @abs(v));
        try expectMatches(name, want, got, 2e-5 * peak + 1e-8);
    }

    // A second backward accumulates: every gradient doubles.
    const before = try downloadAlloc(allocator, &backend, grads.lm_head);
    defer allocator.free(before);
    try model.backward(&acts, &bufs, &grads, idx, targets, 1);
    const after = try downloadAlloc(allocator, &backend, grads.lm_head);
    defer allocator.free(after);
    for (before, after) |b, a| try std.testing.expectApproxEqAbs(2 * b, a, 1e-6 + 1e-5 * @abs(b));
}

test "gpt init weights follows nanochat's scheme" {
    const allocator = std.testing.allocator;
    var backend = try mod.Backend.init(allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 32, .vocab_size = 70, .n_layer = 3, .n_head = 2, .n_kv_head = 1, .n_embd = 32 };
    var model = try mod.Gpt.init(allocator, &backend, config);
    defer model.deinit();
    var rng = mod.Random.init(0);
    try model.initWeights(&rng);

    var resid: [3]f32 = undefined;
    try backend.download(model.weights.resid_lambdas, f32, &resid);
    try std.testing.expectApproxEqAbs(@as(f32, 1.15), resid[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.05), resid[2], 1e-6);
    var proj: [32 * 32]f32 = undefined;
    try backend.download(model.weights.layers[0].c_proj, f32, &proj);
    for (proj) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
