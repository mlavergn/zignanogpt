const std = @import("std");
const log = std.log.scoped(.zignanogpt_gpt_activations);
const mod = @import("module.zig");

/// One block's saved forward values (all `[B, T, ...]`).
pub const GptLayerActivations = struct {
    /// Block input after the resid/x0 mix: `[B, T, C]`.
    x_in: mod.Tensor,
    /// `norm(x_in)`, the attention input: `[B, T, C]`.
    xn: mod.Tensor,
    /// Queries / keys after rotary, before QK norm: `[B, T, H|Hkv, D]`.
    q_rot: mod.Tensor,
    k_rot: mod.Tensor,
    /// Queries / keys after QK norm and the 1.2 scale: `[B, T, H|Hkv, D]`.
    q: mod.Tensor,
    k: mod.Tensor,
    /// Values, after the value-embedding mix where the layer has one: `[B, T, Hkv, D]`.
    v: mod.Tensor,
    /// Value embedding lookup and its gate logits, for layers that have one.
    ve: ?mod.Tensor,
    ve_gate: ?mod.Tensor,
    /// Attention output `[B, T, H, D]` and per-query log-sum-exp `[B, H, T]`.
    y: mod.Tensor,
    lse: mod.Tensor,
    /// After the attention residual: `[B, T, C]`.
    x_mid: mod.Tensor,
    /// `norm(x_mid)`, the MLP input: `[B, T, C]`.
    xn2: mod.Tensor,
    /// MLP hidden pre-activation and `relu^2` of it: `[B, T, 4C]`.
    h: mod.Tensor,
    a: mod.Tensor,
    /// Block output: `[B, T, C]`.
    x_out: mod.Tensor,
};

/// Every buffer one forward pass over a `[B, T]` batch writes, kept per layer
/// so the backward pass can read what it needs.
pub const GptActivations = struct {
    const Self = @This();

    backend: *mod.Backend,
    allocator: std.mem.Allocator,
    batch: usize,
    seq: usize,
    /// Every tensor allocated here, freed in `deinit`.
    owned: std.ArrayList(mod.Tensor),

    /// Token embeddings `[B, T, C]`, then normalized.
    emb: mod.Tensor,
    emb_norm: mod.Tensor,
    /// Smear gate logits `[B * T, 1]`.
    smear_gate: mod.Tensor,
    /// The smeared, normalized embedding every layer mixes back in: `[B, T, C]`.
    x0: mod.Tensor,
    layers: []GptLayerActivations,
    /// After the backout subtraction, then normalized: `[B, T, C]`.
    x_final: mod.Tensor,
    xn_final: mod.Tensor,
    /// `lm_head` output over the padded vocab `[B * T, Vpad]`, then cropped and capped `[B, T, V]`.
    logits_pad: mod.Tensor,
    logits: mod.Tensor,
    /// Scratch for projection outputs: `[B, T, C]`.
    tmp: mod.Tensor,

    /// Allocates the buffers for a `[batch, seq]` forward pass.
    ///
    /// Parameters:
    /// - `allocator`: owns the bookkeeping.
    /// - `backend`: allocates the tensors.
    /// - `config`: the model shape.
    /// - `batch`: rows per batch (B).
    /// - `seq`: tokens per row (T), at most `config.sequence_len`.
    ///
    /// Return: the activations; allocation errors, `error.SequenceTooLong`.
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, config: mod.GptConfig, batch: usize, seq: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (seq > config.sequence_len) {
            log.warn("sequence {d} exceeds the model's {d}", .{ seq, config.sequence_len });
            return error.SequenceTooLong;
        }
        var self: Self = undefined;
        self.backend = backend;
        self.allocator = allocator;
        self.batch = batch;
        self.seq = seq;
        self.owned = .empty;
        errdefer self.freeOwned();

        const b = batch;
        const t = seq;
        const c = config.n_embd;
        const d = config.headDim();
        self.emb = try self.new(&.{ b, t, c });
        self.emb_norm = try self.new(&.{ b, t, c });
        self.smear_gate = try self.new(&.{ b * t, 1 });
        self.x0 = try self.new(&.{ b, t, c });
        self.x_final = try self.new(&.{ b, t, c });
        self.xn_final = try self.new(&.{ b, t, c });
        self.logits_pad = try self.new(&.{ b * t, config.paddedVocab() });
        self.logits = try self.new(&.{ b, t, config.vocab_size });
        self.tmp = try self.new(&.{ b, t, c });

        self.layers = try allocator.alloc(GptLayerActivations, config.n_layer);
        errdefer allocator.free(self.layers);
        for (self.layers, 0..) |*layer, i| {
            const ve = config.hasValueEmbed(i);
            layer.* = .{
                .x_in = try self.new(&.{ b, t, c }),
                .xn = try self.new(&.{ b, t, c }),
                .q_rot = try self.new(&.{ b, t, config.n_head, d }),
                .k_rot = try self.new(&.{ b, t, config.n_kv_head, d }),
                .q = try self.new(&.{ b, t, config.n_head, d }),
                .k = try self.new(&.{ b, t, config.n_kv_head, d }),
                .v = try self.new(&.{ b, t, config.n_kv_head, d }),
                .ve = if (ve) try self.new(&.{ b * t, config.kvDim() }) else null,
                .ve_gate = if (ve) try self.new(&.{ b * t, config.n_kv_head }) else null,
                .y = try self.new(&.{ b, t, config.n_head, d }),
                .lse = try self.new(&.{ b, config.n_head, t }),
                .x_mid = try self.new(&.{ b, t, c }),
                .xn2 = try self.new(&.{ b, t, c }),
                .h = try self.new(&.{ b, t, 4 * c }),
                .a = try self.new(&.{ b, t, 4 * c }),
                .x_out = try self.new(&.{ b, t, c }),
            };
        }
        return self;
    }

    /// Frees every buffer.
    ///
    /// Parameters:
    /// - `self`: the activations.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.layers);
        self.freeOwned();
    }

    /// Allocates one f32 tensor and records it for `deinit`.
    fn new(self: *Self, dims: []const usize) !mod.Tensor {
        const t = try self.backend.alloc(.f32, dims);
        errdefer self.backend.free(t);
        try self.owned.append(self.allocator, t);
        return t;
    }

    fn freeOwned(self: *Self) void {
        for (self.owned.items) |t| self.backend.free(t);
        self.owned.deinit(self.allocator);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "gpt activations size every buffer from the config" {
    var backend = try mod.Backend.init(std.testing.allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 16, .vocab_size = 100, .n_layer = 3, .n_head = 4, .n_kv_head = 2, .n_embd = 32 };
    var acts = try mod.GptActivations.init(std.testing.allocator, &backend, config, 2, 8);
    defer acts.deinit();
    try std.testing.expectEqual(@as(usize, 2 * 8 * 128), acts.logits_pad.numel());
    try std.testing.expect(acts.layers[0].ve != null and acts.layers[1].ve == null and acts.layers[2].ve != null);
    try std.testing.expectError(error.SequenceTooLong, mod.GptActivations.init(std.testing.allocator, &backend, config, 1, 17));
}
