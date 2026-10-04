const std = @import("std");
const log = std.log.scoped(.zignanogpt_gpt_grad_buffers);
const mod = @import("module.zig");

/// Activation gradients for one backward pass over a `[B, T]` batch.
///
/// Unlike the forward activations these are not kept per layer: the backward
/// pass walks the layers in reverse and reuses one set.
pub const GptGradBuffers = struct {
    const Self = @This();

    backend: *mod.Backend,
    allocator: std.mem.Allocator,
    owned: std.ArrayList(mod.Tensor),

    /// Loss gradient over the padded logits `[B * T, Vpad]`.
    dlogits_pad: mod.Tensor,
    /// Gradient of the residual stream at the current point `[B, T, C]`.
    dx: mod.Tensor,
    /// Accumulated gradient of `x0` `[B, T, C]`.
    dx0: mod.Tensor,
    /// Backout's gradient into the mid-depth block output `[B, T, C]`.
    dmid: mod.Tensor,
    /// Gradient of a normalized activation (`norm(x)` inputs) `[B, T, C]`.
    dxn: mod.Tensor,
    /// Attention output, query, key, value gradients.
    dy: mod.Tensor,
    dq: mod.Tensor,
    dk: mod.Tensor,
    dv: mod.Tensor,
    /// MLP hidden gradients `[B, T, 4C]`.
    da: mod.Tensor,
    dh: mod.Tensor,
    /// Value-embedding and gate gradients.
    dve: mod.Tensor,
    dve_gate: mod.Tensor,
    /// Smear gate, normalized and raw embedding gradients.
    dsmear_gate: mod.Tensor,
    demb_norm: mod.Tensor,
    demb: mod.Tensor,

    /// Allocates the buffers for a `[batch, seq]` backward pass.
    ///
    /// Parameters:
    /// - `allocator`: owns the bookkeeping.
    /// - `backend`: allocates the tensors.
    /// - `config`: the model shape.
    /// - `batch`: rows per batch (B).
    /// - `seq`: tokens per row (T).
    ///
    /// Return: the buffers; allocation errors.
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, config: mod.GptConfig, batch: usize, seq: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = undefined;
        self.backend = backend;
        self.allocator = allocator;
        self.owned = .empty;
        errdefer self.freeOwned();
        const b = batch;
        const t = seq;
        const c = config.n_embd;
        const d = config.headDim();
        self.dlogits_pad = try self.new(&.{ b * t, config.paddedVocab() });
        self.dx = try self.new(&.{ b, t, c });
        self.dx0 = try self.new(&.{ b, t, c });
        self.dmid = try self.new(&.{ b, t, c });
        self.dxn = try self.new(&.{ b, t, c });
        self.dy = try self.new(&.{ b, t, config.n_head, d });
        self.dq = try self.new(&.{ b, t, config.n_head, d });
        self.dk = try self.new(&.{ b, t, config.n_kv_head, d });
        self.dv = try self.new(&.{ b, t, config.n_kv_head, d });
        self.da = try self.new(&.{ b, t, 4 * c });
        self.dh = try self.new(&.{ b, t, 4 * c });
        self.dve = try self.new(&.{ b * t, config.kvDim() });
        self.dve_gate = try self.new(&.{ b * t, config.n_kv_head });
        self.dsmear_gate = try self.new(&.{ b * t, 1 });
        self.demb_norm = try self.new(&.{ b, t, c });
        self.demb = try self.new(&.{ b, t, c });
        return self;
    }

    /// Frees every buffer.
    ///
    /// Parameters:
    /// - `self`: the buffers.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.freeOwned();
    }

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

test "gpt grad buffers match the activation shapes" {
    var backend = try mod.Backend.init(std.testing.allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 16, .vocab_size = 100, .n_layer = 2, .n_head = 4, .n_kv_head = 2, .n_embd = 32 };
    var bufs = try mod.GptGradBuffers.init(std.testing.allocator, &backend, config, 2, 8);
    defer bufs.deinit();
    try std.testing.expectEqual(@as(usize, 16 * 128), bufs.dlogits_pad.numel());
    try std.testing.expectEqual(@as(usize, 2 * 8 * 2 * 8), bufs.dk.numel());
}
