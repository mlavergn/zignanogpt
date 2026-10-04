const std = @import("std");
const log = std.log.scoped(.zignanogpt_inference_buffers);
const mod = @import("module.zig");

/// Scratch for one cached forward step over `[B, T]` new tokens. Unlike the
/// training activations nothing is kept per layer: each layer reuses the same
/// buffers, and only the last position's logits come out.
pub const InferenceBuffers = struct {
    const Self = @This();

    backend: *mod.Backend,
    allocator: std.mem.Allocator,
    owned: std.ArrayList(mod.Tensor) = .empty,
    batch: usize,
    seq: usize,

    emb: mod.Tensor,
    emb_norm: mod.Tensor,
    gate: mod.Tensor,
    x0: mod.Tensor,
    x: mod.Tensor,
    x_in: mod.Tensor,
    xn: mod.Tensor,
    q: mod.Tensor,
    k: mod.Tensor,
    v: mod.Tensor,
    ve: mod.Tensor,
    ve_gate: mod.Tensor,
    y: mod.Tensor,
    tmp: mod.Tensor,
    x_mid: mod.Tensor,
    h: mod.Tensor,
    x_backout: mod.Tensor,
    x_final: mod.Tensor,
    /// Logits of each row's last position: `[B, Vpad]`, then capped `[B, V]`.
    logits_pad: mod.Tensor,
    logits: mod.Tensor,

    /// Allocates scratch for `[batch, seq]` new tokens per step.
    ///
    /// Parameters:
    /// - `allocator`: owns the bookkeeping.
    /// - `backend`: allocates the tensors.
    /// - `config`: the model shape.
    /// - `batch`: rows.
    /// - `seq`: new tokens per row per step (prefill length, or 1 to decode).
    ///
    /// Return: the buffers; allocation errors.
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, config: mod.GptConfig, batch: usize, seq: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = undefined;
        self.backend = backend;
        self.allocator = allocator;
        self.owned = .empty;
        self.batch = batch;
        self.seq = seq;
        errdefer self.freeOwned();
        const b = batch;
        const t = seq;
        const c = config.n_embd;
        const d = config.headDim();
        self.emb = try self.new(&.{ b, t, c });
        self.emb_norm = try self.new(&.{ b, t, c });
        self.gate = try self.new(&.{ b * t, 1 });
        self.x0 = try self.new(&.{ b, t, c });
        self.x = try self.new(&.{ b, t, c });
        self.x_in = try self.new(&.{ b, t, c });
        self.xn = try self.new(&.{ b, t, c });
        self.q = try self.new(&.{ b, t, config.n_head, d });
        self.k = try self.new(&.{ b, t, config.n_kv_head, d });
        self.v = try self.new(&.{ b, t, config.n_kv_head, d });
        self.ve = try self.new(&.{ b * t, config.kvDim() });
        self.ve_gate = try self.new(&.{ b * t, config.n_kv_head });
        self.y = try self.new(&.{ b, t, config.n_head, d });
        self.tmp = try self.new(&.{ b, t, c });
        self.x_mid = try self.new(&.{ b, t, c });
        self.h = try self.new(&.{ b, t, 4 * c });
        self.x_backout = try self.new(&.{ b, t, c });
        self.x_final = try self.new(&.{ b, t, c });
        self.logits_pad = try self.new(&.{ b, config.paddedVocab() });
        self.logits = try self.new(&.{ b, config.vocab_size });
        return self;
    }

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
