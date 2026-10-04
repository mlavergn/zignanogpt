const std = @import("std");
const log = std.log.scoped(.zignanogpt_kv_cache);
const mod = @import("module.zig");

/// One layer's cached keys and values: `[B, Tmax, Hkv, D]` each.
pub const KvLayer = struct {
    k: mod.Tensor,
    v: mod.Tensor,
};

/// nanochat's `KVCache`: per-layer keys and values for every position fed so
/// far, the shared position, and the previous token's normalized embedding
/// (smear needs it when decoding one token at a time).
pub const KvCache = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    backend: *mod.Backend,
    batch: usize,
    max_seq: usize,
    layers: []KvLayer,
    /// Positions filled (the same for every row).
    pos: usize = 0,
    /// The last token's normalized, pre-smear embedding `[B, 1, C]`.
    prev: mod.Tensor,
    has_prev: bool = false,

    /// Allocates an empty cache.
    ///
    /// Parameters:
    /// - `allocator`: owns the layer list.
    /// - `backend`: allocates the tensors.
    /// - `config`: the model shape.
    /// - `batch`: rows (samples).
    /// - `max_seq`: positions per row; at most `rotarySeqLen()`.
    ///
    /// Return: the cache; allocation errors.
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, config: mod.GptConfig, batch: usize, max_seq: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (max_seq > config.rotarySeqLen()) return error.SequenceTooLong;
        const layers = try allocator.alloc(KvLayer, config.n_layer);
        errdefer allocator.free(layers);
        var made: usize = 0;
        errdefer for (layers[0..made]) |l| {
            backend.free(l.k);
            backend.free(l.v);
        };
        const d = config.headDim();
        for (layers) |*l| {
            l.k = try backend.alloc(.f32, &.{ batch, max_seq, config.n_kv_head, d });
            l.v = backend.alloc(.f32, &.{ batch, max_seq, config.n_kv_head, d }) catch |err| {
                backend.free(l.k);
                return err;
            };
            made += 1;
        }
        const prev = try backend.alloc(.f32, &.{ batch, 1, config.n_embd });
        return Self{ .allocator = allocator, .backend = backend, .batch = batch, .max_seq = max_seq, .layers = layers, .prev = prev };
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.backend.free(self.prev);
        for (self.layers) |l| {
            self.backend.free(l.k);
            self.backend.free(l.v);
        }
        self.allocator.free(self.layers);
    }

    /// Empties the cache (nanochat's `reset`).
    pub fn reset(self: *Self) void {
        self.pos = 0;
        self.has_prev = false;
    }

    /// The positions still free.
    pub fn remaining(self: *const Self) usize {
        return self.max_seq - self.pos;
    }

    /// Copies a batch-1 cache into every row (nanochat's `prefill`): one
    /// prompt prefill shared by all samples.
    ///
    /// Parameters:
    /// - `self`: an empty cache with room for `other.pos` positions.
    /// - `other`: a filled batch-1 cache of the same model.
    ///
    /// Return: nothing; `error.CacheMismatch`.
    pub fn copyFrom(self: *Self, other: *const Self) !void {
        if (self.pos != 0 or other.batch != 1 or other.layers.len != self.layers.len or other.pos > self.max_seq) return error.CacheMismatch;
        const n = other.pos;
        for (self.layers, other.layers) |dst, src| {
            inline for (.{ "k", "v" }) |which| {
                const s = try rowPositions(@field(src, which), 0, 0, n);
                for (0..self.batch) |b| try self.backend.copy(try rowPositions(@field(dst, which), b, 0, n), s);
            }
        }
        if (other.has_prev) {
            for (0..self.batch) |b| try self.backend.copy(try self.prev.rows(b, 1), other.prev);
        }
        self.has_prev = other.has_prev;
        self.pos = n;
    }

    /// Writes new keys and values `[B, T, Hkv, D]` at positions `pos .. pos + T`.
    ///
    /// Parameters:
    /// - `self`: the cache.
    /// - `layer`: the layer index.
    /// - `k`: new keys.
    /// - `v`: new values.
    ///
    /// Return: nothing; `error.CacheFull`, shape errors.
    pub fn write(self: *Self, layer: usize, k: mod.Tensor, v: mod.Tensor) !void {
        const t = k.shape.dims[1];
        if (self.pos + t > self.max_seq) {
            log.warn("KV cache full: {d} + {d} > {d}", .{ self.pos, t, self.max_seq });
            return error.CacheFull;
        }
        const l = self.layers[layer];
        for (0..self.batch) |b| {
            try self.backend.copy(try rowPositions(l.k, b, self.pos, t), try (try k.rows(b, 1)).reshape(&.{ t, k.numel() / (k.shape.dims[0] * t) }));
            try self.backend.copy(try rowPositions(l.v, b, self.pos, t), try (try v.rows(b, 1)).reshape(&.{ t, v.numel() / (v.shape.dims[0] * t) }));
        }
    }

    /// Positions `start .. start + count` of row `b`, as `[count, Hkv * D]`.
    fn rowPositions(t: mod.Tensor, b: usize, start: usize, count: usize) !mod.Tensor {
        const row = try t.rows(b, 1);
        const flat = try row.reshape(&.{ t.shape.dims[1], t.shape.dims[2] * t.shape.dims[3] });
        return flat.rows(start, count);
    }
};
