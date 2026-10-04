const std = @import("std");
const log = std.log.scoped(.zignanogpt_gpt_weights);
const mod = @import("module.zig");

/// One transformer block's weights (PyTorch `Linear` layout, `[out, in]`).
pub const GptLayer = struct {
    c_q: mod.Tensor,
    c_k: mod.Tensor,
    c_v: mod.Tensor,
    c_proj: mod.Tensor,
    c_fc: mod.Tensor,
    mlp_proj: mod.Tensor,
    /// `[n_kv_head, 12]` gate and `[Vpad, kv_dim]` table, on value-embedding layers.
    ve_gate: ?mod.Tensor,
    value_embed: ?mod.Tensor,
};

/// A parameter under its PyTorch name.
pub const GptParam = struct {
    name: []const u8,
    tensor: mod.Tensor,
};

/// One full set of model-shaped tensors: the weights, or their gradients.
///
/// Every tensor is listed in `params` under its PyTorch `named_parameters`
/// name, in PyTorch's registration order (`transformer.wte`, the blocks,
/// `lm_head`, the scalars, `value_embeds`); the named fields alias the same tensors.
pub const GptWeights = struct {
    const Self = @This();

    backend: *mod.Backend,
    /// Owns the names and the arrays.
    arena: std.heap.ArenaAllocator,
    params: []GptParam,

    wte: mod.Tensor,
    lm_head: mod.Tensor,
    resid_lambdas: mod.Tensor,
    x0_lambdas: mod.Tensor,
    smear_gate: mod.Tensor,
    smear_lambda: mod.Tensor,
    backout_lambda: mod.Tensor,
    layers: []GptLayer,

    /// Allocates a zeroed set shaped by `config`.
    ///
    /// Parameters:
    /// - `allocator`: backs the arena.
    /// - `backend`: allocates the tensors.
    /// - `config`: the model shape.
    ///
    /// Return: the set; allocation errors.
    pub fn init(allocator: std.mem.Allocator, backend: *mod.Backend, config: mod.GptConfig) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = undefined;
        self.backend = backend;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        var params: std.ArrayList(GptParam) = .empty;
        errdefer for (params.items) |p| backend.free(p.tensor);

        const c = config.n_embd;
        const kv = config.kvDim();
        const vpad = config.paddedVocab();
        self.layers = try arena.alloc(GptLayer, config.n_layer);
        self.wte = try add(arena, backend, &params, "transformer.wte.weight", &.{ vpad, c });
        for (self.layers, 0..) |*layer, i| {
            layer.c_q = try add(arena, backend, &params, try name(arena, i, "attn.c_q.weight"), &.{ c, c });
            layer.c_k = try add(arena, backend, &params, try name(arena, i, "attn.c_k.weight"), &.{ kv, c });
            layer.c_v = try add(arena, backend, &params, try name(arena, i, "attn.c_v.weight"), &.{ kv, c });
            layer.c_proj = try add(arena, backend, &params, try name(arena, i, "attn.c_proj.weight"), &.{ c, c });
            layer.ve_gate = if (config.hasValueEmbed(i))
                try add(arena, backend, &params, try name(arena, i, "attn.ve_gate.weight"), &.{ config.n_kv_head, mod.GptConfig.ve_gate_channels })
            else
                null;
            layer.c_fc = try add(arena, backend, &params, try name(arena, i, "mlp.c_fc.weight"), &.{ 4 * c, c });
            layer.mlp_proj = try add(arena, backend, &params, try name(arena, i, "mlp.c_proj.weight"), &.{ c, 4 * c });
        }
        self.lm_head = try add(arena, backend, &params, "lm_head.weight", &.{ vpad, c });
        self.resid_lambdas = try add(arena, backend, &params, "resid_lambdas", &.{config.n_layer});
        self.x0_lambdas = try add(arena, backend, &params, "x0_lambdas", &.{config.n_layer});
        self.smear_gate = try add(arena, backend, &params, "smear_gate.weight", &.{ 1, mod.GptConfig.smear_channels });
        self.smear_lambda = try add(arena, backend, &params, "smear_lambda", &.{1});
        self.backout_lambda = try add(arena, backend, &params, "backout_lambda", &.{1});
        for (self.layers, 0..) |*layer, i| {
            layer.value_embed = if (config.hasValueEmbed(i))
                try add(arena, backend, &params, try std.fmt.allocPrint(arena, "value_embeds.{d}.weight", .{i}), &.{ vpad, kv })
            else
                null;
        }
        self.params = try params.toOwnedSlice(arena);
        return self;
    }

    /// Frees every tensor.
    ///
    /// Parameters:
    /// - `self`: the set.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (self.params) |p| self.backend.free(p.tensor);
        self.arena.deinit();
    }

    /// Sets every tensor to zero (e.g. gradients before accumulation).
    ///
    /// Parameters:
    /// - `self`: the set.
    ///
    /// Return: nothing; backend errors.
    pub fn zero(self: *Self) !void {
        for (self.params) |p| try self.backend.fill(p.tensor, 0);
    }

    /// Finds a tensor by its PyTorch name.
    ///
    /// Parameters:
    /// - `self`: the set.
    /// - `param_name`: e.g. `transformer.h.0.attn.c_q.weight`.
    ///
    /// Return: the tensor, or null.
    pub fn get(self: *const Self, param_name: []const u8) ?mod.Tensor {
        for (self.params) |p| {
            if (std.mem.eql(u8, p.name, param_name)) return p.tensor;
        }
        return null;
    }

    /// Total element count.
    ///
    /// Parameters:
    /// - `self`: the set.
    ///
    /// Return: the count.
    pub fn numel(self: *const Self) usize {
        var total: usize = 0;
        for (self.params) |p| total += p.tensor.numel();
        return total;
    }

    /// Loads every tensor from a file holding them under `prefix ++ name`.
    ///
    /// Parameters:
    /// - `self`: the set.
    /// - `allocator`: for the host staging copy.
    /// - `file`: f32 tensors with matching shapes.
    /// - `prefix`: prepended to each name (`""`, or e.g. `"grad."`).
    ///
    /// Return: nothing; `error.MissingTensor` or `error.ShapeMismatch`.
    pub fn load(self: *Self, allocator: std.mem.Allocator, file: *const mod.SafeTensors, prefix: []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (self.params) |p| {
            const key = try std.mem.concat(allocator, u8, &.{ prefix, p.name });
            defer allocator.free(key);
            const entry = try file.get(key);
            if (!entry.shape.eql(p.tensor.shape)) {
                log.err("{s}: file has {f}, model expects {f}", .{ key, entry.shape, p.tensor.shape });
                return error.ShapeMismatch;
            }
            const host = try file.readAlloc(allocator, key, f32);
            defer allocator.free(host);
            try self.backend.upload(p.tensor, f32, host);
        }
    }

    /// Allocates a zeroed parameter and records it under `param_name`.
    fn add(arena: std.mem.Allocator, backend: *mod.Backend, params: *std.ArrayList(GptParam), param_name: []const u8, dims: []const usize) !mod.Tensor {
        const t = try backend.alloc(.f32, dims);
        errdefer backend.free(t);
        try params.append(arena, .{ .name = param_name, .tensor = t });
        return t;
    }

    /// `transformer.h.<layer>.<suffix>`.
    fn name(arena: std.mem.Allocator, layer: usize, suffix: []const u8) ![]const u8 {
        return std.fmt.allocPrint(arena, "transformer.h.{d}.{s}", .{ layer, suffix });
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "gpt weights follow pytorch names, order and counts" {
    var backend = try mod.Backend.init(std.testing.allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();
    const config = mod.GptConfig{ .sequence_len = 32, .vocab_size = 70, .n_layer = 3, .n_head = 2, .n_kv_head = 1, .n_embd = 32 };
    var weights = try mod.GptWeights.init(std.testing.allocator, &backend, config);
    defer weights.deinit();
    try std.testing.expectEqualStrings("transformer.wte.weight", weights.params[0].name);
    try std.testing.expectEqualStrings("transformer.h.0.attn.c_q.weight", weights.params[1].name);
    try std.testing.expectEqualStrings("value_embeds.2.weight", weights.params[weights.params.len - 1].name);
    try std.testing.expectEqual(config.numParams(), weights.numel());
    try std.testing.expect(weights.get("lm_head.weight") != null);
    try std.testing.expect(weights.get("value_embeds.1.weight") == null);
}
