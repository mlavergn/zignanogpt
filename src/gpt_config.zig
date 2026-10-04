const std = @import("std");
const log = std.log.scoped(.zignanogpt_gpt_config);
const mod = @import("module.zig");

/// The model's shape, as nanochat's `GPTConfig`, plus the quantities derived
/// from it (padded vocab, windows, parameter and FLOP counts).
pub const GptConfig = struct {
    const Self = @This();

    /// Vocab rows are padded to a multiple of this (`GPT.__init__`'s `pad_vocab_size_to`).
    pub const pad_vocab_to = 64;
    /// `F.rms_norm` with no eps uses the dtype's machine epsilon.
    pub const rms_eps: f32 = std.math.floatEps(f32);
    /// Logits are squashed into `[-softcap, softcap]`.
    pub const softcap: f32 = 15;
    pub const rope_base: f64 = 100000;
    /// The rotary table covers this many times `sequence_len`.
    pub const rotary_factor = 10;
    /// Channels the value-embedding gate reads.
    pub const ve_gate_channels = 12;
    /// Channels the smear gate reads.
    pub const smear_channels = 24;
    /// Queries and keys are scaled by this after QK norm.
    pub const qk_scale: f32 = 1.2;

    sequence_len: usize = 2048,
    vocab_size: usize = 32768,
    n_layer: usize = 12,
    n_head: usize = 6,
    n_kv_head: usize = 6,
    n_embd: usize = 768,
    /// Window per layer, tiled: `L` full context, `S` a quarter; the last layer is always `L`.
    window_pattern: []const u8 = "SSSL",

    /// Checks the dimensions fit together.
    ///
    /// Parameters:
    /// - `self`: the config.
    ///
    /// Return: nothing; `error.InvalidConfig` with the reason logged.
    pub fn validate(self: Self) !void {
        const reason: ?[]const u8 = if (self.n_layer == 0 or self.n_head == 0 or self.n_kv_head == 0 or self.n_embd == 0 or self.vocab_size == 0)
            "dimensions must be positive"
        else if (self.n_embd % self.n_head != 0)
            "n_embd must be a multiple of n_head"
        else if (self.n_head % self.n_kv_head != 0)
            "n_head must be a multiple of n_kv_head"
        else if ((self.n_embd / self.n_head) % 2 != 0)
            "head dim must be even (rotary halves)"
        else if (self.n_embd < smear_channels or self.n_embd < ve_gate_channels)
            "n_embd must cover the gate channels"
        else if (self.window_pattern.len == 0)
            "window_pattern is empty"
        else
            null;
        if (reason) |text| {
            log.warn("invalid config: {s}", .{text});
            return error.InvalidConfig;
        }
        for (self.window_pattern) |c| {
            if (std.ascii.toUpper(c) != 'S' and std.ascii.toUpper(c) != 'L') {
                log.warn("invalid window_pattern '{s}': use only S and L", .{self.window_pattern});
                return error.InvalidConfig;
            }
        }
    }

    pub fn headDim(self: Self) usize {
        return self.n_embd / self.n_head;
    }

    /// Width of the keys and values: `n_kv_head * head_dim`.
    pub fn kvDim(self: Self) usize {
        return self.n_kv_head * self.headDim();
    }

    pub fn paddedVocab(self: Self) usize {
        return (self.vocab_size + pad_vocab_to - 1) / pad_vocab_to * pad_vocab_to;
    }

    pub fn rotarySeqLen(self: Self) usize {
        return self.sequence_len * rotary_factor;
    }

    /// The layer whose output the backout subtracts (`n_layer // 2`).
    pub fn backoutLayer(self: Self) usize {
        return self.n_layer / 2;
    }

    /// Whether `layer` has a value embedding: alternating, the last always included.
    pub fn hasValueEmbed(self: Self, layer: usize) bool {
        return layer % 2 == (self.n_layer - 1) % 2;
    }

    /// The attention window of `layer` (`_compute_window_sizes`).
    ///
    /// Parameters:
    /// - `self`: the config.
    /// - `layer`: the layer index.
    ///
    /// Return: keys visible before each query.
    pub fn windowSize(self: Self, layer: usize) usize {
        const long = self.sequence_len;
        if (layer == self.n_layer - 1) return long;
        const c = std.ascii.toUpper(self.window_pattern[layer % self.window_pattern.len]);
        if (c == 'L') return long;
        // -(-long // 4 // 128) * 128: a quarter of the context, rounded up to 128
        const quarter = (long + 3) / 4;
        return (quarter + 127) / 128 * 128;
    }

    /// Weights that multiply the token stream (every `Linear`), 2 FLOPs each forward.
    pub fn numMatmulParams(self: Self) usize {
        const c = self.n_embd;
        var total: usize = self.paddedVocab() * c + smear_channels; // lm_head, smear_gate
        for (0..self.n_layer) |i| {
            total += c * c + 2 * c * self.kvDim() + c * c + 2 * 4 * c * c;
            if (self.hasValueEmbed(i)) total += ve_gate_channels * self.n_kv_head;
        }
        return total;
    }

    /// Every parameter (`sum(p.numel())`).
    pub fn numParams(self: Self) usize {
        var total = self.numMatmulParams() + self.paddedVocab() * self.n_embd; // + wte
        for (0..self.n_layer) |i| {
            if (self.hasValueEmbed(i)) total += self.paddedVocab() * self.kvDim();
        }
        return total + 2 * self.n_layer + 2; // resid/x0 lambdas, smear_lambda, backout_lambda
    }

    /// Training FLOPs per token, forward + backward (`estimate_flops`).
    pub fn flopsPerToken(self: Self) usize {
        var attention: usize = 0;
        for (0..self.n_layer) |i| attention += 12 * self.n_head * self.headDim() * @min(self.windowSize(i), self.sequence_len);
        return 6 * self.numMatmulParams() + attention;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "gpt config derives windows like nanochat" {
    const config = mod.GptConfig{ .sequence_len = 2048, .n_layer = 5, .window_pattern = "sl" };
    try std.testing.expectEqual(@as(usize, 512), config.windowSize(0));
    try std.testing.expectEqual(@as(usize, 2048), config.windowSize(1));
    try std.testing.expectEqual(@as(usize, 512), config.windowSize(2));
    try std.testing.expectEqual(@as(usize, 2048), config.windowSize(4)); // last layer is always L
    try std.testing.expectEqual(@as(usize, 256), (mod.GptConfig{ .sequence_len = 1000 }).windowSize(0));
}

test "gpt config counts match the python fixture" {
    const config = mod.GptConfig{ .sequence_len = 512, .vocab_size = 300, .n_layer = 4, .n_head = 4, .n_kv_head = 2, .n_embd = 64 };
    try config.validate();
    try std.testing.expectEqual(@as(usize, 320), config.paddedVocab());
    try std.testing.expectEqual(@as(usize, 241746), config.numParams());
    try std.testing.expectEqual(@as(usize, 1892784), config.flopsPerToken());
    try std.testing.expect(config.hasValueEmbed(1) and config.hasValueEmbed(3) and !config.hasValueEmbed(0));
}

test "gpt config rejects inconsistent shapes" {
    try std.testing.expectError(error.InvalidConfig, (mod.GptConfig{ .n_embd = 100, .n_head = 3 }).validate());
    try std.testing.expectError(error.InvalidConfig, (mod.GptConfig{ .window_pattern = "SX" }).validate());
}
