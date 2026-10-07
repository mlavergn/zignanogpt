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

    /// Reads nanochat's `model_config` JSON (as saved in a checkpoint's meta).
    /// A missing `window_pattern` means `L`, as `checkpoint_manager.py` patches
    /// old checkpoints.
    ///
    /// Parameters:
    /// - `arena`: copies the window pattern (it must outlive the config).
    /// - `value`: the `model_config` object.
    ///
    /// Return: the config; `error.InvalidConfig` on a missing or bad field.
    pub fn fromJson(arena: std.mem.Allocator, value: std.json.Value) !Self {
        if (value != .object) return error.InvalidConfig;
        const field = struct {
            fn int(v: std.json.Value, key: []const u8) !usize {
                const x = v.object.get(key) orelse {
                    log.warn("model_config lacks {s}", .{key});
                    return error.InvalidConfig;
                };
                if (x != .integer or x.integer <= 0) return error.InvalidConfig;
                return @intCast(x.integer);
            }
        };
        const pattern = if (value.object.get("window_pattern")) |p| (if (p == .string) p.string else return error.InvalidConfig) else "L";
        const self = Self{
            .sequence_len = try field.int(value, "sequence_len"),
            .vocab_size = try field.int(value, "vocab_size"),
            .n_layer = try field.int(value, "n_layer"),
            .n_head = try field.int(value, "n_head"),
            .n_kv_head = try field.int(value, "n_kv_head"),
            .n_embd = try field.int(value, "n_embd"),
            .window_pattern = try arena.dupe(u8, pattern),
        };
        try self.validate();
        return self;
    }

    /// Writes the config as nanochat's `model_config` JSON object.
    ///
    /// Parameters:
    /// - `self`: the config.
    /// - `jw`: a JSON writer positioned where the object goes.
    ///
    /// Return: nothing; write errors.
    pub fn jsonStringify(self: Self, jw: anytype) !void {
        try jw.beginObject();
        inline for (.{ "sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "window_pattern" }) |name| {
            try jw.objectField(name);
            try jw.write(@field(self, name));
        }
        try jw.endObject();
    }

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

    /// The parameters nanochat's scaling laws count: every transformer block
    /// parameter plus `lm_head` (`num_scaling_params`' `transformer_matrices + lm_head`).
    pub fn numScalingParams(self: Self) usize {
        return self.numMatmulParams() - smear_channels;
    }

    /// Every parameter (`sum(p.numel())`).
    pub fn numParams(self: Self) usize {
        var total = self.numMatmulParams() + self.paddedVocab() * self.n_embd; // + wte
        for (0..self.n_layer) |i| {
            if (self.hasValueEmbed(i)) total += self.paddedVocab() * self.kvDim();
        }
        return total + 2 * self.n_layer + 2; // resid/x0 lambdas, smear_lambda, backout_lambda
    }

    /// Forward FLOPs to decode one token at a context length
    /// (`estimate_decode_flops`): 2 per matmul weight, plus attention over
    /// `min(context, window)` keys per layer.
    ///
    /// Parameters:
    /// - `self`: the config.
    /// - `context`: the tokens already in the cache.
    ///
    /// Return: the FLOPs.
    pub fn decodeFlops(self: Self, context: usize) usize {
        var attention: usize = 0;
        for (0..self.n_layer) |i| attention += 4 * self.n_head * self.headDim() * @min(context, self.windowSize(i));
        return 2 * self.numMatmulParams() + attention;
    }

    /// Forward FLOPs to prefill a prompt (`estimate_prefill_flops`): causal, so
    /// token t attends `min(t, window)` keys.
    ///
    /// Parameters:
    /// - `self`: the config.
    /// - `tokens`: the prompt length.
    ///
    /// Return: the FLOPs.
    pub fn prefillFlops(self: Self, tokens: usize) usize {
        var attention: usize = 0;
        for (0..self.n_layer) |i| {
            const w = @min(self.windowSize(i), tokens);
            const attended = w * (w + 1) / 2 + (tokens - w) * w; // ramp up to w, then flat
            attention += 4 * self.n_head * self.headDim() * attended;
        }
        return 2 * self.numMatmulParams() * tokens + attention;
    }

    /// Bytes one token of KV cache stores per row, all layers
    /// (`kv_bytes_per_token`; this port's cache is f32).
    pub fn kvBytesPerToken(self: Self) usize {
        return self.n_layer * 2 * self.kvDim() * @sizeOf(f32);
    }

    /// Bytes of KV cache one decode step reads per row at a context length
    /// (`kv_read_bytes`): sliding-window layers read only their window.
    ///
    /// Parameters:
    /// - `self`: the config.
    /// - `context`: the tokens in the cache.
    ///
    /// Return: the bytes.
    pub fn kvReadBytes(self: Self, context: usize) usize {
        var total: usize = 0;
        for (0..self.n_layer) |i| total += 2 * self.kvDim() * @sizeOf(f32) * @min(context, self.windowSize(i));
        return total;
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
    try std.testing.expectEqual(@as(usize, 241746 - 320 * 64 - 2 * 320 * 32 - 2 * 4 - 2 - 24), config.numScalingParams());
    try std.testing.expect(config.hasValueEmbed(1) and config.hasValueEmbed(3) and !config.hasValueEmbed(0));
}

test "gpt config inference costs match nanochat's estimators" {
    // From nanochat's GPT on the meta device (f32 KV, as this port's cache).
    const small = mod.GptConfig{ .sequence_len = 256, .vocab_size = 1000, .n_layer = 4, .n_head = 4, .n_kv_head = 2, .n_embd = 128, .window_pattern = "SSSL" };
    try std.testing.expectEqual(@as(usize, 852040), small.numMatmulParams());
    try std.testing.expectEqual(@as(usize, 2003088), small.decodeFlops(200));
    try std.testing.expectEqual(@as(usize, 377944192), small.prefillFlops(200));
    try std.testing.expectEqual(@as(usize, 2048), small.kvBytesPerToken());
    try std.testing.expectEqual(@as(usize, 299008), small.kvReadBytes(200));
    const d20 = mod.GptConfig{ .sequence_len = 2048, .vocab_size = 32768, .n_layer = 20, .n_head = 10, .n_kv_head = 10, .n_embd = 1280, .window_pattern = "SSSL" };
    try std.testing.expectEqual(@as(usize, 435160264), d20.numMatmulParams());
    try std.testing.expectEqual(@as(usize, 890800528), d20.decodeFlops(200));
    try std.testing.expectEqual(@as(usize, 176122345600), d20.prefillFlops(200));
    try std.testing.expectEqual(@as(usize, 204800), d20.kvBytesPerToken());
    try std.testing.expectEqual(@as(usize, 40960000), d20.kvReadBytes(200));
}

test "gpt config round-trips nanochat's model_config json" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"sequence_len": 64, "vocab_size": 1033, "n_layer": 1, "n_head": 4, "n_kv_head": 2, "n_embd": 64}
    , .{});
    defer parsed.deinit();
    const config = try mod.GptConfig.fromJson(arena.allocator(), parsed.value);
    try std.testing.expectEqualStrings("L", config.window_pattern); // legacy default
    const text = try std.json.Stringify.valueAlloc(allocator, config, .{});
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"window_pattern\":\"L\"") != null);
}

test "gpt config rejects inconsistent shapes" {
    try std.testing.expectError(error.InvalidConfig, (mod.GptConfig{ .n_embd = 100, .n_head = 3 }).validate());
    try std.testing.expectError(error.InvalidConfig, (mod.GptConfig{ .window_pattern = "SX" }).validate());
}
