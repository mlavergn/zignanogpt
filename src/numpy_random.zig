const std = @import("std");
const log = std.log.scoped(.zignanogpt_numpy_random);

/// numpy's `np.random.default_rng(seed)`: a PCG64 (XSL-RR 128/64) seeded
/// through `SeedSequence`, bit for bit. nanochat's tasks order their rows with
/// `default_rng(42).permutation(n)` (`HubDataset.shuffle`), so the port needs
/// the identical stream.
pub const NumpyRandom = struct {
    const Self = @This();

    const multiplier: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

    state: u128,
    inc: u128,
    /// numpy hands out 32-bit draws as halves of one 64-bit draw.
    has_u32: bool = false,
    next_u32: u32 = 0,

    /// Seeds like `np.random.default_rng(seed)`.
    ///
    /// Parameters:
    /// - `seed`: a non-negative integer seed.
    ///
    /// Return: the generator.
    pub fn init(seed: u64) Self {
        var words: [8]u32 = undefined;
        seedSequence(seed, &words);
        var v: [4]u64 = undefined;
        for (&v, 0..) |*x, i| x.* = @as(u64, words[2 * i]) | (@as(u64, words[2 * i + 1]) << 32);
        const initstate = (@as(u128, v[0]) << 64) | v[1];
        const initseq = (@as(u128, v[2]) << 64) | v[3];
        var self = Self{ .state = 0, .inc = (initseq << 1) | 1 };
        self.step();
        self.state +%= initstate;
        self.step();
        return self;
    }

    /// The next 64-bit draw.
    pub fn next64(self: *Self) u64 {
        self.step();
        const s = self.state;
        const x: u64 = @truncate((s >> 64) ^ s);
        const r: u6 = @intCast(s >> 122);
        return std.math.rotr(u64, x, r);
    }

    /// The next 32-bit draw (the low half of a 64-bit draw, then its high half).
    pub fn next32(self: *Self) u32 {
        if (self.has_u32) {
            self.has_u32 = false;
            return self.next_u32;
        }
        const n = self.next64();
        self.has_u32 = true;
        self.next_u32 = @intCast(n >> 32);
        return @truncate(n);
    }

    /// A uniform integer in `[0, max]` (numpy's `random_interval`: masked rejection).
    pub fn interval(self: *Self, max: u64) u64 {
        if (max == 0) return 0;
        var mask = max;
        inline for (.{ 1, 2, 4, 8, 16, 32 }) |s| mask |= mask >> s;
        if (max <= std.math.maxInt(u32)) {
            while (true) {
                const v = self.next32() & mask;
                if (v <= max) return v;
            }
        }
        while (true) {
            const v = self.next64() & mask;
            if (v <= max) return v;
        }
    }

    /// `rng.permutation(n)`: `arange(n)` shuffled in place (Fisher-Yates from the end).
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `allocator`: owns the result.
    /// - `n`: the length.
    ///
    /// Return: the permutation; allocation errors.
    pub fn permutation(self: *Self, allocator: std.mem.Allocator, n: usize) ![]u32 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const out = try allocator.alloc(u32, n);
        for (out, 0..) |*o, i| o.* = @intCast(i);
        var i = n;
        while (i > 1) {
            i -= 1;
            const j = self.interval(i);
            std.mem.swap(u32, &out[i], &out[j]);
        }
        return out;
    }

    fn step(self: *Self) void {
        self.state = self.state *% multiplier +% self.inc;
    }

    /// `SeedSequence(seed).generate_state(n)` as 32-bit words (pool size 4).
    fn seedSequence(seed: u64, out: []u32) void {
        const init_a: u32 = 0x43b0d7e5;
        const mult_a: u32 = 0x931e8875;
        const init_b: u32 = 0x8b51f9dd;
        const mult_b: u32 = 0x58f38ded;
        const mix_l: u32 = 0xca01f9dd;
        const mix_r: u32 = 0x4973f715;
        // The seed as little-endian 32-bit words (at least one).
        const entropy: [2]u32 = .{ @truncate(seed), @truncate(seed >> 32) };
        const n_entropy: usize = if (seed >> 32 == 0) 1 else 2;
        var hash_const = init_a;
        const Mix = struct {
            fn hashmix(value_in: u32, h: *u32) u32 {
                var value = value_in ^ h.*;
                h.* *%= mult_a;
                value *%= h.*;
                return value ^ (value >> 16);
            }
            fn mix(x: u32, y: u32) u32 {
                const r = mix_l *% x -% mix_r *% y;
                return r ^ (r >> 16);
            }
        };
        var pool: [4]u32 = undefined;
        for (&pool, 0..) |*p, i| p.* = Mix.hashmix(if (i < n_entropy) entropy[i] else 0, &hash_const);
        for (0..4) |src| {
            for (0..4) |dst| {
                if (src != dst) pool[dst] = Mix.mix(pool[dst], Mix.hashmix(pool[src], &hash_const));
            }
        }
        var h = init_b;
        for (out, 0..) |*o, i| {
            var v = pool[i % 4] ^ h;
            h *%= mult_b;
            v *%= h;
            o.* = v ^ (v >> 16);
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "numpy random reproduces default_rng(42).permutation" {
    var rng = NumpyRandom.init(42);
    const p = try rng.permutation(std.testing.allocator, 20);
    defer std.testing.allocator.free(p);
    // np.random.default_rng(42).permutation(20)
    try std.testing.expectEqualSlices(u32, &.{ 15, 9, 14, 7, 12, 10, 6, 19, 3, 0, 16, 5, 11, 18, 2, 4, 17, 1, 13, 8 }, p);
}
