const std = @import("std");
const log = std.log.scoped(.zignanogpt_python_random);

/// Python's `random.Random(seed)` (MT19937 seeded by `init_by_array`), for the
/// draws nanochat orders data with: `TaskMixture`'s `shuffle`, CORE's few-shot
/// `sample`. `random()` and `getrandbits` match CPython bit for bit.
pub const PythonRandom = struct {
    const Self = @This();

    const n = 624;
    const m = 397;

    mt: [n]u32,
    index: usize,

    /// Seeds like `random.Random(seed)` with a non-negative int.
    pub fn init(seed: u64) Self {
        var self: Self = .{ .mt = undefined, .index = n };
        // The key: the seed's 32-bit words, least significant first (at least one).
        const key: [2]u32 = .{ @truncate(seed), @truncate(seed >> 32) };
        self.initByArray(key[0..if (seed >> 32 == 0) 1 else 2]);
        return self;
    }

    /// The next 32 random bits (`genrand_uint32`).
    pub fn next32(self: *Self) u32 {
        if (self.index >= n) self.twist();
        var y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// `getrandbits(k)` for `k <= 64`.
    pub fn getrandbits(self: *Self, k: u7) u64 {
        if (k == 0) return 0;
        if (k <= 32) return self.next32() >> @intCast(32 - @as(u8, k));
        // Words least significant first; the last one keeps its top bits.
        const lo: u64 = self.next32();
        const hi: u64 = self.next32() >> @intCast(64 - @as(u8, k));
        return lo | (hi << 32);
    }

    /// `_randbelow(bound)`: rejection sampling on `bound.bit_length()` bits.
    pub fn below(self: *Self, bound: u64) u64 {
        std.debug.assert(bound > 0);
        const k: u7 = @intCast(64 - @clz(bound));
        while (true) {
            const r = self.getrandbits(k);
            if (r < bound) return r;
        }
    }

    /// `random()`: a float in [0, 1) with 53 random bits.
    pub fn random(self: *Self) f64 {
        const a: f64 = @floatFromInt(self.next32() >> 5);
        const b: f64 = @floatFromInt(self.next32() >> 6);
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    /// `shuffle(x)` in place.
    pub fn shuffle(self: *Self, comptime T: type, items: []T) void {
        var i = items.len;
        while (i > 1) {
            i -= 1;
            const j = self.below(i + 1);
            std.mem.swap(T, &items[i], &items[j]);
        }
    }

    /// `sample(range(population), k)` as indices (CPython's two strategies:
    /// a pool for small populations, a selected-set otherwise).
    ///
    /// Parameters:
    /// - `self`: the generator.
    /// - `allocator`: owns the result.
    /// - `population`: the population size.
    /// - `k`: picks, at most `population`.
    ///
    /// Return: the picked indices in draw order; allocation errors.
    pub fn sample(self: *Self, allocator: std.mem.Allocator, population: usize, k: usize) ![]usize {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (k > population) return error.SampleTooLarge;
        const out = try allocator.alloc(usize, k);
        errdefer allocator.free(out);
        var setsize: usize = 21;
        if (k > 5) {
            // 4 ** ceil(log(3k, 4)); 3k is never a power of 4.
            var power: usize = 1;
            while (power < k * 3) power *= 4;
            setsize += power;
        }
        if (population <= setsize) {
            const pool = try allocator.alloc(usize, population);
            defer allocator.free(pool);
            for (pool, 0..) |*p, i| p.* = i;
            for (out, 0..) |*o, i| {
                const j = self.below(population - i);
                o.* = pool[j];
                pool[j] = pool[population - i - 1];
            }
        } else {
            var selected: std.AutoHashMapUnmanaged(usize, void) = .empty;
            defer selected.deinit(allocator);
            for (out) |*o| {
                var j = self.below(population);
                while (selected.contains(j)) j = self.below(population);
                try selected.put(allocator, j, {});
                o.* = j;
            }
        }
        return out;
    }

    fn initGenrand(self: *Self, s: u32) void {
        self.mt[0] = s;
        for (1..n) |i| {
            const prev = self.mt[i - 1];
            self.mt[i] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(i));
        }
        self.index = n;
    }

    fn initByArray(self: *Self, key: []const u32) void {
        self.initGenrand(19650218);
        var i: usize = 1;
        var j: usize = 0;
        var k: usize = @max(n, key.len);
        while (k > 0) : (k -= 1) {
            const prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)) *% 1664525)) +% key[j] +% @as(u32, @intCast(j));
            i += 1;
            j += 1;
            if (i >= n) {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            if (j >= key.len) j = 0;
        }
        k = n - 1;
        while (k > 0) : (k -= 1) {
            const prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)) *% 1566083941)) -% @as(u32, @intCast(i));
            i += 1;
            if (i >= n) {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x80000000;
    }

    fn twist(self: *Self) void {
        for (0..n) |i| {
            const y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % n] & 0x7fffffff);
            var v = self.mt[(i + m) % n] ^ (y >> 1);
            if (y & 1 != 0) v ^= 0x9908b0df;
            self.mt[i] = v;
        }
        self.index = 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "python random matches CPython's shuffle, random and sample" {
    var rng = PythonRandom.init(42);
    var x: [20]u32 = undefined;
    for (&x, 0..) |*v, i| v.* = @intCast(i);
    rng.shuffle(u32, &x);
    // x = list(range(20)); random.Random(42).shuffle(x)
    try std.testing.expectEqualSlices(u32, &.{ 19, 5, 14, 4, 9, 13, 15, 18, 6, 12, 17, 10, 1, 11, 2, 16, 7, 8, 0, 3 }, &x);
    var r = PythonRandom.init(42);
    // random.Random(42).random()
    try std.testing.expectEqual(@as(f64, 0.6394267984578837), r.random());
}
