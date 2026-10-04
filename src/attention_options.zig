const std = @import("std");
const log = std.log.scoped(.zignanogpt_attention_options);
const mod = @import("module.zig");

/// Causal sliding-window attention parameters, matching nanochat's
/// `flash_attn` semantics (FA3 `window_size=(window, 0)`).
///
/// Queries `q[B, Tq, H, D]`, keys/values `k, v[B, Tk, Hkv, D]` with `H` a
/// multiple of `Hkv` (query head `h` reads kv head `h / (H / Hkv)`). The
/// queries sit at the last `Tq` of the first `keys` key positions, so training
/// (`keys = Tq = Tk`) and KV-cache decoding (`keys` = filled length) are one op.
/// A query at absolute position `i` attends keys `j` with `0 <= i - j <= window`.
/// Scores are scaled by `1 / sqrt(D)`.
pub const AttentionOptions = struct {
    const Self = @This();

    /// Keys before the query that stay visible; `>= Tk` means full causal context.
    window: usize,
    /// Valid key positions in `k`/`v` (KV-cache fill); null means all `Tk`.
    keys: ?usize = null,

    /// Resolved problem dimensions.
    pub const Dims = struct { b: usize, tq: usize, tk: usize, keys: usize, h: usize, hkv: usize, d: usize };

    /// Validates the operand shapes and resolves the dimensions.
    ///
    /// Parameters:
    /// - `self`: the options.
    /// - `out`: `[B, Tq, H, D]`.
    /// - `q`: `[B, Tq, H, D]`.
    /// - `k`: `[B, Tk, Hkv, D]`.
    /// - `v`: same shape as `k`.
    /// - `lse`: optional `[B, H, Tq]` log-sum-exp output.
    ///
    /// Return: the dimensions; `error.ShapeMismatch` when the shapes disagree.
    pub fn dims(self: Self, out: mod.Shape, q: mod.Shape, k: mod.Shape, v: mod.Shape, lse: ?mod.Shape) error{ShapeMismatch}!Dims {
        if (q.rank != 4 or k.rank != 4 or !out.eql(q) or !k.eql(v)) return mismatch(q, k);
        const d = Dims{
            .b = q.dims[0],
            .tq = q.dims[1],
            .tk = k.dims[1],
            .keys = self.keys orelse k.dims[1],
            .h = q.dims[2],
            .hkv = k.dims[2],
            .d = q.dims[3],
        };
        if (k.dims[0] != d.b or k.dims[3] != d.d or d.hkv == 0 or d.h % d.hkv != 0) return mismatch(q, k);
        if (d.keys > d.tk or d.keys < d.tq) return mismatch(q, k);
        if (lse) |shape| {
            if (shape.rank != 3 or shape.dims[0] != d.b or shape.dims[1] != d.h or shape.dims[2] != d.tq) return mismatch(q, k);
        }
        return d;
    }

    fn mismatch(q: mod.Shape, k: mod.Shape) error{ShapeMismatch} {
        log.debug("attention shapes disagree: q {f}, k {f}", .{ q, k });
        return error.ShapeMismatch;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "attention options resolve training and cache shapes" {
    const q = try mod.Shape.init(&.{ 2, 5, 4, 8 });
    const k = try mod.Shape.init(&.{ 2, 5, 2, 8 });
    const d = try (mod.AttentionOptions{ .window = 3 }).dims(q, q, k, k, null);
    try std.testing.expectEqual(@as(usize, 5), d.keys);
    try std.testing.expectEqual(@as(usize, 2), d.hkv);

    const one = try mod.Shape.init(&.{ 2, 1, 4, 8 });
    const cache = try mod.Shape.init(&.{ 2, 16, 2, 8 });
    const c = try (mod.AttentionOptions{ .window = 16, .keys = 7 }).dims(one, one, cache, cache, null);
    try std.testing.expectEqual(@as(usize, 7), c.keys);

    const bad = try mod.Shape.init(&.{ 2, 5, 3, 8 });
    try std.testing.expectError(error.ShapeMismatch, (mod.AttentionOptions{ .window = 3 }).dims(q, q, bad, bad, null));
}
