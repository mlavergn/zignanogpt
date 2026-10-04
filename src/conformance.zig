const std = @import("std");
const log = std.log.scoped(.zignanogpt_conformance);
const mod = @import("module.zig");

/// The backend contract, as an executable specification.
///
/// `assertContract` checks a backend type declares every required member;
/// `runAll` exercises each op against a host reference computed in f64. Every
/// backend's tests call `runAll`, so a behavioural change has to hold on all of
/// them. Results are compared with tolerances: summation order differs between
/// backends (and from PyTorch), bit-exactness is not part of the contract.
///
/// The contract, shaped for CPU, Metal and CUDA alike:
/// - Memory is opaque (`Buffer`); host data moves only via `upload`/`download`.
///   An element offset maps to Metal's `setBuffer:offset:` and CUDA's
///   `CUdeviceptr + bytes`, so views cost nothing on any backend.
/// - Ops may be queued (Metal command buffer, CUDA stream). Uploads, ops and
///   downloads take effect in issue order; `sync` waits for everything issued,
///   and `download` syncs implicitly. A failure inside queued work may surface
///   only at the next `sync`/`download`.
/// - No op returns host data: results (a loss, a norm) land in tensors, read
///   with `download` at an explicit sync point, never one sync per op.
/// - `free` is safe right after issuing ops that use the tensor; a queued
///   backend defers the release until that work completes.
/// - Tensors are contiguous, row-major, with an element offset (views along
///   dimension 0 and reshapes are free; no strides). Transposes are op flags
///   (`MatmulOptions`), never layout.
/// - Ops validate dtype and shape on the host before issuing (shared rules
///   such as `MatmulOptions.dims`) and return errors, never read out of bounds.
/// - Kernel workspace is backend-internal (CPU: matmul packing scratch).
/// - One instance is driven from one thread; parallelism is the backend's own.
pub const Conformance = struct {
    /// Declarations every backend type must provide.
    pub const required_decls = [_][]const u8{
        "Buffer",            "Options",            "name",              "init",
        "deinit",            "alloc",              "free",              "sync",
        "upload",            "download",           "fill",              "copy",
        "add",               "mul",                "scale",             "matmul",
        "embedding",         "rmsnorm",            "rope",              "attention",
        "reluSquare",        "combine",            "gateLinear",        "smear",
        "valueMix",          "softcap",            "crossEntropy",      "crossEntropyBackward",
        "rmsnormBackward",   "ropeBackward",       "attentionBackward", "reluSquareBackward",
        "dot",               "gateLinearBackward", "smearBackward",     "valueMixBackward",
        "embeddingBackward", "adamwStep",          "muonMomentum",      "muonPrepare",
        "muonFinish",        "crossEntropyRows",   "gatedAdd",
    };

    /// Fails compilation when `B` misses part of the contract.
    ///
    /// Parameters:
    /// - `B`: the backend type.
    ///
    /// Return: nothing.
    pub fn assertContract(comptime B: type) void {
        inline for (required_decls) |decl| {
            if (!@hasDecl(B, decl)) @compileError(@typeName(B) ++ " is missing backend member '" ++ decl ++ "'");
        }
    }

    /// Runs every conformance check on a fresh `B`.
    ///
    /// Parameters:
    /// - `B`: the backend type (the build's `mod.Backend`).
    /// - `allocator`: for host-side reference data.
    /// - `io`: handed to the backend.
    ///
    /// Return: nothing; the first failing check's error.
    pub fn runAll(comptime B: type, allocator: std.mem.Allocator, io: std.Io) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        assertContract(B);
        var backend = try B.init(allocator, io, .{});
        defer backend.deinit();
        try roundTrip(B, &backend);
        try fillAndCopy(B, &backend);
        try elementwise(B, &backend, allocator);
        try views(B, &backend);
        try matmul(B, &backend, allocator);
        try rowOps(B, &backend, allocator);
        try gateOps(B, &backend, allocator);
        try ropeOp(B, &backend, allocator);
        try attentionOp(B, &backend, allocator);
        try rowBackward(B, &backend, allocator);
        try attentionBackwardOp(B, &backend, allocator);
        try gateBackward(B, &backend, allocator);
        try lossBackward(B, &backend, allocator);
        try optimizerOps(B, &backend, allocator);
    }

    /// A tensor holding `data` (f32 or i32), allocated on `backend`.
    fn tensorFrom(comptime B: type, backend: *B, comptime T: type, dims: []const usize, data: []const T) !mod.Tensor {
        const t = try backend.alloc(mod.Dtype.of(T), dims);
        errdefer backend.free(t);
        try backend.upload(t, T, data);
        return t;
    }

    /// Downloads an f32 tensor into new host memory.
    fn hostCopy(comptime B: type, backend: *B, allocator: std.mem.Allocator, t: mod.Tensor) ![]f32 {
        const out = try allocator.alloc(f32, t.numel());
        errdefer allocator.free(out);
        try backend.download(t, f32, out);
        return out;
    }

    fn expectClose(expected: []const f32, got: []const f32, tolerance: f32) !void {
        try std.testing.expectEqual(expected.len, got.len);
        for (expected, got) |e, g| try std.testing.expectApproxEqAbs(e, g, tolerance);
    }

    fn sigmoid(x: f32) f32 {
        return 1 / (1 + @exp(-x));
    }

    /// embedding, rmsnorm, reluSquare, combine, softcap.
    fn rowOps(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        const table = try tensorFrom(B, backend, f32, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
        defer backend.free(table);
        const ids = try tensorFrom(B, backend, i32, &.{ 2, 2 }, &.{ 2, 0, 1, 2 });
        defer backend.free(ids);
        const emb = try backend.alloc(.f32, &.{ 2, 2, 2 });
        defer backend.free(emb);
        try backend.embedding(emb, table, ids);
        const got = try hostCopy(B, backend, allocator, emb);
        defer allocator.free(got);
        try std.testing.expectEqualSlices(f32, &.{ 5, 6, 1, 2, 3, 4, 5, 6 }, got);
        const bad_ids = try tensorFrom(B, backend, i32, &.{4}, &.{ 0, 3, 0, 0 });
        defer backend.free(bad_ids);
        try std.testing.expectError(error.OutOfBounds, backend.embedding(emb, table, bad_ids));

        const x = try tensorFrom(B, backend, f32, &.{ 2, 3 }, &.{ 1, -2, 2, 0, 0, 0 });
        defer backend.free(x);
        const y = try backend.alloc(.f32, &.{ 2, 3 });
        defer backend.free(y);
        try backend.rmsnorm(y, x, 1e-6);
        const norm = try hostCopy(B, backend, allocator, y);
        defer allocator.free(norm);
        const r = 1 / @sqrt(@as(f32, 3) + 1e-6); // mean square of (1, -2, 2) is 3
        try expectClose(&.{ r, -2 * r, 2 * r, 0, 0, 0 }, norm, 1e-6);

        try backend.reluSquare(y, x);
        const relu = try hostCopy(B, backend, allocator, y);
        defer allocator.free(relu);
        try std.testing.expectEqualSlices(f32, &.{ 1, 0, 4, 0, 0, 0 }, relu);

        const lam = try tensorFrom(B, backend, f32, &.{2}, &.{ 0.5, 3 });
        defer backend.free(lam);
        try backend.combine(y, x, mod.Scalar.of(try lam.rows(1, 1), 2), x, mod.Scalar.constant(-1));
        const comb = try hostCopy(B, backend, allocator, y);
        defer allocator.free(comb);
        try std.testing.expectEqualSlices(f32, &.{ 5, -10, 10, 0, 0, 0 }, comb); // (3 * 2 - 1) * x

        const padded = try tensorFrom(B, backend, f32, &.{ 2, 3 }, &.{ 30, -15, 99, 0, 7.5, 99 });
        defer backend.free(padded);
        const capped = try backend.alloc(.f32, &.{ 2, 2 });
        defer backend.free(capped);
        try backend.softcap(capped, padded, 15);
        const cap = try hostCopy(B, backend, allocator, capped);
        defer allocator.free(cap);
        try expectClose(&.{ 15 * std.math.tanh(@as(f32, 2)), 15 * std.math.tanh(@as(f32, -1)), 0, 15 * std.math.tanh(@as(f32, 0.5)) }, cap, 1e-5);
        try std.testing.expectError(error.Aliasing, backend.softcap(padded, padded, 15));
    }

    /// gateLinear, smear, valueMix.
    fn gateOps(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        // x: [B=1, T=3, C=3]; gate reads the first 2 channels.
        const x = try tensorFrom(B, backend, f32, &.{ 1, 3, 3 }, &.{ 1, 2, 9, 3, 4, 9, 5, 6, 9 });
        defer backend.free(x);
        const w = try tensorFrom(B, backend, f32, &.{ 2, 2 }, &.{ 1, 0, 0.5, -1 });
        defer backend.free(w);
        const g = try backend.alloc(.f32, &.{ 3, 2 });
        defer backend.free(g);
        try backend.gateLinear(g, try x.reshape(&.{ 3, 3 }), w);
        const gates = try hostCopy(B, backend, allocator, g);
        defer allocator.free(gates);
        try expectClose(&.{ 1, -1.5, 3, -2.5, 5, -3.5 }, gates, 1e-6);

        const g1 = try tensorFrom(B, backend, f32, &.{ 3, 1 }, &.{ 7, 0, -1 });
        defer backend.free(g1);
        const lam = try tensorFrom(B, backend, f32, &.{1}, &.{0.5});
        defer backend.free(lam);
        const out = try backend.alloc(.f32, &.{ 1, 3, 3 });
        defer backend.free(out);
        try backend.smear(out, x, g1, mod.Scalar.of(lam, 1));
        const sm = try hostCopy(B, backend, allocator, out);
        defer allocator.free(sm);
        const a = 0.5 * sigmoid(0);
        const b = 0.5 * sigmoid(-1);
        try expectClose(&.{ 1, 2, 9, 3 + a * 1, 4 + a * 2, 9 + a * 9, 5 + b * 3, 6 + b * 4, 9 + b * 9 }, sm, 1e-6);
        try std.testing.expectError(error.Aliasing, backend.smear(x, x, g1, mod.Scalar.of(lam, 1)));

        // gatedAdd: out = x + lambda * sigmoid(gate) * y, row by row (in place).
        const ga = try tensorFrom(B, backend, f32, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
        defer backend.free(ga);
        const gb = try tensorFrom(B, backend, f32, &.{ 2, 2 }, &.{ 10, 20, 30, 40 });
        defer backend.free(gb);
        const gg = try tensorFrom(B, backend, f32, &.{ 2, 1 }, &.{ 0, -1 });
        defer backend.free(gg);
        try backend.gatedAdd(ga, ga, gb, gg, mod.Scalar.of(lam, 1));
        const added = try hostCopy(B, backend, allocator, ga);
        defer allocator.free(added);
        try expectClose(&.{ 1 + a * 10, 2 + a * 20, 3 + b * 30, 4 + b * 40 }, added, 1e-5);

        // v: [R=3, H=2, D=1] += 3 * sigmoid(gate) * ve
        const v = try tensorFrom(B, backend, f32, &.{ 3, 2 }, &.{ 1, 1, 1, 1, 1, 1 });
        defer backend.free(v);
        const ve = try tensorFrom(B, backend, f32, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
        defer backend.free(ve);
        try backend.valueMix(v, ve, g);
        const mixed = try hostCopy(B, backend, allocator, v);
        defer allocator.free(mixed);
        var expected: [6]f32 = undefined;
        for (&expected, 0..) |*e, i| e.* = 1 + 3 * sigmoid(gates[i]) * @as(f32, @floatFromInt(i + 1));
        try expectClose(&expected, mixed, 1e-5);
    }

    /// rope against the formula, at a position offset.
    fn ropeOp(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        const dims = [_]usize{ 2, 3, 2, 4 }; // B, T, H, D
        const n = 2 * 3 * 2 * 4;
        var rng = mod.Random.init(3);
        var host: [n]f32 = undefined;
        rng.fillUniform(&host, -1, 1);
        var cos_h: [10 * 2]f32 = undefined;
        var sin_h: [10 * 2]f32 = undefined;
        for (0..10) |p| for (0..2) |i| {
            const angle = @as(f32, @floatFromInt(p)) * (0.3 + @as(f32, @floatFromInt(i)));
            cos_h[p * 2 + i] = @cos(angle);
            sin_h[p * 2 + i] = @sin(angle);
        };
        const x = try tensorFrom(B, backend, f32, &dims, &host);
        defer backend.free(x);
        const cos = try tensorFrom(B, backend, f32, &.{ 10, 2 }, &cos_h);
        defer backend.free(cos);
        const sin = try tensorFrom(B, backend, f32, &.{ 10, 2 }, &sin_h);
        defer backend.free(sin);
        try backend.rope(x, x, cos, sin, 4); // in place
        const got = try hostCopy(B, backend, allocator, x);
        defer allocator.free(got);
        var expected: [n]f32 = undefined;
        for (0..2 * 3 * 2) |r| {
            const pos = 4 + (r / 2) % 3;
            for (0..2) |i| {
                const x1 = host[r * 4 + i];
                const x2 = host[r * 4 + 2 + i];
                expected[r * 4 + i] = x1 * cos_h[pos * 2 + i] + x2 * sin_h[pos * 2 + i];
                expected[r * 4 + 2 + i] = x2 * cos_h[pos * 2 + i] - x1 * sin_h[pos * 2 + i];
            }
        }
        try expectClose(&expected, got, 1e-6);
        try std.testing.expectError(error.OutOfBounds, backend.rope(x, x, cos, sin, 8));
    }

    /// attention against an explicit masked softmax, for windows, GQA and a
    /// partly filled cache.
    fn attentionOp(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        const Case = struct { b: usize, tq: usize, tk: usize, keys: usize, h: usize, hkv: usize, d: usize, window: usize };
        const cases = [_]Case{
            .{ .b = 2, .tq = 9, .tk = 9, .keys = 9, .h = 4, .hkv = 2, .d = 8, .window = 100 },
            .{ .b = 1, .tq = 20, .tk = 20, .keys = 20, .h = 2, .hkv = 1, .d = 5, .window = 3 },
            .{ .b = 2, .tq = 3, .tk = 12, .keys = 10, .h = 3, .hkv = 3, .d = 4, .window = 4 },
        };
        var rng = mod.Random.init(9);
        for (cases) |c| {
            const qn = c.b * c.tq * c.h * c.d;
            const kn = c.b * c.tk * c.hkv * c.d;
            const hq = try allocator.alloc(f32, qn);
            defer allocator.free(hq);
            const hk = try allocator.alloc(f32, kn);
            defer allocator.free(hk);
            const hv = try allocator.alloc(f32, kn);
            defer allocator.free(hv);
            rng.fillUniform(hq, -2, 2);
            rng.fillUniform(hk, -2, 2);
            rng.fillUniform(hv, -2, 2);
            const q = try tensorFrom(B, backend, f32, &.{ c.b, c.tq, c.h, c.d }, hq);
            defer backend.free(q);
            const k = try tensorFrom(B, backend, f32, &.{ c.b, c.tk, c.hkv, c.d }, hk);
            defer backend.free(k);
            const v = try tensorFrom(B, backend, f32, &.{ c.b, c.tk, c.hkv, c.d }, hv);
            defer backend.free(v);
            const out = try backend.alloc(.f32, &.{ c.b, c.tq, c.h, c.d });
            defer backend.free(out);
            const lse = try backend.alloc(.f32, &.{ c.b, c.h, c.tq });
            defer backend.free(lse);
            try backend.attention(out, q, k, v, lse, .{ .window = c.window, .keys = c.keys });
            const got = try hostCopy(B, backend, allocator, out);
            defer allocator.free(got);
            const got_lse = try hostCopy(B, backend, allocator, lse);
            defer allocator.free(got_lse);

            const scale = 1 / @sqrt(@as(f64, @floatFromInt(c.d)));
            for (0..c.b) |b| for (0..c.h) |h| for (0..c.tq) |t| {
                const pos = c.keys - c.tq + t;
                const kh = h / (c.h / c.hkv);
                var max: f64 = -std.math.inf(f64);
                for (0..pos + 1) |j| {
                    if (pos - j > c.window) continue;
                    max = @max(max, dotAt(hq, hk, c, b, t, h, j, kh) * scale);
                }
                var sum: f64 = 0;
                var acc = [_]f64{0} ** 8;
                for (0..pos + 1) |j| {
                    if (pos - j > c.window) continue;
                    const p = @exp(dotAt(hq, hk, c, b, t, h, j, kh) * scale - max);
                    sum += p;
                    for (0..c.d) |i| acc[i] += p * hv[((b * c.tk + j) * c.hkv + kh) * c.d + i];
                }
                for (0..c.d) |i| {
                    const g = got[((b * c.tq + t) * c.h + h) * c.d + i];
                    try std.testing.expectApproxEqAbs(acc[i] / sum, @as(f64, g), 1e-5);
                }
                try std.testing.expectApproxEqAbs(max + @log(sum), @as(f64, got_lse[(b * c.h + h) * c.tq + t]), 1e-5);
            };
        }
    }

    fn dotAt(q: []const f32, k: []const f32, c: anytype, b: usize, t: usize, h: usize, j: usize, kh: usize) f64 {
        var s: f64 = 0;
        for (0..c.d) |i| s += @as(f64, q[((b * c.tq + t) * c.h + h) * c.d + i]) * @as(f64, k[((b * c.tk + j) * c.hkv + kh) * c.d + i]);
        return s;
    }

    /// `<weights, output>` in f64.
    fn lossOf(comptime B: type, backend: *B, allocator: std.mem.Allocator, output: mod.Tensor, weights: []const f32) !f64 {
        const host = try hostCopy(B, backend, allocator, output);
        defer allocator.free(host);
        var sum: f64 = 0;
        for (host, weights) |o, w| sum += @as(f64, o) * @as(f64, w);
        return sum;
    }

    /// Compares `analytic` (dL/d`input`) with central differences of
    /// `L = <weights, output>`, where `ctx.run()` recomputes `output`.
    fn gradCheck(comptime B: type, backend: *B, allocator: std.mem.Allocator, label: []const u8, input: mod.Tensor, host: []f32, output: mod.Tensor, weights: []const f32, analytic: []const f32, ctx: anytype) !void {
        const h: f32 = 1e-2;
        for (host, 0..) |*x, i| {
            const saved = x.*;
            x.* = saved + h;
            try backend.upload(input, f32, host);
            try ctx.run();
            const plus = try lossOf(B, backend, allocator, output, weights);
            x.* = saved - h;
            try backend.upload(input, f32, host);
            try ctx.run();
            const minus = try lossOf(B, backend, allocator, output, weights);
            x.* = saved;
            const numeric = (plus - minus) / (2 * h);
            if (@abs(numeric - analytic[i]) > 2e-2 * (1 + @abs(numeric))) {
                std.debug.print("{s}[{d}]: numeric {d}, analytic {d}\n", .{ label, i, numeric, analytic[i] });
                return error.GradientMismatch;
            }
        }
        try backend.upload(input, f32, host);
        try ctx.run();
    }

    /// Random host data in `[lo, hi)`.
    fn randomHost(allocator: std.mem.Allocator, rng: *mod.Random, n: usize, lo: f32, hi: f32) ![]f32 {
        const out = try allocator.alloc(f32, n);
        rng.fillUniform(out, lo, hi);
        return out;
    }

    /// rmsnorm, rope and reluSquare backward against finite differences.
    fn rowBackward(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        var rng = mod.Random.init(21);
        const hx = try randomHost(allocator, &rng, 3 * 8, -2, 2);
        defer allocator.free(hx);
        const w = try randomHost(allocator, &rng, 3 * 8, -1, 1);
        defer allocator.free(w);
        const x = try tensorFrom(B, backend, f32, &.{ 1, 3, 1, 8 }, hx);
        defer backend.free(x);
        const y = try backend.alloc(.f32, &.{ 1, 3, 1, 8 });
        defer backend.free(y);
        const dy = try tensorFrom(B, backend, f32, &.{ 1, 3, 1, 8 }, w);
        defer backend.free(dy);
        const dx = try backend.alloc(.f32, &.{ 1, 3, 1, 8 });
        defer backend.free(dx);

        // rmsnorm, then the same with accumulate onto ones.
        const Norm = struct {
            be: *B,
            y: mod.Tensor,
            x: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.rmsnorm(c.y, c.x, 1e-6);
            }
        };
        try backend.rmsnormBackward(dx, dy, x, 1e-6, false);
        const g_norm = try hostCopy(B, backend, allocator, dx);
        defer allocator.free(g_norm);
        try gradCheck(B, backend, allocator, "rmsnorm", x, hx, y, w, g_norm, Norm{ .be = backend, .y = y, .x = x });
        try backend.fill(dx, 1);
        try backend.rmsnormBackward(dx, dy, x, 1e-6, true);
        const g_acc = try hostCopy(B, backend, allocator, dx);
        defer allocator.free(g_acc);
        for (g_acc, g_norm) |a, g| try std.testing.expectApproxEqAbs(g + 1, a, 1e-6);

        // rope over a 5-position table, starting at position 1.
        var cos_h: [5 * 4]f32 = undefined;
        var sin_h: [5 * 4]f32 = undefined;
        for (&cos_h, &sin_h, 0..) |*c, *s, i| {
            c.* = @cos(@as(f32, @floatFromInt(i)) * 0.7);
            s.* = @sin(@as(f32, @floatFromInt(i)) * 0.7);
        }
        const cos = try tensorFrom(B, backend, f32, &.{ 5, 4 }, &cos_h);
        defer backend.free(cos);
        const sin = try tensorFrom(B, backend, f32, &.{ 5, 4 }, &sin_h);
        defer backend.free(sin);
        const Rope = struct {
            be: *B,
            y: mod.Tensor,
            x: mod.Tensor,
            cos: mod.Tensor,
            sin: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.rope(c.y, c.x, c.cos, c.sin, 1);
            }
        };
        try backend.ropeBackward(dx, dy, cos, sin, 1);
        const g_rope = try hostCopy(B, backend, allocator, dx);
        defer allocator.free(g_rope);
        try gradCheck(B, backend, allocator, "rope", x, hx, y, w, g_rope, Rope{ .be = backend, .y = y, .x = x, .cos = cos, .sin = sin });

        // relu^2, with inputs kept away from the kink at 0.
        for (hx) |*v| v.* = if (v.* >= 0) v.* + 0.1 else v.* - 0.1;
        try backend.upload(x, f32, hx);
        const Relu = struct {
            be: *B,
            y: mod.Tensor,
            x: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.reluSquare(c.y, c.x);
            }
        };
        try backend.reluSquareBackward(dx, dy, x);
        const g_relu = try hostCopy(B, backend, allocator, dx);
        defer allocator.free(g_relu);
        try gradCheck(B, backend, allocator, "reluSquare", x, hx, y, w, g_relu, Relu{ .be = backend, .y = y, .x = x });
    }

    /// attention backward (dq, dk, dv) against finite differences.
    fn attentionBackwardOp(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        var rng = mod.Random.init(22);
        const qd = [_]usize{ 2, 6, 4, 4 }; // B, T, H, D
        const kd = [_]usize{ 2, 6, 2, 4 };
        const hq = try randomHost(allocator, &rng, 2 * 6 * 4 * 4, -1, 1);
        defer allocator.free(hq);
        const hk = try randomHost(allocator, &rng, 2 * 6 * 2 * 4, -1, 1);
        defer allocator.free(hk);
        const hv = try randomHost(allocator, &rng, 2 * 6 * 2 * 4, -1, 1);
        defer allocator.free(hv);
        const w = try randomHost(allocator, &rng, 2 * 6 * 4 * 4, -1, 1);
        defer allocator.free(w);
        const q = try tensorFrom(B, backend, f32, &qd, hq);
        defer backend.free(q);
        const k = try tensorFrom(B, backend, f32, &kd, hk);
        defer backend.free(k);
        const v = try tensorFrom(B, backend, f32, &kd, hv);
        defer backend.free(v);
        const out = try backend.alloc(.f32, &qd);
        defer backend.free(out);
        const lse = try backend.alloc(.f32, &.{ 2, 4, 6 });
        defer backend.free(lse);
        const dout = try tensorFrom(B, backend, f32, &qd, w);
        defer backend.free(dout);
        const dq = try backend.alloc(.f32, &qd);
        defer backend.free(dq);
        const dk = try backend.alloc(.f32, &kd);
        defer backend.free(dk);
        const dv = try backend.alloc(.f32, &kd);
        defer backend.free(dv);
        const options = mod.AttentionOptions{ .window = 2 };

        try backend.attention(out, q, k, v, lse, options);
        try backend.attentionBackward(dq, dk, dv, dout, q, k, v, out, lse, options);
        const gq = try hostCopy(B, backend, allocator, dq);
        defer allocator.free(gq);
        const gk = try hostCopy(B, backend, allocator, dk);
        defer allocator.free(gk);
        const gv = try hostCopy(B, backend, allocator, dv);
        defer allocator.free(gv);
        const Attn = struct {
            be: *B,
            out: mod.Tensor,
            q: mod.Tensor,
            k: mod.Tensor,
            v: mod.Tensor,
            options: mod.AttentionOptions,
            fn run(c: @This()) !void {
                try c.be.attention(c.out, c.q, c.k, c.v, null, c.options);
            }
        };
        const ctx = Attn{ .be = backend, .out = out, .q = q, .k = k, .v = v, .options = options };
        try gradCheck(B, backend, allocator, "attention dq", q, hq, out, w, gq, ctx);
        try gradCheck(B, backend, allocator, "attention dk", k, hk, out, w, gk, ctx);
        try gradCheck(B, backend, allocator, "attention dv", v, hv, out, w, gv, ctx);
    }

    /// gateLinear, smear and valueMix backward against finite differences.
    fn gateBackward(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        var rng = mod.Random.init(23);
        // gateLinear: x [6, 5], w [2, 3]
        const hx = try randomHost(allocator, &rng, 6 * 5, -1, 1);
        defer allocator.free(hx);
        const hw = try randomHost(allocator, &rng, 2 * 3, -1, 1);
        defer allocator.free(hw);
        const wt = try randomHost(allocator, &rng, 6 * 2, -1, 1);
        defer allocator.free(wt);
        const x = try tensorFrom(B, backend, f32, &.{ 6, 5 }, hx);
        defer backend.free(x);
        const w = try tensorFrom(B, backend, f32, &.{ 2, 3 }, hw);
        defer backend.free(w);
        const out = try backend.alloc(.f32, &.{ 6, 2 });
        defer backend.free(out);
        const dout = try tensorFrom(B, backend, f32, &.{ 6, 2 }, wt);
        defer backend.free(dout);
        const dx = try backend.alloc(.f32, &.{ 6, 5 });
        defer backend.free(dx);
        const dw = try backend.alloc(.f32, &.{ 2, 3 });
        defer backend.free(dw);
        try backend.gateLinearBackward(dx, dw, dout, x, w);
        const gx = try hostCopy(B, backend, allocator, dx);
        defer allocator.free(gx);
        const gw = try hostCopy(B, backend, allocator, dw);
        defer allocator.free(gw);
        const Gate = struct {
            be: *B,
            out: mod.Tensor,
            x: mod.Tensor,
            w: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.gateLinear(c.out, c.x, c.w);
            }
        };
        const gate_ctx = Gate{ .be = backend, .out = out, .x = x, .w = w };
        try gradCheck(B, backend, allocator, "gateLinear dx", x, hx, out, wt, gx, gate_ctx);
        try gradCheck(B, backend, allocator, "gateLinear dw", w, hw, out, wt, gw, gate_ctx);

        // smear: x [2, 3, 4], gate [6, 1], lambda [1]
        const sx = try randomHost(allocator, &rng, 2 * 3 * 4, -1, 1);
        defer allocator.free(sx);
        const sg = try randomHost(allocator, &rng, 6, -2, 2);
        defer allocator.free(sg);
        const sl = [_]f32{0.7};
        var sl_host = sl;
        const sw = try randomHost(allocator, &rng, 2 * 3 * 4, -1, 1);
        defer allocator.free(sw);
        const smx = try tensorFrom(B, backend, f32, &.{ 2, 3, 4 }, sx);
        defer backend.free(smx);
        const smg = try tensorFrom(B, backend, f32, &.{ 6, 1 }, sg);
        defer backend.free(smg);
        const sml = try tensorFrom(B, backend, f32, &.{1}, &sl);
        defer backend.free(sml);
        const smout = try backend.alloc(.f32, &.{ 2, 3, 4 });
        defer backend.free(smout);
        const smdout = try tensorFrom(B, backend, f32, &.{ 2, 3, 4 }, sw);
        defer backend.free(smdout);
        const smdx = try backend.alloc(.f32, &.{ 2, 3, 4 });
        defer backend.free(smdx);
        const smdg = try backend.alloc(.f32, &.{ 6, 1 });
        defer backend.free(smdg);
        const smdl = try backend.alloc(.f32, &.{1});
        defer backend.free(smdl);
        try backend.smearBackward(smdx, smdg, smdl, smdout, smx, smg, mod.Scalar.of(sml, 1));
        const g_sx = try hostCopy(B, backend, allocator, smdx);
        defer allocator.free(g_sx);
        const g_sg = try hostCopy(B, backend, allocator, smdg);
        defer allocator.free(g_sg);
        const g_sl = try hostCopy(B, backend, allocator, smdl);
        defer allocator.free(g_sl);
        const Smear = struct {
            be: *B,
            out: mod.Tensor,
            x: mod.Tensor,
            gate: mod.Tensor,
            lambda: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.smear(c.out, c.x, c.gate, mod.Scalar.of(c.lambda, 1));
            }
        };
        const smear_ctx = Smear{ .be = backend, .out = smout, .x = smx, .gate = smg, .lambda = sml };
        try gradCheck(B, backend, allocator, "smear dx", smx, sx, smout, sw, g_sx, smear_ctx);
        try gradCheck(B, backend, allocator, "smear dgate", smg, sg, smout, sw, g_sg, smear_ctx);
        try gradCheck(B, backend, allocator, "smear dlambda", sml, &sl_host, smout, sw, g_sl, smear_ctx);

        // valueMix: v, ve [3, 2, 2], gate [3, 2]; recomputed from a pristine v each run.
        const hv = try randomHost(allocator, &rng, 12, -1, 1);
        defer allocator.free(hv);
        const hve = try randomHost(allocator, &rng, 12, -1, 1);
        defer allocator.free(hve);
        const hg = try randomHost(allocator, &rng, 6, -2, 2);
        defer allocator.free(hg);
        const vw = try randomHost(allocator, &rng, 12, -1, 1);
        defer allocator.free(vw);
        const v0 = try tensorFrom(B, backend, f32, &.{ 3, 2, 2 }, hv);
        defer backend.free(v0);
        const ve = try tensorFrom(B, backend, f32, &.{ 3, 2, 2 }, hve);
        defer backend.free(ve);
        const vg = try tensorFrom(B, backend, f32, &.{ 3, 2 }, hg);
        defer backend.free(vg);
        const vout = try backend.alloc(.f32, &.{ 3, 2, 2 });
        defer backend.free(vout);
        const vdv = try tensorFrom(B, backend, f32, &.{ 3, 2, 2 }, vw);
        defer backend.free(vdv);
        const vdve = try backend.alloc(.f32, &.{ 3, 2, 2 });
        defer backend.free(vdve);
        const vdg = try backend.alloc(.f32, &.{ 3, 2 });
        defer backend.free(vdg);
        try backend.valueMixBackward(vdve, vdg, vdv, ve, vg);
        const g_ve = try hostCopy(B, backend, allocator, vdve);
        defer allocator.free(g_ve);
        const g_vg = try hostCopy(B, backend, allocator, vdg);
        defer allocator.free(g_vg);
        const Mix = struct {
            be: *B,
            out: mod.Tensor,
            v0: mod.Tensor,
            ve: mod.Tensor,
            gate: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.copy(c.out, c.v0);
                try c.be.valueMix(c.out, c.ve, c.gate);
            }
        };
        const mix_ctx = Mix{ .be = backend, .out = vout, .v0 = v0, .ve = ve, .gate = vg };
        try gradCheck(B, backend, allocator, "valueMix dve", ve, hve, vout, vw, g_ve, mix_ctx);
        try gradCheck(B, backend, allocator, "valueMix dgate", vg, hg, vout, vw, g_vg, mix_ctx);
    }

    /// crossEntropy (through softcap) backward, dot and embeddingBackward.
    fn lossBackward(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        var rng = mod.Random.init(24);
        const hp = try randomHost(allocator, &rng, 4 * 6, -4, 4);
        defer allocator.free(hp);
        const pad = try tensorFrom(B, backend, f32, &.{ 4, 6 }, hp);
        defer backend.free(pad);
        const capped = try backend.alloc(.f32, &.{ 4, 5 });
        defer backend.free(capped);
        const targets = try tensorFrom(B, backend, i32, &.{4}, &.{ 1, -1, 4, 0 });
        defer backend.free(targets);
        const loss = try backend.alloc(.f32, &.{1});
        defer backend.free(loss);
        const Xent = struct {
            be: *B,
            loss: mod.Tensor,
            capped: mod.Tensor,
            pad: mod.Tensor,
            targets: mod.Tensor,
            fn run(c: @This()) !void {
                try c.be.softcap(c.capped, c.pad, 3);
                try c.be.crossEntropy(c.loss, c.capped, c.targets);
            }
        };
        const ctx = Xent{ .be = backend, .loss = loss, .capped = capped, .pad = pad, .targets = targets };
        try ctx.run();
        const dpad = try backend.alloc(.f32, &.{ 4, 6 });
        defer backend.free(dpad);
        try backend.crossEntropyBackward(dpad, capped, targets, 3, 1);
        const g = try hostCopy(B, backend, allocator, dpad);
        defer allocator.free(g);
        for (0..4) |r| try std.testing.expectEqual(@as(f32, 0), g[r * 6 + 5]); // padding column
        for (0..6) |c| try std.testing.expectEqual(@as(f32, 0), g[6 + c]); // ignored row
        try gradCheck(B, backend, allocator, "crossEntropy", pad, hp, loss, &.{1}, g, ctx);

        // Per-row losses average to the mean loss over the non-ignored rows.
        const rows = try backend.alloc(.f32, &.{4});
        defer backend.free(rows);
        try ctx.run();
        try backend.crossEntropyRows(rows, capped, targets);
        var row_losses: [4]f32 = undefined;
        try backend.download(rows, f32, &row_losses);
        var mean: [1]f32 = undefined;
        try backend.download(loss, f32, &mean);
        try std.testing.expectEqual(@as(f32, 0), row_losses[1]);
        try std.testing.expectApproxEqAbs(mean[0], (row_losses[0] + row_losses[2] + row_losses[3]) / 3, 1e-6);
        try backend.crossEntropyBackward(dpad, capped, targets, 3, 0.5);
        const half = try hostCopy(B, backend, allocator, dpad);
        defer allocator.free(half);
        for (half, g) |hv, gv| try std.testing.expectApproxEqAbs(gv * 0.5, hv, 1e-7);

        // dot, with accumulate
        const a = try tensorFrom(B, backend, f32, &.{3}, &.{ 1, 2, 3 });
        defer backend.free(a);
        const out = try tensorFrom(B, backend, f32, &.{2}, &.{ 10, 0 });
        defer backend.free(out);
        try backend.dot(out, a, a, -1, true);
        var host_out: [2]f32 = undefined;
        try backend.download(out, f32, &host_out);
        try std.testing.expectEqualSlices(f32, &.{ -4, 0 }, &host_out);

        // embeddingBackward with a repeated id
        const table = try backend.alloc(.f32, &.{ 3, 2 });
        defer backend.free(table);
        const ids = try tensorFrom(B, backend, i32, &.{3}, &.{ 2, 0, 2 });
        defer backend.free(ids);
        const grads = try tensorFrom(B, backend, f32, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
        defer backend.free(grads);
        try backend.embeddingBackward(table, grads, ids);
        try backend.embeddingBackward(table, grads, ids);
        var host_table: [6]f32 = undefined;
        try backend.download(table, f32, &host_table);
        try std.testing.expectEqualSlices(f32, &.{ 6, 8, 0, 0, 12, 16 }, &host_table);
    }

    /// adamwStep, muonMomentum, muonPrepare and muonFinish against f64 host references.
    fn optimizerOps(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        var rng = mod.Random.init(31);
        const n = 4 * 6;
        const hp = try randomHost(allocator, &rng, n, -1, 1);
        defer allocator.free(hp);
        const hg = try randomHost(allocator, &rng, n, -1, 1);
        defer allocator.free(hg);
        const hm = try randomHost(allocator, &rng, n, -0.1, 0.1);
        defer allocator.free(hm);
        const hv = try randomHost(allocator, &rng, n, 0, 0.1);
        defer allocator.free(hv);
        const p = try tensorFrom(B, backend, f32, &.{ 4, 6 }, hp);
        defer backend.free(p);
        const g = try tensorFrom(B, backend, f32, &.{ 4, 6 }, hg);
        defer backend.free(g);
        const m = try tensorFrom(B, backend, f32, &.{ 4, 6 }, hm);
        defer backend.free(m);
        const v = try tensorFrom(B, backend, f32, &.{ 4, 6 }, hv);
        defer backend.free(v);

        // AdamW, step 3.
        const params = mod.AdamWParams{ .lr = 0.1, .beta1 = 0.8, .beta2 = 0.95, .eps = 1e-8, .weight_decay = 0.05, .step = 3 };
        try backend.adamwStep(p, g, m, v, params);
        const got_p = try hostCopy(B, backend, allocator, p);
        defer allocator.free(got_p);
        for (got_p, hp, hg, hm, hv) |gp, p0, g0, m0, v0| {
            const m1 = m0 + 0.2 * (@as(f64, g0) - m0);
            const v1 = v0 + 0.05 * (@as(f64, g0) * g0 - v0);
            const denom = @sqrt(v1 / (1 - std.math.pow(f64, 0.95, 3))) + 1e-8;
            const want = p0 * (1 - 0.1 * 0.05) - 0.1 / (1 - std.math.pow(f64, 0.8, 3)) * m1 / denom;
            try std.testing.expectApproxEqAbs(want, @as(f64, gp), 1e-5);
        }

        // Nesterov momentum.
        try backend.upload(m, f32, hm);
        try backend.upload(g, f32, hg);
        try backend.muonMomentum(g, m, 0.9);
        const got_g = try hostCopy(B, backend, allocator, g);
        defer allocator.free(got_g);
        for (got_g, hg, hm) |gg, g0, m0| {
            const buf = m0 + 0.1 * (@as(f64, g0) - m0);
            try std.testing.expectApproxEqAbs(g0 + 0.9 * (buf - g0), @as(f64, gg), 1e-6);
        }

        // MuonEq + normalization: equal row norms, Frobenius norm 1 / 1.01.
        try backend.upload(g, f32, hg);
        try backend.muonPrepare(g);
        const prepped = try hostCopy(B, backend, allocator, g);
        defer allocator.free(prepped);
        var total: f64 = 0;
        var first_row: f64 = 0;
        for (0..4) |r| {
            var row: f64 = 0;
            for (prepped[r * 6 ..][0..6]) |x| row += @as(f64, x) * x;
            if (r == 0) first_row = row;
            try std.testing.expectApproxEqAbs(first_row, row, 1e-6);
            total += row;
        }
        try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 1.01), @sqrt(total), 1e-5);

        // Muon+ / NorMuon / cautious update on a wide [4, 6] matrix (per-column second moment).
        const second = try tensorFrom(B, backend, f32, &.{ 1, 6 }, &.{ 0.5, 0.1, 0.2, 0.3, 0.05, 1 });
        defer backend.free(second);
        try backend.upload(p, f32, hp);
        try backend.upload(g, f32, hg);
        try backend.muonFinish(p, g, second, .{ .lr = 0.05, .weight_decay = 0.1, .beta2 = 0.9 });
        const fin = try hostCopy(B, backend, allocator, p);
        defer allocator.free(fin);
        var want = try allocator.alloc(f64, n);
        defer allocator.free(want);
        var norm: f64 = 0;
        for (hg) |x| norm += @as(f64, x) * x;
        const renorm = @sqrt(4.0) / @sqrt(norm);
        var col_mean: [6]f64 = @splat(0);
        for (0..4) |r| for (0..6) |c| {
            const x = hg[r * 6 + c] * renorm;
            col_mean[c] += x * x / 4;
        };
        var v_norm_sq: f64 = 0;
        var new_sq: f64 = 0;
        const s0 = [_]f64{ 0.5, 0.1, 0.2, 0.3, 0.05, 1 };
        var step_size: [6]f64 = undefined;
        for (0..6) |c| {
            v_norm_sq += col_mean[c] * 4;
            const s = s0[c] + 0.1 * (col_mean[c] - s0[c]);
            step_size[c] = 1 / @sqrt(s);
            new_sq += col_mean[c] * 4 * step_size[c] * step_size[c];
        }
        const ratio = @sqrt(v_norm_sq) / @sqrt(new_sq);
        for (0..4) |r| for (0..6) |c| {
            const i = r * 6 + c;
            const upd = hg[i] * renorm * step_size[c] * ratio;
            const mask: f64 = if (upd * hp[i] >= 0) 1 else 0;
            want[i] = hp[i] - 0.05 * upd - 0.05 * 0.1 * hp[i] * mask;
        };
        for (want, fin) |w, f| try std.testing.expectApproxEqAbs(w, @as(f64, f), 1e-5);
    }

    /// upload/download preserve f32 and i32 data; dtype and size are checked.
    fn roundTrip(comptime B: type, backend: *B) !void {
        const f = try backend.alloc(.f32, &.{ 2, 2 });
        defer backend.free(f);
        try backend.upload(f, f32, &.{ 1.5, -2, 0, 3.25 });
        var fout: [4]f32 = undefined;
        try backend.download(f, f32, &fout);
        try std.testing.expectEqualSlices(f32, &.{ 1.5, -2, 0, 3.25 }, &fout);

        const ids = try backend.alloc(.i32, &.{3});
        defer backend.free(ids);
        try backend.upload(ids, i32, &.{ 7, -1, 32767 });
        var iout: [3]i32 = undefined;
        try backend.download(ids, i32, &iout);
        try std.testing.expectEqualSlices(i32, &.{ 7, -1, 32767 }, &iout);

        try std.testing.expectError(error.DtypeMismatch, backend.upload(ids, f32, &.{ 1, 2, 3 }));
        try std.testing.expectError(error.ShapeMismatch, backend.upload(f, f32, &.{ 1, 2 }));

        const zeroed = try backend.alloc(.f32, &.{5});
        defer backend.free(zeroed);
        var zout: [5]f32 = undefined;
        try backend.download(zeroed, f32, &zout);
        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0 }, &zout);
    }

    /// fill sets every element; copy moves data between equal-size tensors.
    fn fillAndCopy(comptime B: type, backend: *B) !void {
        const a = try backend.alloc(.f32, &.{ 2, 3 });
        defer backend.free(a);
        const b = try backend.alloc(.f32, &.{6});
        defer backend.free(b);
        try backend.fill(a, 0.75);
        try backend.copy(b, a);
        var out: [6]f32 = undefined;
        try backend.download(b, f32, &out);
        for (out) |v| try std.testing.expectEqual(@as(f32, 0.75), v);

        const short = try backend.alloc(.f32, &.{5});
        defer backend.free(short);
        try std.testing.expectError(error.ShapeMismatch, backend.copy(short, a));
    }

    /// add, mul, scale against host arithmetic, including in-place use.
    fn elementwise(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        const n = 1000;
        var rng = mod.Random.init(11);
        const xs = try allocator.alloc(f32, n);
        defer allocator.free(xs);
        const ys = try allocator.alloc(f32, n);
        defer allocator.free(ys);
        const out = try allocator.alloc(f32, n);
        defer allocator.free(out);
        rng.fillUniform(xs, -2, 2);
        rng.fillUniform(ys, -2, 2);

        const x = try backend.alloc(.f32, &.{ 10, 100 });
        defer backend.free(x);
        const y = try backend.alloc(.f32, &.{ 10, 100 });
        defer backend.free(y);
        const z = try backend.alloc(.f32, &.{ 10, 100 });
        defer backend.free(z);
        try backend.upload(x, f32, xs);
        try backend.upload(y, f32, ys);

        try backend.add(z, x, y);
        try backend.download(z, f32, out);
        for (out, xs, ys) |o, a, b| try std.testing.expectEqual(a + b, o);

        try backend.mul(z, x, y);
        try backend.download(z, f32, out);
        for (out, xs, ys) |o, a, b| try std.testing.expectEqual(a * b, o);

        try backend.scale(x, x, -3); // in place
        try backend.download(x, f32, out);
        for (out, xs) |o, a| try std.testing.expectEqual(a * -3, o);

        const wrong = try backend.alloc(.f32, &.{ 100, 10 });
        defer backend.free(wrong);
        try std.testing.expectError(error.ShapeMismatch, backend.add(z, x, wrong));
    }

    /// Ops on row views touch only the view's elements.
    fn views(comptime B: type, backend: *B) !void {
        const t = try backend.alloc(.f32, &.{ 4, 2 });
        defer backend.free(t);
        try backend.fill(try t.rows(1, 2), 9);
        var out: [8]f32 = undefined;
        try backend.download(t, f32, &out);
        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 9, 9, 9, 9, 0, 0 }, &out);
    }

    /// matmul in all four layouts, with alpha, accumulate and view operands.
    fn matmul(comptime B: type, backend: *B, allocator: std.mem.Allocator) !void {
        const cases = [_][3]usize{ .{ 1, 1, 1 }, .{ 3, 5, 7 }, .{ 64, 48, 300 }, .{ 130, 70, 33 } };
        var rng = mod.Random.init(5);
        for (cases) |case| {
            for ([_]bool{ false, true }) |ta| {
                for ([_]bool{ false, true }) |tb| {
                    const options = mod.MatmulOptions{ .transpose_a = ta, .transpose_b = tb, .alpha = 0.5, .accumulate = ta != tb };
                    try matmulCase(B, backend, allocator, &rng, case[0], case[1], case[2], options);
                }
            }
        }

        // Operands that are row views of larger tensors.
        const big = try backend.alloc(.f32, &.{ 10, 4 });
        defer backend.free(big);
        try backend.fill(big, 1);
        const c = try backend.alloc(.f32, &.{ 3, 3 });
        defer backend.free(c);
        try backend.matmul(c, try big.rows(2, 3), try big.rows(6, 3), .{ .transpose_b = true });
        var out: [9]f32 = undefined;
        try backend.download(c, f32, &out);
        for (out) |v| try std.testing.expectEqual(@as(f32, 4), v);

        try std.testing.expectError(error.ShapeMismatch, backend.matmul(c, big, big, .{}));
    }

    fn matmulCase(comptime B: type, backend: *B, allocator: std.mem.Allocator, rng: *mod.Random, m: usize, n: usize, k: usize, options: mod.MatmulOptions) !void {
        const ha = try allocator.alloc(f32, m * k);
        defer allocator.free(ha);
        const hb = try allocator.alloc(f32, k * n);
        defer allocator.free(hb);
        const hc = try allocator.alloc(f32, m * n);
        defer allocator.free(hc);
        rng.fillUniform(ha, -1, 1);
        rng.fillUniform(hb, -1, 1);
        rng.fillUniform(hc, -1, 1);

        const a = try backend.alloc(.f32, if (options.transpose_a) &.{ k, m } else &.{ m, k });
        defer backend.free(a);
        const b = try backend.alloc(.f32, if (options.transpose_b) &.{ n, k } else &.{ k, n });
        defer backend.free(b);
        const c = try backend.alloc(.f32, &.{ m, n });
        defer backend.free(c);
        try backend.upload(a, f32, ha);
        try backend.upload(b, f32, hb);
        try backend.upload(c, f32, hc);
        try backend.matmul(c, a, b, options);

        const got = try allocator.alloc(f32, m * n);
        defer allocator.free(got);
        try backend.download(c, f32, got);
        const tolerance = 1e-5 * @as(f64, @floatFromInt(k + 1));
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f64 = 0;
                for (0..k) |kk| {
                    const av = if (options.transpose_a) ha[kk * m + i] else ha[i * k + kk];
                    const bv = if (options.transpose_b) hb[j * k + kk] else hb[kk * n + j];
                    sum += @as(f64, av) * @as(f64, bv);
                }
                var expected = options.alpha * sum;
                if (options.accumulate) expected += hc[i * n + j];
                try std.testing.expectApproxEqAbs(expected, @as(f64, got[i * n + j]), tolerance);
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "conformance contract lists the members the build's backend has" {
    mod.Conformance.assertContract(mod.Backend);
    try std.testing.expect(mod.Conformance.required_decls.len > 0);
}
