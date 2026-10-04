const std = @import("std");
const log = std.log.scoped(.zignanogpt_cpu_backend);
const mod = @import("module.zig");

/// CPU storage: one 64-byte-aligned heap block (cache line and AVX-512 friendly).
pub const CpuBuffer = struct {
    bytes: []align(64) u8,
};

/// Elementwise ops split into chunks of this many elements per work item.
const chunk_len = 1 << 16;

/// The reference backend: host memory, SIMD kernels, `std.Io` thread pool.
///
/// Implements the backend contract (see `Conformance`). Every op runs to
/// completion before it returns, so `sync` is a no-op here; callers still call
/// it where a GPU backend would need it.
pub const CpuBackend = struct {
    const Self = @This();

    pub const Buffer = CpuBuffer;
    pub const name = "cpu";

    pub const Options = struct {
        /// Worker threads, the caller included; 0 means one per CPU.
        threads: usize = 0,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    parallel: mod.Parallel,
    /// Matmul packing scratch, `CpuMatmul.scratch_len` floats per worker.
    scratch: []f32,

    /// Creates the backend.
    ///
    /// Parameters:
    /// - `allocator`: owns every buffer and the scratch.
    /// - `io`: runs the worker threads.
    /// - `options`: thread count.
    ///
    /// Return: the backend; allocation errors.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, options: Options) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const parallel = mod.Parallel.init(io, options.threads);
        const scratch = try allocator.alloc(f32, parallel.threads * mod.CpuMatmul.scratch_len);
        return Self{ .allocator = allocator, .io = io, .parallel = parallel, .scratch = scratch };
    }

    /// Frees the scratch. Tensors must be freed by their owners first.
    ///
    /// Parameters:
    /// - `self`: the backend.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.scratch);
    }

    /// Allocates a zero-filled tensor.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dtype`: the element type.
    /// - `dims`: the shape.
    ///
    /// Return: the tensor, owned by the caller (`free`).
    pub fn alloc(self: *Self, dtype: mod.Dtype, dims: []const usize) !mod.Tensor {
        const shape = try mod.Shape.init(dims);
        const bytes = try self.allocator.alignedAlloc(u8, .@"64", shape.numel() * dtype.size());
        @memset(bytes, 0);
        return mod.Tensor{ .buffer = .{ .bytes = bytes }, .dtype = dtype, .shape = shape };
    }

    /// Frees a tensor returned by `alloc` (not a view of one).
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `tensor`: the tensor.
    ///
    /// Return: nothing.
    pub fn free(self: *Self, tensor: mod.Tensor) void {
        self.allocator.free(tensor.buffer.bytes);
    }

    /// Waits for queued work. A no-op: CPU ops complete before returning.
    ///
    /// Parameters:
    /// - `self`: the backend.
    ///
    /// Return: nothing.
    pub fn sync(self: *Self) !void {
        _ = self;
    }

    /// Copies host data into a tensor.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `tensor`: the destination.
    /// - `T`: the element type; must match `tensor.dtype`.
    /// - `data`: exactly `tensor.numel()` elements.
    ///
    /// Return: nothing; `error.DtypeMismatch` or `error.ShapeMismatch`.
    pub fn upload(self: *Self, tensor: mod.Tensor, comptime T: type, data: []const T) !void {
        _ = self;
        if (tensor.dtype != mod.Dtype.of(T)) return error.DtypeMismatch;
        if (data.len != tensor.numel()) return error.ShapeMismatch;
        @memcpy(elems(T, tensor), data);
    }

    /// Copies a tensor out to host memory.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `tensor`: the source.
    /// - `T`: the element type; must match `tensor.dtype`.
    /// - `out`: exactly `tensor.numel()` elements.
    ///
    /// Return: nothing; `error.DtypeMismatch` or `error.ShapeMismatch`.
    pub fn download(self: *Self, tensor: mod.Tensor, comptime T: type, out: []T) !void {
        try self.sync();
        if (tensor.dtype != mod.Dtype.of(T)) return error.DtypeMismatch;
        if (out.len != tensor.numel()) return error.ShapeMismatch;
        @memcpy(out, elems(T, tensor));
    }

    /// Sets every element of an f32 tensor.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `tensor`: the destination.
    /// - `value`: the value.
    ///
    /// Return: nothing; `error.DtypeMismatch` for a non-f32 tensor.
    pub fn fill(self: *Self, tensor: mod.Tensor, value: f32) !void {
        _ = self;
        if (tensor.dtype != .f32) return error.DtypeMismatch;
        @memset(elems(f32, tensor), value);
    }

    /// Copies `src` into `dst` element for element (any shapes of equal size).
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dst`: the destination.
    /// - `src`: the source; same dtype and element count.
    ///
    /// Return: nothing; `error.DtypeMismatch` or `error.ShapeMismatch`.
    pub fn copy(self: *Self, dst: mod.Tensor, src: mod.Tensor) !void {
        _ = self;
        if (dst.dtype != src.dtype) return error.DtypeMismatch;
        if (dst.numel() != src.numel()) return error.ShapeMismatch;
        const n = dst.numel() * dst.dtype.size();
        const to = dst.buffer.bytes[dst.offset * dst.dtype.size() ..][0..n];
        const from = src.buffer.bytes[src.offset * src.dtype.size() ..][0..n];
        @memmove(to, from);
    }

    /// `out = a + b`, elementwise; `out` may alias an input.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: the result.
    /// - `a`: an operand.
    /// - `b`: an operand.
    ///
    /// Return: nothing; shape or dtype mismatch errors.
    pub fn add(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try self.binary(out, a, b, .add);
    }

    /// `out = a * b`, elementwise; `out` may alias an input.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: the result.
    /// - `a`: an operand.
    /// - `b`: an operand.
    ///
    /// Return: nothing; shape or dtype mismatch errors.
    pub fn mul(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try self.binary(out, a, b, .mul);
    }

    /// `out = a * s`; `out` may alias `a`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: the result.
    /// - `a`: the operand.
    /// - `s`: the scalar.
    ///
    /// Return: nothing; shape or dtype mismatch errors.
    pub fn scale(self: *Self, out: mod.Tensor, a: mod.Tensor, s: f32) !void {
        try checkSame(out, a);
        const Ctx = struct {
            out: []f32,
            a: []const f32,
            s: f32,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                const start = item * chunk_len;
                const end = @min(start + chunk_len, ctx.out.len);
                for (ctx.out[start..end], ctx.a[start..end]) |*o, x| o.* = x * ctx.s;
            }
        };
        try self.parallel.run(chunks(out.numel()), Ctx{ .out = elems(f32, out), .a = elems(f32, a), .s = s }, Ctx.work);
    }

    /// `c (+)= alpha * op(a) @ op(b)`; see `MatmulOptions`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `c`: the result, viewed as `[M, N]`; must not alias `a` or `b`.
    /// - `a`: viewed as `[M, K]`, or `[K, M]` with `transpose_a`.
    /// - `b`: viewed as `[K, N]`, or `[N, K]` with `transpose_b`.
    /// - `options`: transposes, scale, accumulate.
    ///
    /// Return: nothing; `error.ShapeMismatch` or `error.DtypeMismatch`.
    pub fn matmul(self: *Self, c: mod.Tensor, a: mod.Tensor, b: mod.Tensor, options: mod.MatmulOptions) !void {
        if (c.dtype != .f32 or a.dtype != .f32 or b.dtype != .f32) return error.DtypeMismatch;
        const m, const n, const k = try options.dims(c.shape, a.shape, b.shape);
        try checkDisjoint(c, &.{ a, b });
        const problem = mod.CpuMatmul{
            .c = elems(f32, c),
            .a = elems(f32, a),
            .b = elems(f32, b),
            .m = m,
            .n = n,
            .k = k,
            .options = options,
            .scratch = self.scratch,
        };
        try problem.run(&self.parallel);
    }

    /// `out[r, :] = table[ids[r], :]`: the embedding lookup.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: `[R, C]` (any shape with `R * C` elements, `C` = table cols).
    /// - `table`: `[V, C]`.
    /// - `ids`: `R` i32 indices below `V`.
    ///
    /// Return: nothing; shape/dtype errors, `error.OutOfBounds` for an id outside the table.
    pub fn embedding(self: *Self, out: mod.Tensor, table: mod.Tensor, ids: mod.Tensor) !void {
        if (out.dtype != .f32 or table.dtype != .f32 or ids.dtype != .i32) return error.DtypeMismatch;
        const cols = table.shape.cols();
        if (out.numel() != ids.numel() * cols) return mismatch("embedding", out.shape, table.shape);
        const vocab = table.shape.rows();
        const id_slice = elems(i32, ids);
        for (id_slice) |id| {
            if (id < 0 or id >= vocab) {
                log.debug("embedding id {d} outside vocab {d}", .{ id, vocab });
                return error.OutOfBounds;
            }
        }
        const Ctx = struct {
            out: []f32,
            table: []const f32,
            ids: []const i32,
            cols: usize,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const id: usize = @intCast(ctx.ids[r]);
                    @memcpy(ctx.out[r * ctx.cols ..][0..ctx.cols], ctx.table[id * ctx.cols ..][0..ctx.cols]);
                }
            }
        };
        try self.forRows(id_slice.len, cols, Ctx{ .out = elems(f32, out), .table = elems(f32, table), .ids = id_slice, .cols = cols }, Ctx.body);
    }

    /// RMS norm over the last dimension, no weight: `out = x / sqrt(mean(x^2) + eps)`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: same shape as `x`; may alias it.
    /// - `x`: the input, normalized row by row.
    /// - `eps`: added to the mean square (nanochat: f32 machine epsilon).
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn rmsnorm(self: *Self, out: mod.Tensor, x: mod.Tensor, eps: f32) !void {
        try checkSame(out, x);
        const Ctx = struct {
            out: []f32,
            x: []const f32,
            cols: usize,
            eps: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const row = ctx.x[r * ctx.cols ..][0..ctx.cols];
                    var sum: f32 = 0;
                    for (row) |v| sum += v * v;
                    const inv = 1 / @sqrt(sum / @as(f32, @floatFromInt(ctx.cols)) + ctx.eps);
                    for (ctx.out[r * ctx.cols ..][0..ctx.cols], row) |*o, v| o.* = v * inv;
                }
            }
        };
        const cols = x.shape.cols();
        try self.forRows(x.shape.rows(), cols, Ctx{ .out = elems(f32, out), .x = elems(f32, x), .cols = cols, .eps = eps }, Ctx.body);
    }

    /// Rotary embedding (nanochat's convention, rotating by -theta): for each
    /// `[B, T, H, D]` row at position `pos0 + t`, with halves `x1, x2`:
    /// `out1 = x1 * cos + x2 * sin`, `out2 = x2 * cos - x1 * sin`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: same shape as `x`; may alias it.
    /// - `x`: `[B, T, H, D]`, `D` even.
    /// - `cos`: `[Tmax, D / 2]`.
    /// - `sin`: `[Tmax, D / 2]`.
    /// - `pos0`: the position of `t = 0`.
    ///
    /// Return: nothing; shape/dtype errors, `error.OutOfBounds` past `Tmax`.
    pub fn rope(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        try self.rotate(out, x, cos, sin, pos0, 1);
    }

    /// Backward of `rope`: the inverse rotation, `dx1 = dy1 * cos - dy2 * sin`,
    /// `dx2 = dy2 * cos + dy1 * sin`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dx`: same shape as `dy`; may alias it.
    /// - `dy`: the output gradient, `[B, T, H, D]`.
    /// - `cos`: `[Tmax, D / 2]`.
    /// - `sin`: `[Tmax, D / 2]`.
    /// - `pos0`: the position of `t = 0`.
    ///
    /// Return: nothing; as `rope`.
    pub fn ropeBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        try self.rotate(dx, dy, cos, sin, pos0, -1);
    }

    /// Shared body of `rope` (`sign = 1`) and its inverse (`sign = -1`).
    fn rotate(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize, sign: f32) !void {
        try checkSame(out, x);
        try checkSame(cos, sin);
        if (x.shape.rank != 4 or x.shape.dims[3] % 2 != 0 or cos.shape.cols() * 2 != x.shape.dims[3]) {
            return mismatch("rope", x.shape, cos.shape);
        }
        if (pos0 + x.shape.dims[1] > cos.shape.rows()) {
            log.debug("rope positions {d}..{d} exceed the table's {d}", .{ pos0, pos0 + x.shape.dims[1], cos.shape.rows() });
            return error.OutOfBounds;
        }
        const Ctx = struct {
            out: []f32,
            x: []const f32,
            cos: []const f32,
            sin: []const f32,
            t: usize,
            h: usize,
            half: usize,
            pos0: usize,
            sign: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    // rows are (b, t, h); the position is t
                    const pos = ctx.pos0 + (r / ctx.h) % ctx.t;
                    const c = ctx.cos[pos * ctx.half ..][0..ctx.half];
                    const s = ctx.sin[pos * ctx.half ..][0..ctx.half];
                    const base = r * 2 * ctx.half;
                    for (0..ctx.half) |i| {
                        const x1 = ctx.x[base + i];
                        const x2 = ctx.x[base + ctx.half + i];
                        const si = ctx.sign * s[i];
                        ctx.out[base + i] = x1 * c[i] + x2 * si;
                        ctx.out[base + ctx.half + i] = x2 * c[i] - x1 * si;
                    }
                }
            }
        };
        const ctx = Ctx{
            .out = elems(f32, out),
            .x = elems(f32, x),
            .cos = elems(f32, cos),
            .sin = elems(f32, sin),
            .t = x.shape.dims[1],
            .h = x.shape.dims[2],
            .half = x.shape.dims[3] / 2,
            .pos0 = pos0,
            .sign = sign,
        };
        try self.forRows(x.shape.dims[0] * ctx.t * ctx.h, x.shape.dims[3], ctx, Ctx.body);
    }

    /// Causal sliding-window GQA attention; see `AttentionOptions`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: `[B, Tq, H, D]`; must not alias the inputs.
    /// - `q`: `[B, Tq, H, D]`.
    /// - `k`: `[B, Tk, Hkv, D]`.
    /// - `v`: same shape as `k`.
    /// - `lse`: optional `[B, H, Tq]` output of each query's log-sum-exp.
    /// - `options`: window and valid key count.
    ///
    /// Return: nothing; shape/dtype/aliasing errors.
    pub fn attention(self: *Self, out: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, lse: ?mod.Tensor, options: mod.AttentionOptions) !void {
        try checkSame(out, q);
        try checkSame(k, v);
        const dims = try options.dims(out.shape, q.shape, k.shape, v.shape, if (lse) |t| t.shape else null);
        if (lse) |t| if (t.dtype != .f32) return error.DtypeMismatch;
        try checkDisjoint(out, &.{ q, k, v });
        const problem = mod.CpuAttention{
            .out = elems(f32, out),
            .q = elems(f32, q),
            .k = elems(f32, k),
            .v = elems(f32, v),
            .lse = if (lse) |t| elems(f32, t) else null,
            .dims = dims,
            .window = options.window,
        };
        try problem.run(&self.parallel);
    }

    /// `out = max(x, 0)^2`, elementwise; may alias.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: the result.
    /// - `x`: the input.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn reluSquare(self: *Self, out: mod.Tensor, x: mod.Tensor) !void {
        try checkSame(out, x);
        const Ctx = struct {
            out: []f32,
            x: []const f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (ctx.out[start..end], ctx.x[start..end]) |*o, v| {
                    const r = @max(v, 0);
                    o.* = r * r;
                }
            }
        };
        try self.forRows(x.numel(), 1, Ctx{ .out = elems(f32, out), .x = elems(f32, x) }, Ctx.body);
    }

    /// `out = xs * x + ys * y`, elementwise with device scalars; may alias.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: the result.
    /// - `x`: an operand.
    /// - `xs`: its scale.
    /// - `y`: an operand, same shape.
    /// - `ys`: its scale.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn combine(self: *Self, out: mod.Tensor, x: mod.Tensor, xs: mod.Scalar, y: mod.Tensor, ys: mod.Scalar) !void {
        try checkSame(out, x);
        try checkSame(out, y);
        try xs.validate();
        try ys.validate();
        const Ctx = struct {
            out: []f32,
            x: []const f32,
            y: []const f32,
            a: f32,
            b: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (ctx.out[start..end], ctx.x[start..end], ctx.y[start..end]) |*o, u, w| o.* = ctx.a * u + ctx.b * w;
            }
        };
        const ctx = Ctx{ .out = elems(f32, out), .x = elems(f32, x), .y = elems(f32, y), .a = resolve(xs), .b = resolve(ys) };
        try self.forRows(x.numel(), 1, ctx, Ctx.body);
    }

    /// A linear map of each row's first `cin` channels: `out[r, h] = sum_{c < cin} x[r, c] * w[h, c]`.
    /// nanochat's smear gate (`cin = 24`) and value-embedding gate (`cin = 12`).
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: `[R, H]`.
    /// - `x`: `[R, C]` with `C >= cin`.
    /// - `w`: `[H, cin]`.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn gateLinear(self: *Self, out: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        if (out.dtype != .f32 or x.dtype != .f32 or w.dtype != .f32) return error.DtypeMismatch;
        const heads = w.shape.rows();
        const cin = w.shape.cols();
        if (cin > x.shape.cols() or out.shape.rows() != x.shape.rows() or out.shape.cols() != heads) {
            return mismatch("gateLinear", x.shape, w.shape);
        }
        const Ctx = struct {
            out: []f32,
            x: []const f32,
            w: []const f32,
            cols: usize,
            cin: usize,
            heads: usize,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const row = ctx.x[r * ctx.cols ..][0..ctx.cin];
                    for (0..ctx.heads) |h| {
                        var sum: f32 = 0;
                        for (row, ctx.w[h * ctx.cin ..][0..ctx.cin]) |a, b| sum += a * b;
                        ctx.out[r * ctx.heads + h] = sum;
                    }
                }
            }
        };
        const ctx = Ctx{ .out = elems(f32, out), .x = elems(f32, x), .w = elems(f32, w), .cols = x.shape.cols(), .cin = cin, .heads = heads };
        try self.forRows(x.shape.rows(), cin * heads, ctx, Ctx.body);
    }

    /// nanochat's smear: mix the previous token into each position,
    /// `out[b, t] = x[b, t] + lambda * sigmoid(gate[b, t]) * x[b, t - 1]` for
    /// `t >= 1`, `out[b, 0] = x[b, 0]`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: `[B, T, C]`; must not alias `x`.
    /// - `x`: `[B, T, C]`.
    /// - `gate`: `B * T` gate logits (e.g. `[B * T, 1]`).
    /// - `lambda`: the mixing strength.
    ///
    /// Return: nothing; shape/dtype/aliasing errors.
    pub fn smear(self: *Self, out: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        try checkSame(out, x);
        try lambda.validate();
        if (gate.dtype != .f32) return error.DtypeMismatch;
        if (x.shape.rank != 3 or gate.numel() != x.shape.dims[0] * x.shape.dims[1]) return mismatch("smear", x.shape, gate.shape);
        try checkDisjoint(out, &.{x});
        const Ctx = struct {
            out: []f32,
            x: []const f32,
            gate: []const f32,
            lambda: f32,
            t: usize,
            cols: usize,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const row = ctx.x[r * ctx.cols ..][0..ctx.cols];
                    const dst = ctx.out[r * ctx.cols ..][0..ctx.cols];
                    if (r % ctx.t == 0) {
                        @memcpy(dst, row);
                        continue;
                    }
                    const g = ctx.lambda * sigmoid(ctx.gate[r]);
                    for (dst, row, ctx.x[(r - 1) * ctx.cols ..][0..ctx.cols]) |*o, cur, prev| o.* = cur + g * prev;
                }
            }
        };
        const cols = x.shape.dims[2];
        const ctx = Ctx{ .out = elems(f32, out), .x = elems(f32, x), .gate = elems(f32, gate), .lambda = resolve(lambda), .t = x.shape.dims[1], .cols = cols };
        try self.forRows(x.shape.rows(), cols, ctx, Ctx.body);
    }

    /// nanochat's value residual: `v[r, h, :] += 3 * sigmoid(gate[r, h]) * ve[r, h, :]`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `v`: `R * H * D` values, updated in place.
    /// - `ve`: the value embeddings, same element count as `v`.
    /// - `gate`: `[R, H]` gate logits.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn valueMix(self: *Self, v: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        if (v.dtype != .f32 or ve.dtype != .f32 or gate.dtype != .f32) return error.DtypeMismatch;
        const rows = gate.shape.rows();
        const heads = gate.shape.cols();
        if (v.numel() != ve.numel() or rows * heads == 0 or v.numel() % (rows * heads) != 0) return mismatch("valueMix", v.shape, gate.shape);
        const Ctx = struct {
            v: []f32,
            ve: []const f32,
            gate: []const f32,
            d: usize,
            fn body(ctx: @This(), start: usize, end: usize) void {
                // one item per (r, h) pair
                for (start..end) |rh| {
                    const g = 3 * sigmoid(ctx.gate[rh]);
                    for (ctx.v[rh * ctx.d ..][0..ctx.d], ctx.ve[rh * ctx.d ..][0..ctx.d]) |*o, e| o.* += g * e;
                }
            }
        };
        const d = v.numel() / (rows * heads);
        try self.forRows(rows * heads, d, Ctx{ .v = elems(f32, v), .ve = elems(f32, ve), .gate = elems(f32, gate), .d = d }, Ctx.body);
    }

    /// Crops padded logits to the vocabulary and squashes them:
    /// `out[r, c] = cap * tanh(logits[r, c] / cap)` for `c < V`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: `[R, V]`; must not alias `logits`.
    /// - `logits`: `[R, Vpad]`, `Vpad >= V`.
    /// - `cap`: the soft cap (nanochat: 15).
    ///
    /// Return: nothing; shape/dtype/aliasing errors.
    pub fn softcap(self: *Self, out: mod.Tensor, logits: mod.Tensor, cap: f32) !void {
        if (out.dtype != .f32 or logits.dtype != .f32) return error.DtypeMismatch;
        if (out.shape.rows() != logits.shape.rows() or out.shape.cols() > logits.shape.cols()) return mismatch("softcap", out.shape, logits.shape);
        try checkDisjoint(out, &.{logits});
        const Ctx = struct {
            out: []f32,
            logits: []const f32,
            cols: usize,
            padded: usize,
            cap: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const src = ctx.logits[r * ctx.padded ..][0..ctx.cols];
                    for (ctx.out[r * ctx.cols ..][0..ctx.cols], src) |*o, v| o.* = ctx.cap * std.math.tanh(v / ctx.cap);
                }
            }
        };
        const cols = out.shape.cols();
        const ctx = Ctx{ .out = elems(f32, out), .logits = elems(f32, logits), .cols = cols, .padded = logits.shape.cols(), .cap = cap };
        try self.forRows(out.shape.rows(), cols, ctx, Ctx.body);
    }

    /// Mean cross-entropy over the rows whose target is not -1 (PyTorch's
    /// `ignore_index=-1`, `reduction='mean'`); NaN when every row is ignored.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `loss`: receives the mean in element 0.
    /// - `logits`: `[R, V]`.
    /// - `targets`: `R` i32 classes below `V`, or -1.
    ///
    /// Return: nothing; shape/dtype errors, `error.OutOfBounds` for a bad target.
    pub fn crossEntropy(self: *Self, loss: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor) !void {
        const rows, const vocab = try self.checkTargets(logits, targets);
        if (loss.dtype != .f32 or loss.numel() == 0) return error.DtypeMismatch;
        const per = rowsPerItem(vocab);
        const items = (rows + per - 1) / per;
        const partial = try self.allocator.alloc(f64, items);
        defer self.allocator.free(partial);
        const Ctx = struct {
            logits: []const f32,
            targets: []const i32,
            partial: []f64,
            rows: usize,
            vocab: usize,
            per: usize,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                var sum: f64 = 0;
                for (item * ctx.per..@min((item + 1) * ctx.per, ctx.rows)) |r| {
                    if (ctx.targets[r] < 0) continue;
                    const row = ctx.logits[r * ctx.vocab ..][0..ctx.vocab];
                    sum += logSumExp(row) - row[@intCast(ctx.targets[r])];
                }
                ctx.partial[item] = sum;
            }
        };
        const ctx = Ctx{ .logits = elems(f32, logits), .targets = elems(i32, targets), .partial = partial, .rows = rows, .vocab = vocab, .per = per };
        try self.parallel.run(items, ctx, Ctx.work);
        var total: f64 = 0;
        for (partial) |p| total += p;
        const count: f64 = @floatFromInt(validTargets(elems(i32, targets)));
        elems(f32, loss)[0] = @floatCast(total / count);
    }

    /// Gradient of `scale * crossEntropy(softcap(logits_pad))` with respect to
    /// the padded, uncapped logits: `(softmax - onehot) * scale / count`, times
    /// the soft cap's derivative `1 - (z / cap)^2`; ignored rows and padding
    /// columns get zero.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dpad`: `[R, Vpad]` gradient output; must not alias `logits`.
    /// - `logits`: `[R, V]`, the capped logits the loss saw.
    /// - `targets`: `R` i32 classes, or -1.
    /// - `cap`: the soft cap used in the forward pass.
    /// - `scale`: multiplies the gradient (e.g. `1 / grad_accum_steps`).
    ///
    /// Return: nothing; shape/dtype/aliasing errors, `error.OutOfBounds`.
    pub fn crossEntropyBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, cap: f32, scale_by: f32) !void {
        const rows, const vocab = try self.checkTargets(logits, targets);
        if (dpad.dtype != .f32) return error.DtypeMismatch;
        if (dpad.shape.rows() != rows or dpad.shape.cols() < vocab) return mismatch("crossEntropyBackward", dpad.shape, logits.shape);
        try checkDisjoint(dpad, &.{logits});
        const count: f32 = @floatFromInt(validTargets(elems(i32, targets)));
        const Ctx = struct {
            dpad: []f32,
            logits: []const f32,
            targets: []const i32,
            vocab: usize,
            padded: usize,
            cap: f32,
            factor: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const out = ctx.dpad[r * ctx.padded ..][0..ctx.padded];
                    @memset(out[ctx.vocab..], 0);
                    if (ctx.targets[r] < 0) {
                        @memset(out[0..ctx.vocab], 0);
                        continue;
                    }
                    const row = ctx.logits[r * ctx.vocab ..][0..ctx.vocab];
                    const lse: f32 = @floatCast(logSumExp(row));
                    const target: usize = @intCast(ctx.targets[r]);
                    for (out[0..ctx.vocab], row, 0..) |*o, z, c| {
                        const p = @exp(z - lse) - @as(f32, if (c == target) 1 else 0);
                        const squash = z / ctx.cap;
                        o.* = p * ctx.factor * (1 - squash * squash);
                    }
                }
            }
        };
        const ctx = Ctx{ .dpad = elems(f32, dpad), .logits = elems(f32, logits), .targets = elems(i32, targets), .vocab = vocab, .padded = dpad.shape.cols(), .cap = cap, .factor = scale_by / count };
        try self.forRows(rows, dpad.shape.cols(), ctx, Ctx.body);
    }

    /// Backward of `rmsnorm`: with `r = 1 / sqrt(mean(x^2) + eps)`,
    /// `dx (+)= r * dy - x * r^3 * mean(dy * x)`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dx`: same shape as `x`; may alias `dy`.
    /// - `dy`: the output gradient.
    /// - `x`: the forward input.
    /// - `eps`: as in the forward pass.
    /// - `accumulate`: add into `dx` instead of overwriting it.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn rmsnormBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor, eps: f32, accumulate: bool) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        const Ctx = struct {
            dx: []f32,
            dy: []const f32,
            x: []const f32,
            cols: usize,
            eps: f32,
            accumulate: bool,
            fn body(ctx: @This(), start: usize, end: usize) void {
                const n: f32 = @floatFromInt(ctx.cols);
                for (start..end) |r| {
                    const xr = ctx.x[r * ctx.cols ..][0..ctx.cols];
                    const gr = ctx.dy[r * ctx.cols ..][0..ctx.cols];
                    var sq: f32 = 0;
                    var dot_gx: f32 = 0;
                    for (xr, gr) |v, g| {
                        sq += v * v;
                        dot_gx += g * v;
                    }
                    const inv = 1 / @sqrt(sq / n + ctx.eps);
                    const k = inv * inv * inv * dot_gx / n;
                    for (ctx.dx[r * ctx.cols ..][0..ctx.cols], xr, gr) |*o, v, g| {
                        const d = inv * g - v * k;
                        o.* = if (ctx.accumulate) o.* + d else d;
                    }
                }
            }
        };
        const cols = x.shape.cols();
        const ctx = Ctx{ .dx = elems(f32, dx), .dy = elems(f32, dy), .x = elems(f32, x), .cols = cols, .eps = eps, .accumulate = accumulate };
        try self.forRows(x.shape.rows(), cols, ctx, Ctx.body);
    }

    /// Backward of `attention`: `dq`, `dk`, `dv` from the output gradient,
    /// recomputing the probabilities from the saved log-sum-exp.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dq`: `[B, Tq, H, D]`, overwritten.
    /// - `dk`: `[B, Tk, Hkv, D]`, overwritten.
    /// - `dv`: `[B, Tk, Hkv, D]`, overwritten.
    /// - `dout`: the output gradient, `[B, Tq, H, D]`.
    /// - `q`, `k`, `v`: the forward inputs.
    /// - `out`: the forward output.
    /// - `lse`: the forward pass's `[B, H, Tq]` log-sum-exp.
    /// - `options`: as in the forward pass.
    ///
    /// Return: nothing; shape/dtype/aliasing errors.
    pub fn attentionBackward(self: *Self, dq: mod.Tensor, dk: mod.Tensor, dv: mod.Tensor, dout: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, out: mod.Tensor, lse: mod.Tensor, options: mod.AttentionOptions) !void {
        try checkSame(dq, q);
        try checkSame(dout, q);
        try checkSame(out, q);
        try checkSame(dk, k);
        try checkSame(dv, k);
        try checkSame(k, v);
        if (lse.dtype != .f32) return error.DtypeMismatch;
        const dims = try options.dims(out.shape, q.shape, k.shape, v.shape, lse.shape);
        const inputs = [_]mod.Tensor{ dout, q, k, v, out, lse };
        try checkDisjoint(dq, &inputs);
        try checkDisjoint(dk, &inputs);
        try checkDisjoint(dv, &inputs);
        const problem = mod.CpuAttentionBackward{
            .dq = elems(f32, dq),
            .dk = elems(f32, dk),
            .dv = elems(f32, dv),
            .dout = elems(f32, dout),
            .q = elems(f32, q),
            .k = elems(f32, k),
            .v = elems(f32, v),
            .out = elems(f32, out),
            .lse = elems(f32, lse),
            .dims = dims,
            .window = options.window,
        };
        try problem.run(&self.parallel);
    }

    /// Backward of `reluSquare`: `dx = dy * 2 * max(x, 0)`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dx`: same shape as `x`; may alias `dy`.
    /// - `dy`: the output gradient.
    /// - `x`: the forward input.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn reluSquareBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        const Ctx = struct {
            dx: []f32,
            dy: []const f32,
            x: []const f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (ctx.dx[start..end], ctx.dy[start..end], ctx.x[start..end]) |*o, g, v| o.* = g * 2 * @max(v, 0);
            }
        };
        try self.forRows(x.numel(), 1, Ctx{ .dx = elems(f32, dx), .dy = elems(f32, dy), .x = elems(f32, x) }, Ctx.body);
    }

    /// `out[0] (+)= factor * sum(a * b)`: the gradient of a scalar that
    /// multiplied `b` (e.g. `resid_lambdas[i]`), accumulated in f64.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `out`: a tensor whose element 0 receives the result.
    /// - `a`: an operand.
    /// - `b`: an operand, same shape.
    /// - `factor`: multiplies the sum.
    /// - `accumulate`: add into `out[0]` instead of overwriting it.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn dot(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor, factor: f32, accumulate: bool) !void {
        try checkSame(a, b);
        if (out.dtype != .f32 or out.numel() == 0) return error.DtypeMismatch;
        const n = a.numel();
        const items = chunks(n);
        const partial = try self.allocator.alloc(f64, items);
        defer self.allocator.free(partial);
        const Ctx = struct {
            a: []const f32,
            b: []const f32,
            partial: []f64,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                const start = item * chunk_len;
                const end = @min(start + chunk_len, ctx.a.len);
                var sum: f64 = 0;
                for (ctx.a[start..end], ctx.b[start..end]) |x, y| sum += @as(f64, x) * @as(f64, y);
                ctx.partial[item] = sum;
            }
        };
        try self.parallel.run(items, Ctx{ .a = elems(f32, a), .b = elems(f32, b), .partial = partial }, Ctx.work);
        var total: f64 = 0;
        for (partial) |p| total += p;
        const dst = &elems(f32, out)[0];
        const value: f32 = @floatCast(total * factor);
        dst.* = if (accumulate) dst.* + value else value;
    }

    /// Backward of `gateLinear`: `dx[:, :cin] += dout @ w`, `dw += dout^T @ x[:, :cin]`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dx`: same shape as `x`; accumulated into (first `cin` channels).
    /// - `dw`: same shape as `w`; accumulated into.
    /// - `dout`: `[R, H]`.
    /// - `x`: the forward input `[R, C]`.
    /// - `w`: `[H, cin]`.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn gateLinearBackward(self: *Self, dx: mod.Tensor, dw: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        try checkSame(dx, x);
        try checkSame(dw, w);
        if (dout.dtype != .f32) return error.DtypeMismatch;
        const rows = x.shape.rows();
        const cols = x.shape.cols();
        const heads = w.shape.rows();
        const cin = w.shape.cols();
        if (cin > cols or dout.shape.rows() != rows or dout.shape.cols() != heads) return mismatch("gateLinearBackward", x.shape, w.shape);
        const RowCtx = struct {
            dx: []f32,
            dout: []const f32,
            w: []const f32,
            cols: usize,
            cin: usize,
            heads: usize,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |r| {
                    const g = ctx.dout[r * ctx.heads ..][0..ctx.heads];
                    for (ctx.dx[r * ctx.cols ..][0..ctx.cin], 0..) |*o, c| {
                        var sum: f32 = 0;
                        for (g, 0..) |gh, h| sum += gh * ctx.w[h * ctx.cin + c];
                        o.* += sum;
                    }
                }
            }
        };
        try self.forRows(rows, cin * heads, RowCtx{ .dx = elems(f32, dx), .dout = elems(f32, dout), .w = elems(f32, w), .cols = cols, .cin = cin, .heads = heads }, RowCtx.body);
        // dw: one item per weight, each summing over every row in order.
        const WeightCtx = struct {
            dw: []f32,
            dout: []const f32,
            x: []const f32,
            rows: usize,
            cols: usize,
            cin: usize,
            heads: usize,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                const h = item / ctx.cin;
                const c = item % ctx.cin;
                var sum: f32 = 0;
                for (0..ctx.rows) |r| sum += ctx.dout[r * ctx.heads + h] * ctx.x[r * ctx.cols + c];
                ctx.dw[item] += sum;
            }
        };
        try self.parallel.run(heads * cin, WeightCtx{ .dw = elems(f32, dw), .dout = elems(f32, dout), .x = elems(f32, x), .rows = rows, .cols = cols, .cin = cin, .heads = heads }, WeightCtx.work);
    }

    /// Backward of `smear`. With `s_t = sigmoid(gate[t])`:
    /// `dx[t] = dout[t] + lambda * s_{t+1} * dout[t+1]` (within a batch row),
    /// `dgate[t] = lambda * s_t * (1 - s_t) * <dout[t], x[t-1]>` (0 at `t = 0`),
    /// `dlambda += sum_t s_t * <dout[t], x[t-1]>`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dx`: `[B, T, C]`, overwritten; must not alias the inputs.
    /// - `dgate`: `B * T` gate gradients, overwritten.
    /// - `dlambda`: element 0 accumulates the lambda gradient.
    /// - `dout`: `[B, T, C]` output gradient.
    /// - `x`: the forward input.
    /// - `gate`: the forward gate logits.
    /// - `lambda`: the forward mixing strength.
    ///
    /// Return: nothing; shape/dtype/aliasing errors.
    pub fn smearBackward(self: *Self, dx: mod.Tensor, dgate: mod.Tensor, dlambda: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        try checkSame(dx, x);
        try checkSame(dout, x);
        try checkSame(dgate, gate);
        try lambda.validate();
        if (dlambda.dtype != .f32 or dlambda.numel() == 0) return error.DtypeMismatch;
        if (x.shape.rank != 3 or gate.numel() != x.shape.dims[0] * x.shape.dims[1]) return mismatch("smearBackward", x.shape, gate.shape);
        try checkDisjoint(dx, &.{ dout, x });
        const cols = x.shape.dims[2];
        const rows = x.shape.rows();
        const per = rowsPerItem(cols);
        const items = (rows + per - 1) / per;
        const partial = try self.allocator.alloc(f64, items);
        defer self.allocator.free(partial);
        const Ctx = struct {
            dx: []f32,
            dgate: []f32,
            dout: []const f32,
            x: []const f32,
            gate: []const f32,
            partial: []f64,
            lambda: f32,
            rows: usize,
            t: usize,
            cols: usize,
            per: usize,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                var lambda_sum: f64 = 0;
                for (item * ctx.per..@min((item + 1) * ctx.per, ctx.rows)) |r| {
                    const pos = r % ctx.t;
                    const g = ctx.dout[r * ctx.cols ..][0..ctx.cols];
                    const out = ctx.dx[r * ctx.cols ..][0..ctx.cols];
                    if (pos + 1 < ctx.t) {
                        const next = ctx.lambda * sigmoid(ctx.gate[r + 1]);
                        for (out, g, ctx.dout[(r + 1) * ctx.cols ..][0..ctx.cols]) |*o, cur, nxt| o.* = cur + next * nxt;
                    } else {
                        @memcpy(out, g);
                    }
                    if (pos == 0) {
                        ctx.dgate[r] = 0;
                        continue;
                    }
                    var s: f32 = 0;
                    for (g, ctx.x[(r - 1) * ctx.cols ..][0..ctx.cols]) |a, b| s += a * b;
                    const sg = sigmoid(ctx.gate[r]);
                    ctx.dgate[r] = ctx.lambda * sg * (1 - sg) * s;
                    lambda_sum += sg * s;
                }
                ctx.partial[item] = lambda_sum;
            }
        };
        const ctx = Ctx{
            .dx = elems(f32, dx),
            .dgate = elems(f32, dgate),
            .dout = elems(f32, dout),
            .x = elems(f32, x),
            .gate = elems(f32, gate),
            .partial = partial,
            .lambda = resolve(lambda),
            .rows = rows,
            .t = x.shape.dims[1],
            .cols = cols,
            .per = per,
        };
        try self.parallel.run(items, ctx, Ctx.work);
        var total: f64 = 0;
        for (partial) |p| total += p;
        elems(f32, dlambda)[0] += @floatCast(total * lambda.factor);
    }

    /// Backward of `valueMix` (the `v` gradient passes through unchanged):
    /// `dve = 3 * s * dv`, `dgate = 3 * s * (1 - s) * <dv, ve>` per `(r, h)`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dve`: same shape as `ve`, overwritten.
    /// - `dgate`: same shape as `gate`, overwritten.
    /// - `dv`: the gradient of the mixed values.
    /// - `ve`: the forward value embeddings.
    /// - `gate`: the forward gate logits `[R, H]`.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn valueMixBackward(self: *Self, dve: mod.Tensor, dgate: mod.Tensor, dv: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        try checkSame(dve, ve);
        try checkSame(dgate, gate);
        if (dv.dtype != .f32) return error.DtypeMismatch;
        const pairs = gate.numel();
        if (dv.numel() != ve.numel() or pairs == 0 or ve.numel() % pairs != 0) return mismatch("valueMixBackward", dv.shape, gate.shape);
        const Ctx = struct {
            dve: []f32,
            dgate: []f32,
            dv: []const f32,
            ve: []const f32,
            gate: []const f32,
            d: usize,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (start..end) |rh| {
                    const s = sigmoid(ctx.gate[rh]);
                    const g = ctx.dv[rh * ctx.d ..][0..ctx.d];
                    var sum: f32 = 0;
                    for (ctx.dve[rh * ctx.d ..][0..ctx.d], g, ctx.ve[rh * ctx.d ..][0..ctx.d]) |*o, gv, e| {
                        o.* = 3 * s * gv;
                        sum += gv * e;
                    }
                    ctx.dgate[rh] = 3 * s * (1 - s) * sum;
                }
            }
        };
        const d = ve.numel() / pairs;
        try self.forRows(pairs, d, Ctx{ .dve = elems(f32, dve), .dgate = elems(f32, dgate), .dv = elems(f32, dv), .ve = elems(f32, ve), .gate = elems(f32, gate), .d = d }, Ctx.body);
    }

    /// Backward of `embedding`: `dtable[ids[r]] += dout[r]`. Parallel over
    /// column blocks, so repeated ids never race and the order is fixed.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `dtable`: `[V, C]`, accumulated into.
    /// - `dout`: `R * C` gradients.
    /// - `ids`: `R` i32 indices below `V`.
    ///
    /// Return: nothing; shape/dtype errors, `error.OutOfBounds`.
    pub fn embeddingBackward(self: *Self, dtable: mod.Tensor, dout: mod.Tensor, ids: mod.Tensor) !void {
        if (dtable.dtype != .f32 or dout.dtype != .f32 or ids.dtype != .i32) return error.DtypeMismatch;
        const cols = dtable.shape.cols();
        const rows = ids.numel();
        if (dout.numel() != rows * cols) return mismatch("embeddingBackward", dout.shape, dtable.shape);
        const vocab = dtable.shape.rows();
        for (elems(i32, ids)) |id| {
            if (id < 0 or id >= vocab) return error.OutOfBounds;
        }
        const block = 64;
        const Ctx = struct {
            dtable: []f32,
            dout: []const f32,
            ids: []const i32,
            cols: usize,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                const start = item * block;
                const end = @min(start + block, ctx.cols);
                for (ctx.ids, 0..) |id, r| {
                    const base: usize = @as(usize, @intCast(id)) * ctx.cols;
                    for (ctx.dtable[base + start .. base + end], ctx.dout[r * ctx.cols + start .. r * ctx.cols + end]) |*o, g| o.* += g;
                }
            }
        };
        try self.parallel.run((cols + block - 1) / block, Ctx{ .dtable = elems(f32, dtable), .dout = elems(f32, dout), .ids = elems(i32, ids), .cols = cols }, Ctx.work);
    }

    /// One fused AdamW update, in place (`adamw_step_fused`): decoupled weight
    /// decay `p *= 1 - lr * wd`, moments `m = lerp(m, g, 1 - beta1)`,
    /// `v = lerp(v, g^2, 1 - beta2)`, then
    /// `p -= lr / (1 - beta1^step) * m / (sqrt(v / (1 - beta2^step)) + eps)`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `p`: the parameter.
    /// - `g`: its gradient, same shape.
    /// - `m`: first moment, same shape.
    /// - `v`: second moment, same shape.
    /// - `params`: hyperparameters and step count.
    ///
    /// Return: nothing; shape/dtype/param errors.
    pub fn adamwStep(self: *Self, p: mod.Tensor, g: mod.Tensor, m: mod.Tensor, v: mod.Tensor, params: mod.AdamWParams) !void {
        try checkSame(p, g);
        try checkSame(p, m);
        try checkSame(p, v);
        try params.validate();
        const step: f32 = @floatFromInt(params.step);
        const Ctx = struct {
            p: []f32,
            g: []const f32,
            m: []f32,
            v: []f32,
            decay: f32,
            w1: f32,
            w2: f32,
            bias2: f32,
            step_size: f32,
            eps: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (ctx.p[start..end], ctx.g[start..end], ctx.m[start..end], ctx.v[start..end]) |*pv, gv, *mv, *vv| {
                    pv.* *= ctx.decay;
                    mv.* = lerp(mv.*, gv, ctx.w1);
                    vv.* = lerp(vv.*, gv * gv, ctx.w2);
                    const denom = @sqrt(vv.* / ctx.bias2) + ctx.eps;
                    pv.* += -ctx.step_size * (mv.* / denom);
                }
            }
        };
        const ctx = Ctx{
            .p = elems(f32, p),
            .g = elems(f32, g),
            .m = elems(f32, m),
            .v = elems(f32, v),
            .decay = 1 - params.lr * params.weight_decay,
            .w1 = 1 - params.beta1,
            .w2 = 1 - params.beta2,
            .bias2 = 1 - std.math.pow(f32, params.beta2, step),
            .step_size = params.lr / (1 - std.math.pow(f32, params.beta1, step)),
            .eps = params.eps,
        };
        try self.forRows(p.numel(), 1, ctx, Ctx.body);
    }

    /// Muon's Nesterov momentum, in place: `buf = lerp(buf, g, 1 - momentum)`,
    /// then `g = lerp(g, buf, momentum)`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `g`: the gradient; becomes the Nesterov update.
    /// - `buf`: the momentum buffer, same shape.
    /// - `momentum`: the coefficient.
    ///
    /// Return: nothing; shape/dtype errors.
    pub fn muonMomentum(self: *Self, g: mod.Tensor, buf: mod.Tensor, momentum: f32) !void {
        try checkSame(g, buf);
        const Ctx = struct {
            g: []f32,
            buf: []f32,
            momentum: f32,
            fn body(ctx: @This(), start: usize, end: usize) void {
                for (ctx.g[start..end], ctx.buf[start..end]) |*gv, *bv| {
                    bv.* = lerp(bv.*, gv.*, 1 - ctx.momentum);
                    gv.* = lerp(gv.*, bv.*, ctx.momentum);
                }
            }
        };
        try self.forRows(g.numel(), 1, Ctx{ .g = elems(f32, g), .buf = elems(f32, buf), .momentum = momentum }, Ctx.body);
    }

    /// Muon's pre-orthogonalization scaling of a matrix, in place: MuonEq row
    /// equilibration (every row rescaled to the mean row norm
    /// `||X||_F / sqrt(rows)`), then `X /= ||X||_F * 1.01 + 1e-6`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `x`: the matrix `[rows, cols]`.
    ///
    /// Return: nothing; dtype/allocation errors.
    pub fn muonPrepare(self: *Self, x: mod.Tensor) !void {
        if (x.dtype != .f32) return error.DtypeMismatch;
        const rows = x.shape.rows();
        const cols = x.shape.cols();
        const data = elems(f32, x);
        const row_sq = try self.allocator.alloc(f64, rows);
        defer self.allocator.free(row_sq);
        rowSquares(data, cols, row_sq);
        var total: f64 = 0;
        for (row_sq) |s| total += s;
        const target: f32 = @floatCast(@sqrt(total) / @sqrt(@as(f64, @floatFromInt(rows))));
        for (0..rows) |r| {
            const row_norm = @max(@as(f32, @floatCast(@sqrt(row_sq[r]))), 1e-6);
            const factor = target / row_norm;
            for (data[r * cols ..][0..cols]) |*v| v.* *= factor;
        }
        rowSquares(data, cols, row_sq);
        total = 0;
        for (row_sq) |s| total += s;
        const norm: f32 = @floatCast(@sqrt(total));
        const denom = norm * 1.01 + 1e-6;
        for (data) |*v| v.* /= denom;
    }

    /// Muon's final stage on one matrix: Muon+ renormalization of the
    /// orthogonalized update `g` to Frobenius norm `sqrt(min(rows, cols))`,
    /// NorMuon variance reduction against the factored second moment, then the
    /// cautious update `p -= lr * g + lr * wd * p * [g * p >= 0]`.
    ///
    /// Parameters:
    /// - `self`: the backend.
    /// - `p`: the parameter `[rows, cols]`.
    /// - `g`: the orthogonalized update, same shape; scaled in place.
    /// - `second`: the second-moment buffer, `MuonParams.secondShape(rows, cols)`.
    /// - `params`: learning rate, weight decay, beta2.
    ///
    /// Return: nothing; shape/dtype/allocation errors.
    pub fn muonFinish(self: *Self, p: mod.Tensor, g: mod.Tensor, second: mod.Tensor, params: mod.MuonParams) !void {
        try checkSame(p, g);
        if (second.dtype != .f32) return error.DtypeMismatch;
        const rows = p.shape.rows();
        const cols = p.shape.cols();
        const by_row = rows >= cols;
        const shape = mod.MuonParams.secondShape(rows, cols);
        if (second.shape.rows() != shape[0] or second.shape.cols() != shape[1]) return mismatch("muonFinish", p.shape, second.shape);
        const gd = elems(f32, g);
        const pd = elems(f32, p);
        const sd = elems(f32, second);

        // Muon+: snap the Frobenius norm to sqrt(min(rows, cols)).
        const row_sq = try self.allocator.alloc(f64, rows);
        defer self.allocator.free(row_sq);
        rowSquares(gd, cols, row_sq);
        var total: f64 = 0;
        for (row_sq) |s| total += s;
        const target: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(@min(rows, cols)))));
        const renorm = target / @max(@as(f32, @floatCast(@sqrt(total))), 1e-6);
        for (gd) |*v| v.* *= renorm;

        // NorMuon: v_mean over the reduced dimension, an EMA of it, and a
        // per-row (or per-column) step that preserves the update's norm.
        const n = sd.len;
        const red_size: f32 = @floatFromInt(if (by_row) cols else rows);
        const v_mean = try self.allocator.alloc(f32, n);
        defer self.allocator.free(v_mean);
        @memset(v_mean, 0);
        const sums = try self.allocator.alloc(f64, n);
        defer self.allocator.free(sums);
        @memset(sums, 0);
        for (0..rows) |r| {
            for (gd[r * cols ..][0..cols], 0..) |v, c| sums[if (by_row) r else c] += @as(f64, v) * @as(f64, v);
        }
        var v_norm_sq: f64 = 0;
        for (v_mean, sums) |*mean, s| {
            mean.* = @floatCast(s / @as(f64, red_size));
            v_norm_sq += mean.*;
        }
        const v_norm: f32 = @floatCast(@sqrt(v_norm_sq * red_size));
        var new_sq: f64 = 0;
        for (sd, v_mean) |*s, mean| {
            s.* = lerp(s.*, mean, 1 - params.beta2);
            const step_size = 1 / @sqrt(@max(s.*, 1e-10));
            new_sq += @as(f64, mean * red_size * step_size * step_size);
        }
        const v_norm_new: f32 = @floatCast(@sqrt(new_sq));
        const ratio = v_norm / @max(v_norm_new, 1e-10);
        const lr_wd = params.lr * params.weight_decay;
        for (0..rows) |r| {
            for (gd[r * cols ..][0..cols], pd[r * cols ..][0..cols], 0..) |*gv, *pv, c| {
                const idx = if (by_row) r else c;
                const scale_by = (1 / @sqrt(@max(sd[idx], 1e-10))) * ratio;
                gv.* *= scale_by;
                const mask: f32 = if (gv.* * pv.* >= 0) 1 else 0;
                pv.* -= params.lr * gv.* + lr_wd * pv.* * mask;
            }
        }
    }

    const BinaryOp = enum { add, mul };

    /// Shared body of the same-shape elementwise binary ops.
    fn binary(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor, comptime op: BinaryOp) !void {
        try checkSame(out, a);
        try checkSame(out, b);
        const Ctx = struct {
            out: []f32,
            a: []const f32,
            b: []const f32,
            fn work(ctx: @This(), item: usize, worker: usize) void {
                _ = worker;
                const start = item * chunk_len;
                const end = @min(start + chunk_len, ctx.out.len);
                for (ctx.out[start..end], ctx.a[start..end], ctx.b[start..end]) |*o, x, y| {
                    o.* = switch (op) {
                        .add => x + y,
                        .mul => x * y,
                    };
                }
            }
        };
        try self.parallel.run(chunks(out.numel()), Ctx{ .out = elems(f32, out), .a = elems(f32, a), .b = elems(f32, b) }, Ctx.work);
    }

    /// Runs `body(ctx, start, end)` over `rows` rows of `cols` elements, in
    /// parallel pieces of about `chunk_len` elements.
    fn forRows(self: *Self, rows: usize, cols: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize, usize) void) !void {
        const per = @max(1, chunk_len / @max(cols, 1));
        const Wrap = struct {
            ctx: @TypeOf(ctx),
            rows: usize,
            per: usize,
            fn work(w: @This(), item: usize, worker: usize) void {
                _ = worker;
                const start = item * w.per;
                body(w.ctx, start, @min(start + w.per, w.rows));
            }
        };
        try self.parallel.run((rows + per - 1) / per, Wrap{ .ctx = ctx, .rows = rows, .per = per }, Wrap.work);
    }

    /// Fails when `out` shares memory with any of `inputs`.
    fn checkDisjoint(out: mod.Tensor, inputs: []const mod.Tensor) !void {
        const size = out.dtype.size();
        const lo = @intFromPtr(out.buffer.bytes.ptr) + out.offset * size;
        const hi = lo + out.numel() * size;
        for (inputs) |in| {
            const in_lo = @intFromPtr(in.buffer.bytes.ptr) + in.offset * in.dtype.size();
            const in_hi = in_lo + in.numel() * in.dtype.size();
            if (lo < in_hi and in_lo < hi) {
                log.debug("output overlaps an input", .{});
                return error.Aliasing;
            }
        }
    }

    /// Logs and returns a shape mismatch for `op`.
    fn mismatch(comptime op: []const u8, a: mod.Shape, b: mod.Shape) error{ShapeMismatch} {
        log.debug(op ++ ": shapes do not fit: {f} vs {f}", .{ a, b });
        return error.ShapeMismatch;
    }

    /// A `Scalar`'s value: element 0 of its tensor (if any) times its factor.
    fn resolve(scalar: mod.Scalar) f32 {
        const tensor = scalar.tensor orelse return scalar.factor;
        return elems(f32, tensor)[0] * scalar.factor;
    }

    fn sigmoid(x: f32) f32 {
        return 1 / (1 + @exp(-x));
    }

    /// Validates logits `[R, V]` against `R` i32 targets in `[-1, V)`.
    fn checkTargets(self: *Self, logits: mod.Tensor, targets: mod.Tensor) !struct { usize, usize } {
        _ = self;
        if (logits.dtype != .f32 or targets.dtype != .i32) return error.DtypeMismatch;
        const rows = logits.shape.rows();
        const vocab = logits.shape.cols();
        if (targets.numel() != rows) return mismatch("crossEntropy", logits.shape, targets.shape);
        for (elems(i32, targets)) |t| {
            if (t < -1 or t >= vocab) {
                log.debug("target {d} outside vocab {d}", .{ t, vocab });
                return error.OutOfBounds;
            }
        }
        return .{ rows, vocab };
    }

    /// Rows that are not `ignore_index` (-1).
    fn validTargets(targets: []const i32) usize {
        var count: usize = 0;
        for (targets) |t| count += @intFromBool(t >= 0);
        return count;
    }

    /// `log(sum(exp(row)))`, stabilized by the row max, in f64.
    fn logSumExp(row: []const f32) f64 {
        var max = -std.math.inf(f32);
        for (row) |z| max = @max(max, z);
        var sum: f64 = 0;
        for (row) |z| sum += @exp(@as(f64, z - max));
        return @as(f64, max) + @log(sum);
    }

    /// Rows per work item for rows of `cols` elements.
    fn rowsPerItem(cols: usize) usize {
        return @max(1, chunk_len / @max(cols, 1));
    }

    /// `torch.lerp`: `a + w * (b - a)`, evaluated from the nearer end for accuracy.
    fn lerp(a: f32, b: f32, w: f32) f32 {
        return if (@abs(w) < 0.5) a + w * (b - a) else b - (b - a) * (1 - w);
    }

    /// Each row's sum of squares, in f64.
    fn rowSquares(data: []const f32, cols: usize, out: []f64) void {
        for (out, 0..) |*s, r| {
            var sum: f64 = 0;
            for (data[r * cols ..][0..cols]) |v| sum += @as(f64, v) * @as(f64, v);
            s.* = sum;
        }
    }

    /// Fails unless both tensors are f32 with the same shape.
    fn checkSame(x: mod.Tensor, y: mod.Tensor) !void {
        if (x.dtype != .f32 or y.dtype != .f32) return error.DtypeMismatch;
        if (!x.shape.eql(y.shape)) {
            log.debug("elementwise shapes differ: {f} vs {f}", .{ x.shape, y.shape });
            return error.ShapeMismatch;
        }
    }

    /// Work items needed to cover `n` elements in `chunk_len` pieces.
    fn chunks(n: usize) usize {
        return (n + chunk_len - 1) / chunk_len;
    }

    /// A tensor's elements as a host slice. CPU-internal: the contract never
    /// exposes element memory.
    fn elems(comptime T: type, tensor: mod.Tensor) []T {
        const base: [*]T = @ptrCast(tensor.buffer.bytes.ptr);
        return base[tensor.offset..][0..tensor.numel()];
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "cpu backend passes the conformance suite" {
    try mod.Conformance.runAll(mod.CpuBackend, std.testing.allocator, std.testing.io);
}

test "cpu backend parallel elementwise covers multiple chunks" {
    var backend = try mod.CpuBackend.init(std.testing.allocator, std.testing.io, .{ .threads = 4 });
    defer backend.deinit();
    const n = 3 * chunk_len + 5;
    const x = try backend.alloc(.f32, &.{n});
    defer backend.free(x);
    try backend.fill(x, 2);
    try backend.mul(x, x, x);
    try backend.scale(x, x, 0.5);
    const out = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(out);
    try backend.download(x, f32, out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 2), v);
}
