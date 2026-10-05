// ---------------------------------------------------------------------------
// CUDA kernels of CudaBackend (src/cuda/backend.zig), compiled for
// nvptx64-cuda to LLVM IR, then to PTX (build.zig: tools/nvptx_fixup.zig and
// `zig cc`). Each mirrors one CpuBackend op; one thread per output element.
// Only builtins and PTX-native math (`@sqrt`, `@max`): no libm on the device.
// ---------------------------------------------------------------------------

// The PTX entry names CudaBackend looks up.
comptime {
    @export(&fillF32, .{ .name = "fill_f32" });
    @export(&copyU32, .{ .name = "copy_u32" });
    @export(&addF32, .{ .name = "add_f32" });
    @export(&mulF32, .{ .name = "mul_f32" });
    @export(&scaleF32, .{ .name = "scale_f32" });
    @export(&combineF32, .{ .name = "combine_f32" });
    @export(&reluSquareF32, .{ .name = "relu_square_f32" });
    @export(&reluSquareBackwardF32, .{ .name = "relu_square_backward_f32" });
    @export(&embeddingF32, .{ .name = "embedding_f32" });
    @export(&rmsnormF32, .{ .name = "rmsnorm_f32" });
    @export(&rmsnormBackwardF32, .{ .name = "rmsnorm_backward_f32" });
    @export(&matmulF32, .{ .name = "matmul_f32" });
}

fn index() u32 {
    return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
}

fn fillF32(out: [*]f32, value: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) out[i] = value;
}

fn copyU32(out: [*]u32, src: [*]const u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) out[i] = src[i];
}

fn addF32(out: [*]f32, a: [*]const f32, b: [*]const f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) out[i] = a[i] + b[i];
}

fn mulF32(out: [*]f32, a: [*]const f32, b: [*]const f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) out[i] = a[i] * b[i];
}

fn scaleF32(out: [*]f32, a: [*]const f32, s: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) out[i] = a[i] * s;
}

/// `out = xs * x + ys * y`; a scalar is `factor * value[0]` when `has` is set.
fn combineF32(out: [*]f32, x: [*]const f32, y: [*]const f32, xs: [*]const f32, ys: [*]const f32, x_factor: f32, y_factor: f32, x_has: u32, y_has: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const a = if (x_has != 0) xs[0] * x_factor else x_factor;
    const b = if (y_has != 0) ys[0] * y_factor else y_factor;
    out[i] = a * x[i] + b * y[i];
}

fn reluSquareF32(out: [*]f32, x: [*]const f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const r = @max(x[i], 0);
    out[i] = r * r;
}

fn reluSquareBackwardF32(dx: [*]f32, dy: [*]const f32, x: [*]const f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) dx[i] = dy[i] * 2 * @max(x[i], 0);
}

fn embeddingF32(out: [*]f32, table: [*]const f32, ids: [*]const i32, cols: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const id: u32 = @intCast(ids[i / cols]);
    out[i] = table[id * cols + i % cols];
}

/// One thread per row.
fn rmsnormF32(out: [*]f32, x: [*]const f32, rows: u32, cols: u32, eps: f32) callconv(.kernel) void {
    const r = index();
    if (r >= rows) return;
    const base = r * cols;
    var sq: f32 = 0;
    for (0..cols) |c| sq += x[base + c] * x[base + c];
    const inv = 1 / @sqrt(sq / @as(f32, @floatFromInt(cols)) + eps);
    for (0..cols) |c| out[base + c] = x[base + c] * inv;
}

/// One thread per row.
fn rmsnormBackwardF32(dx: [*]f32, dy: [*]const f32, x: [*]const f32, rows: u32, cols: u32, eps: f32, accumulate: u32) callconv(.kernel) void {
    const r = index();
    if (r >= rows) return;
    const base = r * cols;
    var sq: f32 = 0;
    var gx: f32 = 0;
    for (0..cols) |c| {
        sq += x[base + c] * x[base + c];
        gx += dy[base + c] * x[base + c];
    }
    const n: f32 = @floatFromInt(cols);
    const inv = 1 / @sqrt(sq / n + eps);
    const k = inv * inv * inv * gx / n;
    for (0..cols) |c| {
        const d = inv * dy[base + c] - x[base + c] * k;
        dx[base + c] = if (accumulate != 0) dx[base + c] + d else d;
    }
}

/// `C (+)= alpha * op(A) @ op(B)`, one thread per element of C.
fn matmulF32(c: [*]f32, a: [*]const f32, b: [*]const f32, m: u32, n: u32, k: u32, transpose_a: u32, transpose_b: u32, accumulate: u32, alpha: f32) callconv(.kernel) void {
    const i = index();
    if (i >= m * n) return;
    const row = i / n;
    const col = i % n;
    var sum: f32 = 0;
    for (0..k) |kk| {
        const av = if (transpose_a != 0) a[kk * m + row] else a[row * k + kk];
        const bv = if (transpose_b != 0) b[col * k + kk] else b[kk * n + col];
        sum += av * bv;
    }
    c[i] = if (accumulate != 0) c[i] + alpha * sum else alpha * sum;
}
