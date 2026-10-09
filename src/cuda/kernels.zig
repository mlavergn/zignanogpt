// ---------------------------------------------------------------------------
// CUDA kernels of CudaBackend (src/cuda/backend.zig), compiled for
// nvptx64-cuda, whose assembly is PTX (build.zig: `cudaKernels`). Each
// mirrors one CpuBackend op (and its Metal kernel's layout).
//
// No libm on the device: math is PTX-native (`@sqrt`, division) or inline PTX
// (`ex2.approx`, `lg2.approx`, barriers, warp shuffles). Shared memory is a
// container-level `addrspace(.shared)` array; PTX gives each kernel that
// touches one its own copy. Reductions run in a fixed order (deterministic);
// sums that the CPU keeps in f64 are f64 here too (CUDA has native f64).
//
// Not yet run on hardware: written blind for the DGX Spark (GB10), checked by
// compiling to PTX. The inline-asm strings are validated only when the driver
// JIT-compiles the PTX (`cuModuleLoadData`).
// ---------------------------------------------------------------------------

const std = @import("std");

// The PTX entry names CudaBackend looks up (`Kernel` there lists the same).
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
    @export(&embeddingBackwardF32, .{ .name = "embedding_backward_f32" });
    @export(&rmsnormF32, .{ .name = "rmsnorm_f32" });
    @export(&rmsnormBackwardF32, .{ .name = "rmsnorm_backward_f32" });
    @export(&matmulF32, .{ .name = "matmul_f32" });
    @export(&ropeF32, .{ .name = "rope_f32" });
    @export(&gateLinearF32, .{ .name = "gate_linear_f32" });
    @export(&gateLinearBackwardDxF32, .{ .name = "gate_linear_backward_dx_f32" });
    @export(&gateLinearBackwardDwF32, .{ .name = "gate_linear_backward_dw_f32" });
    @export(&smearF32, .{ .name = "smear_f32" });
    @export(&smearBackwardF32, .{ .name = "smear_backward_f32" });
    @export(&gatedAddF32, .{ .name = "gated_add_f32" });
    @export(&valueMixF32, .{ .name = "value_mix_f32" });
    @export(&valueMixBackwardF32, .{ .name = "value_mix_backward_f32" });
    @export(&softcapF32, .{ .name = "softcap_f32" });
    @export(&softcapLseF32, .{ .name = "softcap_lse_f32" });
    @export(&xentRowsF32, .{ .name = "xent_rows_f32" });
    @export(&xentRowsLseF32, .{ .name = "xent_rows_lse_f32" });
    @export(&xentBackwardF32, .{ .name = "xent_backward_f32" });
    @export(&dotPartialF32, .{ .name = "dot_partial_f32" });
    @export(&reduceSumF64, .{ .name = "reduce_sum_f64" });
    @export(&reduceSumF32, .{ .name = "reduce_sum_f32" });
    @export(&attentionF32, .{ .name = "attention_f32" });
    @export(&attentionDeltaF32, .{ .name = "attention_delta_f32" });
    @export(&attentionDqF32, .{ .name = "attention_dq_f32" });
    @export(&attentionDkvF32, .{ .name = "attention_dkv_f32" });
    @export(&adamwStepF32, .{ .name = "adamw_step_f32" });
    @export(&muonMomentumF32, .{ .name = "muon_momentum_f32" });
    @export(&rowSquaresF64, .{ .name = "row_squares_f64" });
    @export(&colSquaresF64, .{ .name = "col_squares_f64" });
    @export(&totalF64, .{ .name = "total_f64" });
    @export(&muonScaleRowsF32, .{ .name = "muon_scale_rows_f32" });
    @export(&muonScaleAllF32, .{ .name = "muon_scale_all_f32" });
    @export(&muonRenormF32, .{ .name = "muon_renorm_f32" });
    @export(&normuonStatsF32, .{ .name = "normuon_stats_f32" });
    @export(&muonUpdateF32, .{ .name = "muon_update_f32" });
}

// ---------------------------------------------------------------------------
// Thread indexing, barriers, shuffles and math

/// The thread's index in its block.
fn tid() u32 {
    return @workItemId(0);
}

/// The block's size in threads (a multiple of 32 for the block-wide reductions).
fn threads() u32 {
    return @workGroupSize(0);
}

/// The block's index along `dim`.
fn block(comptime dim: u32) u32 {
    return @workGroupId(dim);
}

/// The thread's index in a one-dimensional grid.
fn index() u32 {
    return block(0) * threads() + tid();
}

/// `__syncthreads()`: every thread of the block reaches it before any passes,
/// and shared-memory writes before it are visible after it.
fn syncThreads() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

/// The value `lane ^ mask` holds (the whole warp takes part).
fn shflXor(v: f32, mask: u32) f32 {
    return asm volatile ("shfl.sync.bfly.b32 %[r], %[v], %[m], 0x1f, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (v),
          [m] "r" (mask),
    );
}

/// `shflXor` for an f64, as its two 32-bit halves.
fn shflXorF64(v: f64, mask: u32) f64 {
    const bits: u64 = @bitCast(v);
    const lo: f32 = @bitCast(@as(u32, @truncate(bits)));
    const hi: f32 = @bitCast(@as(u32, @truncate(bits >> 32)));
    const got_lo: u32 = @bitCast(shflXor(lo, mask));
    const got_hi: u32 = @bitCast(shflXor(hi, mask));
    return @bitCast((@as(u64, got_hi) << 32) | got_lo);
}

/// 2^x (PTX `ex2.approx`, about 2 ulp).
fn exp2(x: f32) f32 {
    return asm ("ex2.approx.ftz.f32 %[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
    );
}

/// log2(x) (PTX `lg2.approx`).
fn log2(x: f32) f32 {
    return asm ("lg2.approx.ftz.f32 %[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
    );
}

const log2_e: f32 = 1.4426950408889634;
const ln_2: f32 = 0.6931471805599453;

fn exp(x: f32) f32 {
    return exp2(x * log2_e);
}

fn log(x: f32) f32 {
    return log2(x) * ln_2;
}

/// tanh from `exp` (PTX's `tanh.approx` is only ~11 bits).
fn tanh(x: f32) f32 {
    const e = exp(2 * x);
    return 1 - 2 / (e + 1);
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + exp(-x));
}

/// torch's two-sided lerp (CpuBackend.lerp).
fn lerp(a: f32, b: f32, w: f32) f32 {
    return if (@abs(w) < 0.5) a + w * (b - a) else b - (b - a) * (1 - w);
}

// ---------------------------------------------------------------------------
// Block-wide reductions (blocks of up to 1024 threads, a multiple of 32)

var reduce_f32: [32]f32 addrspace(.shared) = undefined;
var reduce_f64: [32]f64 addrspace(.shared) = undefined;

fn warpSum(v: f32) f32 {
    var x = v;
    inline for (.{ 16, 8, 4, 2, 1 }) |m| x += shflXor(x, m);
    return x;
}

fn warpMax(v: f32) f32 {
    var x = v;
    inline for (.{ 16, 8, 4, 2, 1 }) |m| x = @max(x, shflXor(x, m));
    return x;
}

fn warpSumF64(v: f64) f64 {
    var x = v;
    inline for (.{ 16, 8, 4, 2, 1 }) |m| x += shflXorF64(x, m);
    return x;
}

/// The block's sum of `v`, in every thread; warps added in order.
fn blockSum(v: f32) f32 {
    const w = warpSum(v);
    if (tid() % 32 == 0) reduce_f32[tid() / 32] = w;
    syncThreads();
    var total: f32 = 0;
    for (0..threads() / 32) |i| total += reduce_f32[i];
    syncThreads();
    return total;
}

fn blockMax(v: f32) f32 {
    const w = warpMax(v);
    if (tid() % 32 == 0) reduce_f32[tid() / 32] = w;
    syncThreads();
    var total: f32 = -std.math.floatMax(f32);
    for (0..threads() / 32) |i| total = @max(total, reduce_f32[i]);
    syncThreads();
    return total;
}

fn blockSumF64(v: f64) f64 {
    const w = warpSumF64(v);
    if (tid() % 32 == 0) reduce_f64[tid() / 32] = w;
    syncThreads();
    var total: f64 = 0;
    for (0..threads() / 32) |i| total += reduce_f64[i];
    syncThreads();
    return total;
}

// ---------------------------------------------------------------------------
// Elementwise (one thread per element)

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

/// `dtable[id] += dout[r]` for every row with that id, in row order. `index`
/// holds the rows grouped by id, each id's start (and the end), then the ids
/// (the host groups them). One thread per (id, column).
fn embeddingBackwardF32(dtable: [*]f32, dout: [*]const f32, groups: [*]const u32, cols: u32, unique: u32, rows: u32) callconv(.kernel) void {
    const i = index();
    if (i >= unique * cols) return;
    const u = i / cols;
    const c = i % cols;
    const starts = groups + rows;
    const ids = starts + unique + 1;
    const cell = &dtable[ids[u] * cols + c];
    var acc = cell.*;
    var k = starts[u];
    while (k < starts[u + 1]) : (k += 1) acc += dout[groups[k] * cols + c];
    cell.* = acc;
}

/// Rotary embedding (`sign` 1) or its inverse (-1) over `[B, T, H, 2 * half_d]`.
fn ropeF32(out: [*]f32, x: [*]const f32, cos_t: [*]const f32, sin_t: [*]const f32, t: u32, h: u32, half_d: u32, pos0: u32, sign: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const r = i / half_d;
    const j = i % half_d;
    const pos = pos0 + (r / h) % t;
    const c = cos_t[pos * half_d + j];
    const s = sign * sin_t[pos * half_d + j];
    const base = r * 2 * half_d;
    const x1 = x[base + j];
    const x2 = x[base + half_d + j];
    out[base + j] = x1 * c + x2 * s;
    out[base + half_d + j] = x2 * c - x1 * s;
}

fn gateLinearF32(out: [*]f32, x: [*]const f32, w: [*]const f32, cols: u32, cin: u32, heads: u32, rows: u32) callconv(.kernel) void {
    const i = index();
    if (i >= rows * heads) return;
    const r = i / heads;
    const hd = i % heads;
    var sum: f32 = 0;
    for (0..cin) |c| sum += x[r * cols + c] * w[hd * cin + c];
    out[i] = sum;
}

fn gateLinearBackwardDxF32(dx: [*]f32, dout: [*]const f32, w: [*]const f32, cols: u32, cin: u32, heads: u32, rows: u32) callconv(.kernel) void {
    const i = index();
    if (i >= rows * cin) return;
    const r = i / cin;
    const c = i % cin;
    var sum: f32 = 0;
    for (0..heads) |hd| sum += dout[r * heads + hd] * w[hd * cin + c];
    dx[r * cols + c] += sum;
}

/// One block per weight: the rows split across its threads, then a block sum.
fn gateLinearBackwardDwF32(dw: [*]f32, dout: [*]const f32, x: [*]const f32, cols: u32, cin: u32, heads: u32, rows: u32) callconv(.kernel) void {
    const i = block(0);
    const hd = i / cin;
    const c = i % cin;
    var sum: f32 = 0;
    var r = tid();
    while (r < rows) : (r += threads()) sum += dout[r * heads + hd] * x[r * cols + c];
    const total = blockSum(sum);
    if (tid() == 0) dw[i] += total;
}

/// The previous token smeared in: `out[t] = x[t] + l * sigmoid(gate[t]) * x[t - 1]`.
fn smearF32(out: [*]f32, x: [*]const f32, gate: [*]const f32, lambda: [*]const f32, t: u32, cols: u32, factor: f32, has_lambda: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const r = i / cols;
    if (r % t == 0) {
        out[i] = x[i];
        return;
    }
    const l = if (has_lambda != 0) lambda[0] * factor else factor;
    out[i] = x[i] + l * sigmoid(gate[r]) * x[i - cols];
}

/// One block per row: dx, dgate and the row's share of dlambda (`partial`).
fn smearBackwardF32(dx: [*]f32, dgate: [*]f32, partial: [*]f32, dout: [*]const f32, x: [*]const f32, gate: [*]const f32, lambda: [*]const f32, t: u32, cols: u32, factor: f32, has_lambda: u32) callconv(.kernel) void {
    const r = block(0);
    const l = if (has_lambda != 0) lambda[0] * factor else factor;
    const pos = r % t;
    const g = dout + r * cols;
    const out = dx + r * cols;
    const has_next = pos + 1 < t;
    const next = if (has_next) l * sigmoid(gate[r + 1]) else 0;
    var c = tid();
    while (c < cols) : (c += threads()) out[c] = if (has_next) g[c] + next * g[c + cols] else g[c];
    var s: f32 = 0;
    if (pos > 0) {
        c = tid();
        while (c < cols) : (c += threads()) s += g[c] * x[(r - 1) * cols + c];
    }
    const total = blockSum(s);
    if (tid() == 0) {
        if (pos == 0) {
            dgate[r] = 0;
            partial[r] = 0;
        } else {
            const sg = sigmoid(gate[r]);
            dgate[r] = l * sg * (1 - sg) * total;
            partial[r] = sg * total;
        }
    }
}

/// `out[r] = x[r] + s * sigmoid(gate[r]) * y[r]`.
fn gatedAddF32(out: [*]f32, x: [*]const f32, y: [*]const f32, gate: [*]const f32, scalar: [*]const f32, cols: u32, factor: f32, has_scalar: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const s = if (has_scalar != 0) scalar[0] * factor else factor;
    out[i] = x[i] + s * sigmoid(gate[i / cols]) * y[i];
}

fn valueMixF32(v: [*]f32, ve: [*]const f32, gate: [*]const f32, d: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i < n) v[i] += 3 * sigmoid(gate[i / d]) * ve[i];
}

/// One thread per (row, head) pair.
fn valueMixBackwardF32(dve: [*]f32, dgate: [*]f32, dv: [*]const f32, ve: [*]const f32, gate: [*]const f32, d: u32, pairs: u32) callconv(.kernel) void {
    const rh = index();
    if (rh >= pairs) return;
    const s = sigmoid(gate[rh]);
    var sum: f32 = 0;
    for (0..d) |c| {
        const g = dv[rh * d + c];
        dve[rh * d + c] = 3 * s * g;
        sum += g * ve[rh * d + c];
    }
    dgate[rh] = 3 * s * (1 - s) * sum;
}

// ---------------------------------------------------------------------------
// Row reductions (one block per row)

fn rmsnormF32(out: [*]f32, x: [*]const f32, cols: u32, eps: f32) callconv(.kernel) void {
    const xr = x + block(0) * cols;
    var sq: f32 = 0;
    var c = tid();
    while (c < cols) : (c += threads()) sq += xr[c] * xr[c];
    const total = blockSum(sq);
    const inv = 1 / @sqrt(total / @as(f32, @floatFromInt(cols)) + eps);
    const o = out + block(0) * cols;
    c = tid();
    while (c < cols) : (c += threads()) o[c] = xr[c] * inv;
}

fn rmsnormBackwardF32(dx: [*]f32, dy: [*]const f32, x: [*]const f32, cols: u32, eps: f32, accumulate: u32) callconv(.kernel) void {
    const xr = x + block(0) * cols;
    const gr = dy + block(0) * cols;
    var sq: f32 = 0;
    var gx: f32 = 0;
    var c = tid();
    while (c < cols) : (c += threads()) {
        sq += xr[c] * xr[c];
        gx += gr[c] * xr[c];
    }
    const sq_total = blockSum(sq);
    const gx_total = blockSum(gx);
    const n: f32 = @floatFromInt(cols);
    const inv = 1 / @sqrt(sq_total / n + eps);
    const k = inv * inv * inv * gx_total / n;
    const o = dx + block(0) * cols;
    c = tid();
    while (c < cols) : (c += threads()) {
        const d = inv * gr[c] - xr[c] * k;
        o[c] = if (accumulate != 0) o[c] + d else d;
    }
}

/// The soft cap, elementwise: `out[r, c] = cap * tanh(logits[r, c] / cap)`.
fn softcapF32(out: [*]f32, logits: [*]const f32, cols: u32, padded: u32, cap: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    out[i] = cap * tanh(logits[(i / cols) * padded + i % cols] / cap);
}

/// The soft cap plus the row's log-sum-exp of the capped values.
fn softcapLseF32(out: [*]f32, logits: [*]const f32, lse: [*]f32, cols: u32, padded: u32, cap: f32) callconv(.kernel) void {
    const r = block(0);
    const src = logits + r * padded;
    const dst = out + r * cols;
    var m: f32 = -std.math.floatMax(f32);
    var c = tid();
    while (c < cols) : (c += threads()) {
        const z = cap * tanh(src[c] / cap);
        dst[c] = z;
        m = @max(m, z);
    }
    const row_max = blockMax(m);
    var s: f32 = 0;
    c = tid();
    while (c < cols) : (c += threads()) s += exp(dst[c] - row_max);
    const total = blockSum(s);
    if (tid() == 0) lse[r] = row_max + log(total);
}

/// The log-sum-exp of a row, in every thread of its block.
fn rowLse(row: [*]const f32, cols: u32) f32 {
    var m: f32 = -std.math.floatMax(f32);
    var c = tid();
    while (c < cols) : (c += threads()) m = @max(m, row[c]);
    const row_max = blockMax(m);
    var s: f32 = 0;
    c = tid();
    while (c < cols) : (c += threads()) s += exp(row[c] - row_max);
    return row_max + log(blockSum(s));
}

/// `losses[r] = lse(row) - row[target]`, 0 where the target is -1.
fn xentRowsF32(losses: [*]f32, logits: [*]const f32, targets: [*]const i32, vocab: u32) callconv(.kernel) void {
    const r = block(0);
    const t = targets[r];
    if (t < 0) {
        if (tid() == 0) losses[r] = 0;
        return;
    }
    const row = logits + r * vocab;
    const lse = rowLse(row, vocab);
    if (tid() == 0) losses[r] = lse - row[@intCast(t)];
}

/// `xentRowsF32` with the soft cap's log-sum-exp given: one thread per row.
fn xentRowsLseF32(losses: [*]f32, logits: [*]const f32, targets: [*]const i32, lse: [*]const f32, vocab: u32, rows: u32) callconv(.kernel) void {
    const r = index();
    if (r >= rows) return;
    const t = targets[r];
    losses[r] = if (t < 0) 0 else lse[r] - logits[r * vocab + @as(u32, @intCast(t))];
}

/// Gradient through the soft cap: `weights[r]` (or `factor`) * (softmax -
/// onehot) * (1 - (z / cap)^2); padding columns and ignored rows get zero.
fn xentBackwardF32(dpad: [*]f32, logits: [*]const f32, targets: [*]const i32, weights: [*]const f32, row_lse: [*]const f32, vocab: u32, padded: u32, cap: f32, factor: f32, has_weights: u32, has_lse: u32) callconv(.kernel) void {
    const r = block(0);
    const out = dpad + r * padded;
    const t = targets[r];
    var c = tid();
    if (t < 0) {
        while (c < padded) : (c += threads()) out[c] = 0;
        return;
    }
    const row = logits + r * vocab;
    const lse = if (has_lse != 0) row_lse[r] else rowLse(row, vocab);
    const f = if (has_weights != 0) weights[r] else factor;
    while (c < padded) : (c += threads()) {
        if (c >= vocab) {
            out[c] = 0;
            continue;
        }
        const z = row[c];
        const squash = z / cap;
        const onehot: f32 = if (c == @as(u32, @intCast(t))) 1 else 0;
        out[c] = (exp(z - lse) - onehot) * f * (1 - squash * squash);
    }
}

// ---------------------------------------------------------------------------
// Sums (fixed order, f64)

/// `partial[block] = sum over the block's slice of a[i] * b[i]`, in f64.
fn dotPartialF32(a: [*]const f32, b: [*]const f32, partial: [*]f64, n: u32) callconv(.kernel) void {
    const per = (n + gridBlocks() - 1) / gridBlocks();
    const start = block(0) * per;
    const end = @min(start + per, n);
    var sum: f64 = 0;
    var i = start + tid();
    while (i < end) : (i += threads()) sum += @as(f64, a[i]) * @as(f64, b[i]);
    const total = blockSumF64(sum);
    if (tid() == 0) partial[block(0)] = total;
}

/// The grid's size in blocks (dimension 0), from PTX's `%nctaid.x`.
fn gridBlocks() u32 {
    return asm ("mov.u32 %[r], %%nctaid.x;"
        : [r] "=r" (-> u32),
    );
}

/// `out[0] (+)= factor * sum(values[0 .. count])` (f64 values); one block.
fn reduceSumF64(values: [*]const f64, out: [*]f32, count: u32, factor: f32, accumulate: u32) callconv(.kernel) void {
    var sum: f64 = 0;
    var i = tid();
    while (i < count) : (i += threads()) sum += values[i];
    const total = blockSumF64(sum);
    if (tid() == 0) {
        const v: f32 = @floatCast(total * factor);
        out[0] = if (accumulate != 0) out[0] + v else v;
    }
}

/// `out[0] (+)= factor * sum(values[0 .. count])` (f32 values, summed in f64); one block.
fn reduceSumF32(values: [*]const f32, out: [*]f32, count: u32, factor: f32, accumulate: u32) callconv(.kernel) void {
    var sum: f64 = 0;
    var i = tid();
    while (i < count) : (i += threads()) sum += values[i];
    const total = blockSumF64(sum);
    if (tid() == 0) {
        const v: f32 = @floatCast(total * factor);
        out[0] = if (accumulate != 0) out[0] + v else v;
    }
}

// ---------------------------------------------------------------------------
// Matmul: 64x64 output tiles per block of 256 threads (4x4 per thread), K in
// steps of 16 through shared memory; the transposes are applied while staging
// (loads follow the stored layout, so they stay coalesced). Batched products
// are grid dimension 2, each operand `batch` matrices back to back.

var tile_a: [16 * 64]f32 addrspace(.shared) = undefined; // [k][m]
var tile_b: [16 * 64]f32 addrspace(.shared) = undefined; // [k][n]

fn matmulF32(c_base: [*]f32, a_base: [*]const f32, b_base: [*]const f32, m: u32, n: u32, k: u32, transpose_a: u32, transpose_b: u32, accumulate: u32, alpha: f32) callconv(.kernel) void {
    const z: usize = block(2);
    const a = a_base + z * m * k;
    const b = b_base + z * k * n;
    const c = c_base + z * m * n;
    const row0 = block(1) * 64;
    const col0 = block(0) * 64;
    const t = tid();
    const tx = t % 16;
    const ty = t / 16;
    var sums: [4][4]f32 = @splat(@splat(0));
    var k0: u32 = 0;
    while (k0 < k) : (k0 += 16) {
        for (0..4) |l| {
            const e: u32 = t + @as(u32, @intCast(l)) * 256;
            // Stored A is [m, k] (k fastest) or [k, m] (m fastest).
            const kk = if (transpose_a != 0) e / 64 else e % 16;
            const i = if (transpose_a != 0) e % 64 else e / 16;
            const gi = row0 + i;
            const gk = k0 + kk;
            tile_a[kk * 64 + i] = if (gi < m and gk < k) (if (transpose_a != 0) a[gk * m + gi] else a[gi * k + gk]) else 0;
        }
        for (0..4) |l| {
            const e: u32 = t + @as(u32, @intCast(l)) * 256;
            // Stored B is [k, n] (n fastest) or [n, k] (k fastest).
            const kk = if (transpose_b != 0) e % 16 else e / 64;
            const j = if (transpose_b != 0) e / 16 else e % 64;
            const gj = col0 + j;
            const gk = k0 + kk;
            tile_b[kk * 64 + j] = if (gk < k and gj < n) (if (transpose_b != 0) b[gj * k + gk] else b[gk * n + gj]) else 0;
        }
        syncThreads();
        for (0..16) |kk| {
            var av: [4]f32 = undefined;
            var bv: [4]f32 = undefined;
            for (0..4) |r| av[r] = tile_a[kk * 64 + ty * 4 + r];
            for (0..4) |q| bv[q] = tile_b[kk * 64 + tx * 4 + q];
            for (0..4) |r| {
                for (0..4) |q| sums[r][q] += av[r] * bv[q];
            }
        }
        syncThreads();
    }
    for (0..4) |r| {
        const gi = row0 + ty * 4 + @as(u32, @intCast(r));
        if (gi >= m) continue;
        for (0..4) |q| {
            const gj = col0 + tx * 4 + @as(u32, @intCast(q));
            if (gj >= n) continue;
            const v = alpha * sums[r][q];
            const dst = &c[gi * n + gj];
            dst.* = if (accumulate != 0) dst.* + v else v;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention (CpuAttention semantics): one warp per query row, each lane owning
// head dimensions lane, lane + 32, ... (head dim up to 256). Plain online
// softmax over the visible keys; simple and deterministic, the first target
// for tuning on hardware.

const max_lane_dims = 8;

/// The attention problem, passed by value (the host's `AttnDims` has the same layout).
const AttnDims = extern struct {
    b: u32,
    tq: u32,
    tk: u32,
    keys: u32,
    h: u32,
    hkv: u32,
    d: u32,
    window: u32,
};

/// The warp's index in the grid and the lane within it.
fn warpIndex() u32 {
    return index() / 32;
}

fn lane() u32 {
    return tid() % 32;
}

/// `<x, y>` over the head dimension, in every lane.
fn warpDot(x: [*]const f32, y: [*]const f32, d: u32) f32 {
    var s: f32 = 0;
    var c = lane();
    while (c < d) : (c += 32) s += x[c] * y[c];
    return warpSum(s);
}

fn visible(i: u32, j: u32, keys: u32, window: u32) bool {
    return j < keys and j <= i and i - j <= window;
}

fn attentionF32(out: [*]f32, q: [*]const f32, k: [*]const f32, v: [*]const f32, lse: [*]f32, dims: AttnDims, scale: f32, has_lse: u32) callconv(.kernel) void {
    const w = warpIndex();
    if (w >= dims.b * dims.tq * dims.h) return;
    const hd = w % dims.h;
    const t = (w / dims.h) % dims.tq;
    const b = w / (dims.h * dims.tq);
    const kvh = hd / (dims.h / dims.hkv);
    const i = dims.keys - dims.tq + t;
    const qrow = q + ((b * dims.tq + t) * dims.h + hd) * dims.d;
    var acc: [max_lane_dims]f32 = @splat(0);
    var m: f32 = 0;
    var l: f32 = 0;
    const lo = if (i > dims.window) i - dims.window else 0;
    var j = lo;
    while (j <= i and j < dims.keys) : (j += 1) {
        const krow = k + ((b * dims.tk + j) * dims.hkv + kvh) * dims.d;
        const vrow = v + ((b * dims.tk + j) * dims.hkv + kvh) * dims.d;
        const s = warpDot(qrow, krow, dims.d) * scale;
        if (l == 0) {
            m = s;
            l = 1;
            for (0..max_lane_dims) |e| {
                const c = lane() + @as(u32, @intCast(e)) * 32;
                if (c < dims.d) acc[e] = vrow[c];
            }
            continue;
        }
        const m_new = @max(m, s);
        const alpha = exp(m - m_new);
        const p = exp(s - m_new);
        l = l * alpha + p;
        m = m_new;
        for (0..max_lane_dims) |e| {
            const c = lane() + @as(u32, @intCast(e)) * 32;
            if (c < dims.d) acc[e] = acc[e] * alpha + p * vrow[c];
        }
    }
    const orow = out + ((b * dims.tq + t) * dims.h + hd) * dims.d;
    for (0..max_lane_dims) |e| {
        const c = lane() + @as(u32, @intCast(e)) * 32;
        if (c < dims.d) orow[c] = acc[e] / l;
    }
    if (has_lse != 0 and lane() == 0) lse[(b * dims.h + hd) * dims.tq + t] = m + log(l);
}

/// `delta[b, h, t] = <dout, out>` over the head dimension; one thread per row.
fn attentionDeltaF32(delta: [*]f32, dout: [*]const f32, out: [*]const f32, dims: AttnDims) callconv(.kernel) void {
    const idx = index();
    if (idx >= dims.b * dims.tq * dims.h) return;
    const hd = idx % dims.h;
    const t = (idx / dims.h) % dims.tq;
    const b = idx / (dims.h * dims.tq);
    var sum: f32 = 0;
    for (0..dims.d) |c| sum += dout[idx * dims.d + c] * out[idx * dims.d + c];
    delta[(b * dims.h + hd) * dims.tq + t] = sum;
}

/// `dq = scale * sum_j p_ij (dout_i . v_j - delta_i) k_j`; one warp per query.
fn attentionDqF32(dq: [*]f32, dout: [*]const f32, q: [*]const f32, k: [*]const f32, v: [*]const f32, lse: [*]const f32, delta: [*]const f32, dims: AttnDims, scale: f32) callconv(.kernel) void {
    const w = warpIndex();
    if (w >= dims.b * dims.tq * dims.h) return;
    const hd = w % dims.h;
    const t = (w / dims.h) % dims.tq;
    const b = w / (dims.h * dims.tq);
    const kvh = hd / (dims.h / dims.hkv);
    const i = dims.keys - dims.tq + t;
    const row = ((b * dims.tq + t) * dims.h + hd) * dims.d;
    const stat = (b * dims.h + hd) * dims.tq + t;
    const lse_i = lse[stat];
    const delta_i = delta[stat];
    var acc: [max_lane_dims]f32 = @splat(0);
    const lo = if (i > dims.window) i - dims.window else 0;
    var j = lo;
    while (j <= i and j < dims.keys) : (j += 1) {
        const kv = ((b * dims.tk + j) * dims.hkv + kvh) * dims.d;
        const p = exp(warpDot(q + row, k + kv, dims.d) * scale - lse_i);
        const ds = p * (warpDot(dout + row, v + kv, dims.d) - delta_i);
        for (0..max_lane_dims) |e| {
            const c = lane() + @as(u32, @intCast(e)) * 32;
            if (c < dims.d) acc[e] += ds * k[kv + c];
        }
    }
    for (0..max_lane_dims) |e| {
        const c = lane() + @as(u32, @intCast(e)) * 32;
        if (c < dims.d) dq[row + c] = scale * acc[e];
    }
}

/// `dv_j = sum_i p_ij dout_i`, `dk_j = scale * sum_i ds_ij q_i` over every
/// query head of the kv head's group; one warp per key row (written, not added;
/// rows past `keys` get zero).
fn attentionDkvF32(dk: [*]f32, dv: [*]f32, dout: [*]const f32, q: [*]const f32, k: [*]const f32, v: [*]const f32, lse: [*]const f32, delta: [*]const f32, dims: AttnDims, scale: f32) callconv(.kernel) void {
    const w = warpIndex();
    if (w >= dims.b * dims.tk * dims.hkv) return;
    const kvh = w % dims.hkv;
    const j = (w / dims.hkv) % dims.tk;
    const b = w / (dims.hkv * dims.tk);
    const kv = ((b * dims.tk + j) * dims.hkv + kvh) * dims.d;
    var dk_acc: [max_lane_dims]f32 = @splat(0);
    var dv_acc: [max_lane_dims]f32 = @splat(0);
    const off = dims.keys - dims.tq;
    if (j < dims.keys) {
        const group = dims.h / dims.hkv;
        const i_hi = @min(j + dims.window, dims.keys - 1);
        for (0..group) |hh| {
            const hd = kvh * group + @as(u32, @intCast(hh));
            var i = @max(j, off);
            while (i <= i_hi) : (i += 1) {
                const t = i - off;
                const row = ((b * dims.tq + t) * dims.h + hd) * dims.d;
                const stat = (b * dims.h + hd) * dims.tq + t;
                const p = exp(warpDot(q + row, k + kv, dims.d) * scale - lse[stat]);
                const ds = p * (warpDot(dout + row, v + kv, dims.d) - delta[stat]);
                for (0..max_lane_dims) |e| {
                    const c = lane() + @as(u32, @intCast(e)) * 32;
                    if (c < dims.d) {
                        dv_acc[e] += p * dout[row + c];
                        dk_acc[e] += ds * q[row + c];
                    }
                }
            }
        }
    }
    for (0..max_lane_dims) |e| {
        const c = lane() + @as(u32, @intCast(e)) * 32;
        if (c < dims.d) {
            dk[kv + c] = scale * dk_acc[e];
            dv[kv + c] = dv_acc[e];
        }
    }
}

// ---------------------------------------------------------------------------
// Optimizer (CpuBackend.adamwStep, muon*; norms and sums in f64 as there)

fn adamwStepF32(p: [*]f32, g: [*]const f32, m: [*]f32, v: [*]f32, decay: f32, w1: f32, w2: f32, bias2: f32, step_size: f32, eps: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    var pv = p[i] * decay;
    const gv = g[i];
    const mv = lerp(m[i], gv, w1);
    const vv = lerp(v[i], gv * gv, w2);
    const denom = @sqrt(vv / bias2) + eps;
    pv += -step_size * (mv / denom);
    p[i] = pv;
    m[i] = mv;
    v[i] = vv;
}

fn muonMomentumF32(g: [*]f32, buf: [*]f32, momentum: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const bv = lerp(buf[i], g[i], 1 - momentum);
    buf[i] = bv;
    g[i] = lerp(g[i], bv, momentum);
}

/// `out[r]` = row `r`'s sum of squares in f64; one block per row.
fn rowSquaresF64(x: [*]const f32, out: [*]f64, cols: u32) callconv(.kernel) void {
    const row = x + block(0) * cols;
    var sum: f64 = 0;
    var c = tid();
    while (c < cols) : (c += threads()) sum += @as(f64, row[c]) * @as(f64, row[c]);
    const total = blockSumF64(sum);
    if (tid() == 0) out[block(0)] = total;
}

/// `out[c]` = column `c`'s sum of squares in f64, rows in order; one thread per column.
fn colSquaresF64(x: [*]const f32, out: [*]f64, rows: u32, cols: u32) callconv(.kernel) void {
    const c = index();
    if (c >= cols) return;
    var sum: f64 = 0;
    for (0..rows) |r| sum += @as(f64, x[r * cols + c]) * @as(f64, x[r * cols + c]);
    out[c] = sum;
}

/// `out[0] = sum(values[0 .. count])` in f64; one block.
fn totalF64(values: [*]const f64, out: [*]f64, count: u32) callconv(.kernel) void {
    var sum: f64 = 0;
    var i = tid();
    while (i < count) : (i += threads()) sum += values[i];
    const total = blockSumF64(sum);
    if (tid() == 0) out[0] = total;
}

/// MuonEq: every row to the mean row norm `sqrt(total) / sqrt(rows)`.
fn muonScaleRowsF32(x: [*]f32, row_sq: [*]const f64, total: [*]const f64, rows: u32, cols: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const target: f32 = @floatCast(@sqrt(total[0]) / @sqrt(@as(f64, @floatFromInt(rows))));
    const row_norm = @max(@as(f32, @floatCast(@sqrt(row_sq[i / cols]))), 1e-6);
    x[i] *= target / row_norm;
}

/// `x /= ||x||_F * 1.01 + 1e-6`.
fn muonScaleAllF32(x: [*]f32, total: [*]const f64, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const norm: f32 = @floatCast(@sqrt(total[0]));
    x[i] /= norm * 1.01 + 1e-6;
}

/// Muon+: `g *= target / max(||g||_F, 1e-6)`.
fn muonRenormF32(g: [*]f32, total: [*]const f64, target: f32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    g[i] *= target / @max(@as(f32, @floatCast(@sqrt(total[0]))), 1e-6);
}

/// NorMuon: the second-moment EMA (in place) and the norm-preserving ratio; one block.
fn normuonStatsF32(sums: [*]const f64, second: [*]f32, ratio: [*]f32, count: u32, red: f32, beta2: f32) callconv(.kernel) void {
    var norm_sq: f64 = 0;
    var new_sq: f64 = 0;
    var i = tid();
    while (i < count) : (i += threads()) {
        const mean: f32 = @floatCast(sums[i] / @as(f64, red));
        norm_sq += mean;
        const s = lerp(second[i], mean, 1 - beta2);
        second[i] = s;
        const step = 1 / @sqrt(@max(s, 1e-10));
        new_sq += @as(f64, mean * red * step * step);
    }
    const v_norm_sq = blockSumF64(norm_sq);
    const v_new_sq = blockSumF64(new_sq);
    if (tid() == 0) {
        const v_norm: f32 = @floatCast(@sqrt(v_norm_sq * @as(f64, red)));
        const v_norm_new: f32 = @floatCast(@sqrt(v_new_sq));
        ratio[0] = v_norm / @max(v_norm_new, 1e-10);
    }
}

/// NorMuon scaling, then the cautious update `p -= lr * g + lr * wd * p * [g * p >= 0]`.
fn muonUpdateF32(p: [*]f32, g: [*]f32, second: [*]const f32, ratio: [*]const f32, lr: f32, lr_wd: f32, cols: u32, by_row: u32, n: u32) callconv(.kernel) void {
    const i = index();
    if (i >= n) return;
    const idx = if (by_row != 0) i / cols else i % cols;
    const gv = g[i] * ((1 / @sqrt(@max(second[idx], 1e-10))) * ratio[0]);
    g[i] = gv;
    const pv = p[i];
    const mask: f32 = if (gv * pv >= 0) 1 else 0;
    p[i] = pv - (lr * gv + lr_wd * pv * mask);
}
