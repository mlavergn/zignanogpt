// Metal compute kernels of MetalBackend (src/metal/backend.zig), compiled from
// source at startup. Each kernel mirrors one CpuBackend op; see its doc there.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------- elementwise

kernel void fill_f32(device float* out [[buffer(0)]], constant float& value [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = value;
}

kernel void copy_u32(device uint* out [[buffer(0)]], device const uint* src [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = src[i];
}

kernel void add_f32(device float* out [[buffer(0)]], device const float* a [[buffer(1)]], device const float* b [[buffer(2)]],
                    constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

kernel void mul_f32(device float* out [[buffer(0)]], device const float* a [[buffer(1)]], device const float* b [[buffer(2)]],
                    constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] * b[i];
}

kernel void scale_f32(device float* out [[buffer(0)]], device const float* a [[buffer(1)]], constant float& s [[buffer(2)]],
                      constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] * s;
}

// A device scalar: factor * value[0] when `has` is set, else factor.
struct ScalarArgs {
    float x_factor;
    float y_factor;
    uint x_has;
    uint y_has;
    uint n;
};

kernel void combine_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], device const float* y [[buffer(2)]],
                        device const float* xs [[buffer(3)]], device const float* ys [[buffer(4)]],
                        constant ScalarArgs& p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float a = p.x_has ? xs[0] * p.x_factor : p.x_factor;
    const float b = p.y_has ? ys[0] * p.y_factor : p.y_factor;
    out[i] = a * x[i] + b * y[i];
}

kernel void relu_square_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float r = max(x[i], 0.0f);
    out[i] = r * r;
}

kernel void relu_square_backward_f32(device float* dx [[buffer(0)]], device const float* dy [[buffer(1)]], device const float* x [[buffer(2)]],
                                     constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dx[i] = dy[i] * 2.0f * max(x[i], 0.0f);
}

struct SoftcapArgs {
    uint cols;
    uint padded;
    float cap;
    uint n;
};

kernel void softcap_f32(device float* out [[buffer(0)]], device const float* logits [[buffer(1)]],
                        constant SoftcapArgs& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint r = i / p.cols;
    const uint c = i % p.cols;
    out[i] = p.cap * precise::tanh(logits[r * p.padded + c] / p.cap);
}

struct EmbeddingArgs {
    uint cols;
    uint n;
};

kernel void embedding_f32(device float* out [[buffer(0)]], device const float* table [[buffer(1)]], device const int* ids [[buffer(2)]],
                          constant EmbeddingArgs& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    out[i] = table[uint(ids[i / p.cols]) * p.cols + i % p.cols];
}

// ---------------------------------------------------------------- row norms

struct NormArgs {
    uint cols;
    float eps;
    uint accumulate;
};

// Sums `v` over the threadgroup (at most 1024 threads, 32 per simdgroup).
static float block_sum(float v, threadgroup float* shared, uint tid, uint lane, uint sg, uint groups) {
    v = simd_sum(v);
    if (lane == 0) shared[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < groups; g++) total += shared[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// One threadgroup per row.
kernel void rmsnorm_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], constant NormArgs& p [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    device const float* xr = x + row * p.cols;
    float sq = 0.0f;
    for (uint c = tid; c < p.cols; c += threads) sq += xr[c] * xr[c];
    const float total = block_sum(sq, shared, tid, lane, sg, (threads + 31) / 32);
    const float inv = 1.0f / sqrt(total / float(p.cols) + p.eps);
    device float* outr = out + row * p.cols;
    for (uint c = tid; c < p.cols; c += threads) outr[c] = xr[c] * inv;
}

kernel void rmsnorm_backward_f32(device float* dx [[buffer(0)]], device const float* dy [[buffer(1)]], device const float* x [[buffer(2)]],
                                 constant NormArgs& p [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    device const float* xr = x + row * p.cols;
    device const float* gr = dy + row * p.cols;
    float sq = 0.0f, gx = 0.0f;
    for (uint c = tid; c < p.cols; c += threads) {
        sq += xr[c] * xr[c];
        gx += gr[c] * xr[c];
    }
    const uint groups = (threads + 31) / 32;
    const float sq_total = block_sum(sq, shared, tid, lane, sg, groups);
    const float gx_total = block_sum(gx, shared, tid, lane, sg, groups);
    const float n = float(p.cols);
    const float inv = 1.0f / sqrt(sq_total / n + p.eps);
    const float k = inv * inv * inv * gx_total / n;
    device float* out = dx + row * p.cols;
    for (uint c = tid; c < p.cols; c += threads) {
        const float d = inv * gr[c] - xr[c] * k;
        out[c] = p.accumulate ? out[c] + d : d;
    }
}

// ---------------------------------------------------------------- matmul

struct MatmulArgs {
    uint m;
    uint n;
    uint k;
    uint transpose_a;
    uint transpose_b;
    uint accumulate;
    float alpha;
};

constant constexpr uint TM = 64;
constant constexpr uint TN = 64;
constant constexpr uint TK = 16;

// C (+)= alpha * op(A) @ op(B), 64x64 tiles per threadgroup of 4 simdgroups
// (each 32x32 = 4x4 blocks of 8x8 simdgroup matrices), K in steps of 16
// staged through threadgroup memory; the transposes are applied while staging.
kernel void matmul_f32(device float* C [[buffer(0)]], device const float* A [[buffer(1)]], device const float* B [[buffer(2)]],
                       constant MatmulArgs& p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[TM * TK];
    threadgroup float Bs[TK * TN];
    const uint row0 = tg.y * TM;
    const uint col0 = tg.x * TN;
    const uint sr = (sg / 2) * 32;
    const uint sc = (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.k; k0 += TK) {
        for (uint e = tid; e < TM * TK; e += 128) {
            const uint i = e / TK, kk = e % TK;
            const uint gi = row0 + i, gk = k0 + kk;
            float v = 0.0f;
            if (gi < p.m && gk < p.k) v = p.transpose_a ? A[gk * p.m + gi] : A[gi * p.k + gk];
            As[e] = v;
        }
        for (uint e = tid; e < TK * TN; e += 128) {
            const uint kk = e / TN, j = e % TN;
            const uint gk = k0 + kk, gj = col0 + j;
            float v = 0.0f;
            if (gk < p.k && gj < p.n) v = p.transpose_b ? B[gj * p.k + gk] : B[gk * p.n + gj];
            Bs[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sr + i * 8) * TK + kk, TK);
            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Bs + kk * TN + sc + j * 8, TN);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Through threadgroup memory (reusing As/Bs space is too small: 64x64 floats).
    threadgroup float Cs[TM * TN];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * TN + sc + j * 8, TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < TM * TN; e += 128) {
        const uint gi = row0 + e / TN, gj = col0 + e % TN;
        if (gi >= p.m || gj >= p.n) continue;
        const float v = p.alpha * Cs[e];
        device float* dst = C + gi * p.n + gj;
        *dst = p.accumulate ? *dst + v : v;
    }
}

// ---------------------------------------------------------------- rotary, gates

struct RopeArgs {
    uint t;
    uint h;
    uint half_d;
    uint pos0;
    float sign;
    uint n; // rows * half_d
};

kernel void rope_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], device const float* cos_t [[buffer(2)]],
                     device const float* sin_t [[buffer(3)]], constant RopeArgs& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint r = i / p.half_d, j = i % p.half_d;
    const uint pos = p.pos0 + (r / p.h) % p.t;
    const float c = cos_t[pos * p.half_d + j];
    const float s = p.sign * sin_t[pos * p.half_d + j];
    const uint base = r * 2 * p.half_d;
    const float x1 = x[base + j], x2 = x[base + p.half_d + j];
    out[base + j] = x1 * c + x2 * s;
    out[base + p.half_d + j] = x2 * c - x1 * s;
}

struct RopeNormArgs {
    uint t;
    uint h;
    uint half_d;
    uint pos0;
    float eps;
    float gain;
};

// rope_norm: one threadgroup per [B, T, H] row. Rotates x in place, then
// out = gain * x / rms(x) (rotation keeps the row's norm, so one pass serves).
// out may alias x: each thread reads and writes only its own pairs.
kernel void rope_norm_f32(device float* out [[buffer(0)]], device float* x [[buffer(1)]], device const float* cos_t [[buffer(2)]],
                          device const float* sin_t [[buffer(3)]], constant RopeNormArgs& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const uint pos = p.pos0 + (row / p.h) % p.t;
    device float* xr = x + row * 2 * p.half_d;
    device float* o = out + row * 2 * p.half_d;
    float sq = 0.0f;
    for (uint j = tid; j < p.half_d; j += threads) {
        const float c = cos_t[pos * p.half_d + j], s = sin_t[pos * p.half_d + j];
        const float x1 = xr[j], x2 = xr[p.half_d + j];
        const float r1 = x1 * c + x2 * s, r2 = x2 * c - x1 * s;
        xr[j] = r1;
        xr[p.half_d + j] = r2;
        sq += r1 * r1 + r2 * r2;
    }
    const float total = block_sum(sq, shared, tid, lane, sg, (threads + 31) / 32);
    const float inv = 1.0f / sqrt(total / float(2 * p.half_d) + p.eps);
    for (uint j = tid; j < p.half_d; j += threads) {
        const float r1 = xr[j], r2 = xr[p.half_d + j];
        o[j] = (r1 * inv) * p.gain;
        o[p.half_d + j] = (r2 * inv) * p.gain;
    }
}

// Backward of rope_norm: dx = rope^-1(rmsnorm_backward(gain * dy, x)), x the
// rotated input. dx may alias dy: the sums finish (barrier) before any write,
// and each thread writes only the pairs it read.
kernel void rope_norm_backward_f32(device float* dx [[buffer(0)]], device const float* dy [[buffer(1)]], device const float* x [[buffer(2)]],
                                   device const float* cos_t [[buffer(3)]], device const float* sin_t [[buffer(4)]], constant RopeNormArgs& p [[buffer(5)]],
                                   uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const uint pos = p.pos0 + (row / p.h) % p.t;
    device const float* xr = x + row * 2 * p.half_d;
    device const float* gr = dy + row * 2 * p.half_d;
    float sq = 0.0f, gx = 0.0f;
    for (uint j = tid; j < p.half_d; j += threads) {
        const float x1 = xr[j], x2 = xr[p.half_d + j];
        const float g1 = gr[j] * p.gain, g2 = gr[p.half_d + j] * p.gain;
        sq += x1 * x1 + x2 * x2;
        gx += g1 * x1 + g2 * x2;
    }
    const uint groups = (threads + 31) / 32;
    const float sq_total = block_sum(sq, shared, tid, lane, sg, groups);
    const float gx_total = block_sum(gx, shared, tid, lane, sg, groups);
    const float n = float(2 * p.half_d);
    const float inv = 1.0f / sqrt(sq_total / n + p.eps);
    const float k = inv * inv * inv * gx_total / n;
    device float* out = dx + row * 2 * p.half_d;
    for (uint j = tid; j < p.half_d; j += threads) {
        const float x1 = xr[j], x2 = xr[p.half_d + j];
        const float d1 = inv * (gr[j] * p.gain) - x1 * k;
        const float d2 = inv * (gr[p.half_d + j] * p.gain) - x2 * k;
        const float c = cos_t[pos * p.half_d + j], s = -sin_t[pos * p.half_d + j];
        out[j] = d1 * c + d2 * s;
        out[p.half_d + j] = d2 * c - d1 * s;
    }
}

struct GateArgs {
    uint cols;
    uint cin;
    uint heads;
    uint rows;
};

static float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

kernel void gate_linear_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], device const float* w [[buffer(2)]],
                            constant GateArgs& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.heads) return;
    const uint r = i / p.heads, h = i % p.heads;
    float sum = 0.0f;
    for (uint c = 0; c < p.cin; c++) sum += x[r * p.cols + c] * w[h * p.cin + c];
    out[i] = sum;
}

kernel void gate_linear_backward_dx_f32(device float* dx [[buffer(0)]], device const float* dout [[buffer(1)]], device const float* w [[buffer(2)]],
                                        constant GateArgs& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.cin) return;
    const uint r = i / p.cin, c = i % p.cin;
    float sum = 0.0f;
    for (uint h = 0; h < p.heads; h++) sum += dout[r * p.heads + h] * w[h * p.cin + c];
    dx[r * p.cols + c] += sum;
}

// One threadgroup per weight: the rows split across its threads, then a
// fixed-order tree sum (deterministic).
kernel void gate_linear_backward_dw_f32(device float* dw [[buffer(0)]], device const float* dout [[buffer(1)]], device const float* x [[buffer(2)]],
                                        constant GateArgs& p [[buffer(3)]], uint i [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const uint h = i / p.cin, c = i % p.cin;
    float sum = 0.0f;
    for (uint r = tid; r < p.rows; r += threads) sum += dout[r * p.heads + h] * x[r * p.cols + c];
    sum = simd_sum(sum);
    if (lane == 0) shared[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint g = 0; g < (threads + 31) / 32; g++) total += shared[g];
        dw[i] += total;
    }
}

struct SmearArgs {
    uint t;
    uint cols;
    float factor;
    uint has_lambda;
    uint n;
};

kernel void smear_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], device const float* gate [[buffer(2)]],
                      device const float* lambda [[buffer(3)]], constant SmearArgs& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint r = i / p.cols;
    if (r % p.t == 0) {
        out[i] = x[i];
        return;
    }
    const float l = p.has_lambda ? lambda[0] * p.factor : p.factor;
    out[i] = x[i] + l * sigmoidf(gate[r]) * x[i - p.cols];
}

// Gated add: out[r] = x[r] + s * sigmoid(gate[r]) * y[r].
kernel void gated_add_f32(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], device const float* y [[buffer(2)]],
                          device const float* gate [[buffer(3)]], device const float* scalar [[buffer(4)]], constant SmearArgs& p [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint r = i / p.cols;
    const float s = p.has_lambda ? scalar[0] * p.factor : p.factor;
    out[i] = x[i] + s * sigmoidf(gate[r]) * y[i];
}

// One threadgroup per row: dx, dgate and the row's share of dlambda.
kernel void smear_backward_f32(device float* dx [[buffer(0)]], device float* dgate [[buffer(1)]], device float* partial [[buffer(2)]],
                               device const float* dout [[buffer(3)]], device const float* x [[buffer(4)]], device const float* gate [[buffer(5)]],
                               device const float* lambda [[buffer(6)]], constant SmearArgs& p [[buffer(7)]],
                               uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const float l = p.has_lambda ? lambda[0] * p.factor : p.factor;
    const uint pos = r % p.t;
    device const float* g = dout + r * p.cols;
    device float* out = dx + r * p.cols;
    const bool has_next = pos + 1 < p.t;
    const float next = has_next ? l * sigmoidf(gate[r + 1]) : 0.0f;
    for (uint c = tid; c < p.cols; c += threads) out[c] = has_next ? g[c] + next * g[c + p.cols] : g[c];
    float s = 0.0f;
    if (pos > 0)
        for (uint c = tid; c < p.cols; c += threads) s += g[c] * x[(r - 1) * p.cols + c];
    const float total = block_sum(s, shared, tid, lane, sg, (threads + 31) / 32);
    if (tid == 0) {
        if (pos == 0) {
            dgate[r] = 0.0f;
            partial[r] = 0.0f;
        } else {
            const float sgm = sigmoidf(gate[r]);
            dgate[r] = l * sgm * (1.0f - sgm) * total;
            partial[r] = sgm * total;
        }
    }
}

struct MixArgs {
    uint d;
    uint pairs;
};

kernel void value_mix_f32(device float* v [[buffer(0)]], device const float* ve [[buffer(1)]], device const float* gate [[buffer(2)]],
                          constant MixArgs& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.pairs * p.d) return;
    v[i] += 3.0f * sigmoidf(gate[i / p.d]) * ve[i];
}

kernel void value_mix_backward_f32(device float* dve [[buffer(0)]], device float* dgate [[buffer(1)]], device const float* dv [[buffer(2)]],
                                   device const float* ve [[buffer(3)]], device const float* gate [[buffer(4)]], constant MixArgs& p [[buffer(5)]],
                                   uint rh [[thread_position_in_grid]]) {
    if (rh >= p.pairs) return;
    const float s = sigmoidf(gate[rh]);
    float sum = 0.0f;
    for (uint c = 0; c < p.d; c++) {
        const float g = dv[rh * p.d + c];
        dve[rh * p.d + c] = 3.0f * s * g;
        sum += g * ve[rh * p.d + c];
    }
    dgate[rh] = 3.0f * s * (1.0f - s) * sum;
}

// ---------------------------------------------------------------- cross entropy

struct XentArgs {
    uint vocab;
    uint padded;
    float cap;
    float factor;
    uint has_weights;
    uint has_lse;
};

// The soft cap plus each row's log-sum-exp of the capped values, in one pass
// over the row (one threadgroup per row). Each thread keeps an online max and
// sum; the finite floor keeps fast math away from infinities.
kernel void softcap_lse_f32(device float* out [[buffer(0)]], device const float* logits [[buffer(1)]], device float* lse [[buffer(2)]],
                            constant SoftcapArgs& p [[buffer(3)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared_m[32];
    threadgroup float shared_s[32];
    device const float* src = logits + r * p.padded;
    device float* dst = out + r * p.cols;
    float m = -1.0e30f, s = 0.0f;
    for (uint c = tid; c < p.cols; c += threads) {
        const float z = p.cap * precise::tanh(src[c] / p.cap);
        dst[c] = z;
        if (z > m) {
            s = s * exp(m - z) + 1.0f;
            m = z;
        } else {
            s += exp(z - m);
        }
    }
    for (ushort offset = 16; offset > 0; offset /= 2) {
        const float m2 = simd_shuffle_down(m, offset);
        const float s2 = simd_shuffle_down(s, offset);
        const float mm = max(m, m2);
        s = s * exp(m - mm) + s2 * exp(m2 - mm);
        m = mm;
    }
    if (lane == 0) {
        shared_m[sg] = m;
        shared_s[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mt = -1.0e30f;
        for (uint g = 0; g < (threads + 31) / 32; g++) mt = max(mt, shared_m[g]);
        float st = 0.0f;
        for (uint g = 0; g < (threads + 31) / 32; g++) st += shared_s[g] * exp(shared_m[g] - mt);
        lse[r] = mt + log(st);
    }
}

// losses[r] = lse[r] - logits[r, targets[r]], 0 where the target is -1.
kernel void xent_rows_lse_f32(device float* losses [[buffer(0)]], device const float* logits [[buffer(1)]], device const int* targets [[buffer(2)]],
                              device const float* lse [[buffer(3)]], constant uint2& p [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    if (r >= p.y) return;
    const int t = targets[r];
    losses[r] = t < 0 ? 0.0f : lse[r] - logits[r * p.x + uint(t)];
}

// One threadgroup per row: the row's log-sum-exp.
static float block_lse(device const float* row, uint vocab, threadgroup float* shared, uint tid, uint threads, uint lane, uint sg) {
    const uint groups = (threads + 31) / 32;
    float m = -INFINITY;
    for (uint c = tid; c < vocab; c += threads) m = max(m, row[c]);
    m = simd_max(m);
    if (lane == 0) shared[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint g = 0; g < groups; g++) mx = max(mx, shared[g]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint c = tid; c < vocab; c += threads) s += exp(row[c] - mx);
    return mx + log(block_sum(s, shared, tid, lane, sg, groups));
}

kernel void xent_rows_f32(device float* losses [[buffer(0)]], device const float* logits [[buffer(1)]], device const int* targets [[buffer(2)]],
                          constant XentArgs& p [[buffer(3)]],
                          uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const int t = targets[r];
    if (t < 0) {
        if (tid == 0) losses[r] = 0.0f;
        return;
    }
    device const float* row = logits + r * p.vocab;
    const float lse = block_lse(row, p.vocab, shared, tid, threads, lane, sg);
    if (tid == 0) losses[r] = lse - row[t];
}

// Gradient through the soft cap: weights[r] (or factor) * (softmax - onehot) * (1 - (z / cap)^2).
kernel void xent_backward_f32(device float* dpad [[buffer(0)]], device const float* logits [[buffer(1)]], device const int* targets [[buffer(2)]],
                              device const float* weights [[buffer(3)]], constant XentArgs& p [[buffer(4)]], device const float* row_lse [[buffer(5)]],
                              uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    device float* out = dpad + r * p.padded;
    const int t = targets[r];
    if (t < 0) {
        for (uint c = tid; c < p.padded; c += threads) out[c] = 0.0f;
        return;
    }
    device const float* row = logits + r * p.vocab;
    const float lse = p.has_lse ? row_lse[r] : block_lse(row, p.vocab, shared, tid, threads, lane, sg);
    const float f = p.has_weights ? weights[r] : p.factor;
    for (uint c = tid; c < p.padded; c += threads) {
        if (c >= p.vocab) {
            out[c] = 0.0f;
            continue;
        }
        const float z = row[c];
        const float squash = z / p.cap;
        out[c] = (exp(z - lse) - (int(c) == t ? 1.0f : 0.0f)) * f * (1.0f - squash * squash);
    }
}

// ---------------------------------------------------------------- attention

// Flash attention (CpuAttention / CpuAttentionBackward semantics) on 8x8
// simdgroup matrices. A threadgroup is 4 simdgroups; each owns 8 query rows
// (forward, dq) or 8 key rows (dk/dv). Operands are staged through
// threadgroup memory with the head dimension zero-padded to DP, so one
// instantiation per padded size covers any head dimension up to 128.
struct AttnArgs {
    uint b;
    uint tq;
    uint tk;
    uint keys;
    uint h;
    uint hkv;
    uint d;
    uint window; // clamped to 2^30
    float scale;
    uint has_lse;
};

// Masked scores; finite, since fast math assumes no infinities.
constant constexpr float kMasked = -1.0e30f;

inline bool attn_visible(uint i, uint j, constant AttnArgs& p) {
    return j < p.keys && j <= i && i - j <= p.window;
}

// Stages `rows` rows of a [*, stride] operand starting at row r0 (rows past
// `limit` and columns past d read as zero) into a [rows, DP] tile.
template <uint DP, uint THREADS>
inline void attn_stage(threadgroup float* tile, device const float* base, uint stride, uint d, uint r0, uint rows, uint limit, uint tid) {
    for (uint e = tid; e < rows * DP; e += THREADS) {
        const uint r = e / DP, c = e % DP, row = r0 + r;
        tile[e] = (row < limit && c < d) ? base[row * stride + c] : 0.0f;
    }
}

// out[b, t, h] = softmax(scale q k^T) v over the visible keys; lse = m + log(l).
template <uint DP, uint BK, uint SG>
inline void attention_body(device float* out, device const float* q, device const float* k, device const float* v,
                           device float* lse, constant AttnArgs& p, threadgroup float* tile, threadgroup float* scratch,
                           threadgroup float* diag, uint3 tg, uint tid, uint sg, uint lane) {
    constexpr uint BQ = 8 * SG, ND = DP / 8, NK = BK / 8;
    const uint b = tg.z, h = tg.y, q0 = tg.x * BQ;
    const uint kvh = h / (p.h / p.hkv);
    const uint off = p.keys - p.tq;
    const uint qstride = p.h * p.d, kstride = p.hkv * p.d;
    device const float* qb = q + (b * p.tq * p.h + h) * p.d;
    device const float* kb = k + (b * p.tk * p.hkv + kvh) * p.d;
    device const float* vb = v + (b * p.tk * p.hkv + kvh) * p.d;

    attn_stage<DP, 32 * SG>(tile, qb, qstride, p.d, q0, BQ, p.tq, tid);
    threadgroup float* dg = diag + sg * 64;
    for (uint e = lane; e < 64; e += 32) dg[e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 qr[ND], o[ND];
    for (uint i = 0; i < ND; i++) {
        simdgroup_load(qr[i], tile + sg * 8 * DP + i * 8, DP);
        o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup float* s = scratch + sg * 8 * BK;
    const uint row = lane / 4, part = lane % 4;
    const uint t_row = q0 + sg * 8 + row;
    const bool row_ok = t_row < p.tq;
    const uint i_row = off + t_row;
    float m = kMasked, l = 0.0f;

    const uint i_first = off + q0, i_last = off + min(q0 + BQ, p.tq) - 1;
    const uint j_lo = i_first > p.window ? i_first - p.window : 0;
    const uint j_hi = min(i_last, p.keys - 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j0 = j_lo; j0 <= j_hi; j0 += BK) {
        attn_stage<DP, 32 * SG>(tile, kb, kstride, p.d, j0, BK, j_hi + 1, tid);
        attn_stage<DP, 32 * SG>(tile + BK * DP, vb, kstride, p.d, j0, BK, j_hi + 1, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < NK; n++) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint i = 0; i < ND; i++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, tile + n * 8 * DP + i * 8, DP, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qr[i], kt, acc);
            }
            simdgroup_store(acc, s + n * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax: 4 lanes per row.
        float mx = kMasked;
        for (uint c = part; c < BK; c += 4) {
            const uint j = j0 + c;
            const float x = (row_ok && attn_visible(i_row, j, p)) ? s[row * BK + c] * p.scale : kMasked;
            s[row * BK + c] = x;
            mx = max(mx, x);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m, mx);
        float sum = 0.0f;
        for (uint c = part; c < BK; c += 4) {
            const float x = s[row * BK + c];
            const float pr = x > 0.5f * kMasked ? exp(x - m_new) : 0.0f;
            s[row * BK + c] = pr;
            sum += pr;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        const float alpha = m > 0.5f * kMasked ? exp(m - m_new) : 0.0f;
        l = l * alpha + sum;
        m = m_new;
        if (part == 0) dg[row * 9] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a;
        simdgroup_load(a, dg, 8);
        for (uint i = 0; i < ND; i++) simdgroup_multiply(o[i], a, o[i]);
        for (uint n = 0; n < NK; n++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, s + n * 8, BK);
            for (uint i = 0; i < ND; i++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, tile + BK * DP + n * 8 * DP + i * 8, DP);
                simdgroup_multiply_accumulate(o[i], pm, vm, o[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0) dg[row * 9] = l > 0.0f ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 a;
    simdgroup_load(a, dg, 8);
    threadgroup float* rows = tile + sg * 8 * DP;
    for (uint i = 0; i < ND; i++) {
        simdgroup_multiply(o[i], a, o[i]);
        simdgroup_store(o[i], rows + i * 8, DP);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8 * DP; e += 32) {
        const uint r = e / DP, c = e % DP, t = q0 + sg * 8 + r;
        if (t < p.tq && c < p.d) out[((b * p.tq + t) * p.h + h) * p.d + c] = rows[e];
    }
    if (p.has_lse && part == 0 && row_ok) lse[(b * p.h + h) * p.tq + t_row] = m + log(l);
}

// delta[b, h, t] = <dout, out> over the head dimension.
kernel void attention_delta_f32(device float* delta [[buffer(0)]], device const float* dout [[buffer(1)]], device const float* out [[buffer(2)]],
                                constant AttnArgs& p [[buffer(3)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= p.b * p.tq * p.h) return;
    const uint h = idx % p.h, t = (idx / p.h) % p.tq, b = idx / (p.h * p.tq);
    float sum = 0.0f;
    for (uint c = 0; c < p.d; c++) sum += dout[idx * p.d + c] * out[idx * p.d + c];
    delta[(b * p.h + h) * p.tq + t] = sum;
}

// dq = scale * sum_j p_ij (dout_i . v_j - delta_i) k_j, per block of 32 queries.
template <uint DP, uint BK, uint SG>
inline void attention_dq_body(device float* dq, device const float* dout, device const float* q, device const float* k,
                              device const float* v, device const float* lse, device const float* delta, constant AttnArgs& p,
                              threadgroup float* tile, threadgroup float* scratch, uint3 tg, uint tid, uint sg, uint lane) {
    constexpr uint BQ = 8 * SG, ND = DP / 8, NK = BK / 8;
    const uint b = tg.z, h = tg.y, q0 = tg.x * BQ;
    const uint kvh = h / (p.h / p.hkv);
    const uint off = p.keys - p.tq;
    const uint qstride = p.h * p.d, kstride = p.hkv * p.d;
    device const float* kb = k + (b * p.tk * p.hkv + kvh) * p.d;
    device const float* vb = v + (b * p.tk * p.hkv + kvh) * p.d;
    const uint head = (b * p.tq * p.h + h) * p.d;

    simdgroup_float8x8 qr[ND], gr[ND], acc[ND];
    attn_stage<DP, 32 * SG>(tile, q + head, qstride, p.d, q0, BQ, p.tq, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < ND; i++) simdgroup_load(qr[i], tile + sg * 8 * DP + i * 8, DP);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage<DP, 32 * SG>(tile, dout + head, qstride, p.d, q0, BQ, p.tq, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < ND; i++) {
        simdgroup_load(gr[i], tile + sg * 8 * DP + i * 8, DP);
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup float* s = scratch + sg * 16 * BK;
    threadgroup float* g = s + 8 * BK;
    const uint row = lane / 4, part = lane % 4;
    const uint t_row = q0 + sg * 8 + row;
    const bool row_ok = t_row < p.tq;
    const uint i_row = off + t_row;
    const float lse_row = row_ok ? lse[(b * p.h + h) * p.tq + t_row] : 0.0f;
    const float delta_row = row_ok ? delta[(b * p.h + h) * p.tq + t_row] : 0.0f;

    const uint i_first = off + q0, i_last = off + min(q0 + BQ, p.tq) - 1;
    const uint j_lo = i_first > p.window ? i_first - p.window : 0;
    const uint j_hi = min(i_last, p.keys - 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j0 = j_lo; j0 <= j_hi; j0 += BK) {
        attn_stage<DP, 32 * SG>(tile, kb, kstride, p.d, j0, BK, j_hi + 1, tid);
        attn_stage<DP, 32 * SG>(tile + BK * DP, vb, kstride, p.d, j0, BK, j_hi + 1, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < NK; n++) {
            simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 dp = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint i = 0; i < ND; i++) {
                simdgroup_float8x8 kt, vt;
                simdgroup_load(kt, tile + n * 8 * DP + i * 8, DP, ulong2(0, 0), true);
                simdgroup_load(vt, tile + BK * DP + n * 8 * DP + i * 8, DP, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sc, qr[i], kt, sc);
                simdgroup_multiply_accumulate(dp, gr[i], vt, dp);
            }
            simdgroup_store(sc, s + n * 8, BK);
            simdgroup_store(dp, g + n * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = part; c < BK; c += 4) {
            const uint j = j0 + c;
            const float pr = (row_ok && attn_visible(i_row, j, p)) ? exp(s[row * BK + c] * p.scale - lse_row) : 0.0f;
            s[row * BK + c] = pr * (g[row * BK + c] - delta_row);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < NK; n++) {
            simdgroup_float8x8 ds;
            simdgroup_load(ds, s + n * 8, BK);
            for (uint i = 0; i < ND; i++) {
                simdgroup_float8x8 km;
                simdgroup_load(km, tile + n * 8 * DP + i * 8, DP);
                simdgroup_multiply_accumulate(acc[i], ds, km, acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* rows = tile + sg * 8 * DP;
    for (uint i = 0; i < ND; i++) simdgroup_store(acc[i], rows + i * 8, DP);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8 * DP; e += 32) {
        const uint r = e / DP, c = e % DP, t = q0 + sg * 8 + r;
        if (t < p.tq && c < p.d) dq[((b * p.tq + t) * p.h + h) * p.d + c] = rows[e] * p.scale;
    }
}

// dv_j = sum_i p_ij dout_i, dk_j = scale * sum_i ds_ij q_i over every query
// head of the kv head's group, per block of 32 keys (written, not added).
template <uint DP, uint BQ, uint SG>
inline void attention_dkv_body(device float* dk, device float* dv, device const float* dout, device const float* q,
                               device const float* k, device const float* v, device const float* lse, device const float* delta,
                               constant AttnArgs& p, threadgroup float* tile, threadgroup float* scratch, threadgroup float* stats,
                               uint3 tg, uint tid, uint sg, uint lane) {
    constexpr uint BKEY = 8 * SG, ND = DP / 8, NQ = BQ / 8;
    const uint b = tg.z, kvh = tg.y, j0 = tg.x * BKEY;
    const uint off = p.keys - p.tq;
    const uint qstride = p.h * p.d, kstride = p.hkv * p.d;
    const uint kvbase = (b * p.tk * p.hkv + kvh) * p.d;

    simdgroup_float8x8 kr[ND], vr[ND], dka[ND], dva[ND];
    attn_stage<DP, 32 * SG>(tile, k + kvbase, kstride, p.d, j0, BKEY, p.tk, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < ND; i++) simdgroup_load(kr[i], tile + sg * 8 * DP + i * 8, DP);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage<DP, 32 * SG>(tile, v + kvbase, kstride, p.d, j0, BKEY, p.tk, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < ND; i++) {
        simdgroup_load(vr[i], tile + sg * 8 * DP + i * 8, DP);
        dka[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        dva[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* s = scratch + sg * 16 * BQ;
    threadgroup float* g = s + 8 * BQ;
    const uint row = lane / 4, part = lane % 4;
    const uint j_row = j0 + sg * 8 + row;

    // Queries that see a key of this block: positions j0 .. j_last + window.
    const uint j_last = min(j0 + BKEY, p.keys) - 1;
    const uint i_hi = min(j_last + p.window, p.keys - 1);
    if (j0 < p.keys && i_hi >= off) {
        const uint t_lo = j0 > off ? j0 - off : 0;
        const uint t_end = i_hi + 1 - off; // queries t < t_end
        const uint group = p.h / p.hkv;
        for (uint hh = 0; hh < group; hh++) {
            const uint h = kvh * group + hh;
            const uint head = (b * p.tq * p.h + h) * p.d;
            for (uint t0 = t_lo; t0 < t_end; t0 += BQ) {
                attn_stage<DP, 32 * SG>(tile, q + head, qstride, p.d, t0, BQ, t_end, tid);
                attn_stage<DP, 32 * SG>(tile + BQ * DP, dout + head, qstride, p.d, t0, BQ, t_end, tid);
                for (uint e = tid; e < BQ; e += 32 * SG) {
                    const uint t = t0 + e;
                    stats[e] = t < t_end ? lse[(b * p.h + h) * p.tq + t] : 0.0f;
                    stats[BQ + e] = t < t_end ? delta[(b * p.h + h) * p.tq + t] : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint n = 0; n < NQ; n++) {
                    simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    simdgroup_float8x8 dp = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (uint i = 0; i < ND; i++) {
                        simdgroup_float8x8 qt, gt;
                        simdgroup_load(qt, tile + n * 8 * DP + i * 8, DP, ulong2(0, 0), true);
                        simdgroup_load(gt, tile + BQ * DP + n * 8 * DP + i * 8, DP, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(sc, kr[i], qt, sc);
                        simdgroup_multiply_accumulate(dp, vr[i], gt, dp);
                    }
                    simdgroup_store(sc, s + n * 8, BQ);
                    simdgroup_store(dp, g + n * 8, BQ);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint c = part; c < BQ; c += 4) {
                    const uint t = t0 + c;
                    const bool vis = t < t_end && attn_visible(off + t, j_row, p);
                    const float pr = vis ? exp(s[row * BQ + c] * p.scale - stats[c]) : 0.0f;
                    s[row * BQ + c] = pr;
                    g[row * BQ + c] = pr * (g[row * BQ + c] - stats[BQ + c]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint n = 0; n < NQ; n++) {
                    simdgroup_float8x8 pm, dm;
                    simdgroup_load(pm, s + n * 8, BQ);
                    simdgroup_load(dm, g + n * 8, BQ);
                    for (uint i = 0; i < ND; i++) {
                        simdgroup_float8x8 qm, gm;
                        simdgroup_load(gm, tile + BQ * DP + n * 8 * DP + i * 8, DP);
                        simdgroup_load(qm, tile + n * 8 * DP + i * 8, DP);
                        simdgroup_multiply_accumulate(dva[i], pm, gm, dva[i]);
                        simdgroup_multiply_accumulate(dka[i], dm, qm, dka[i]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    threadgroup float* rows = tile + sg * 8 * DP;
    for (uint i = 0; i < ND; i++) simdgroup_store(dka[i], rows + i * 8, DP);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8 * DP; e += 32) {
        const uint r = e / DP, c = e % DP, j = j0 + sg * 8 + r;
        if (j < p.tk && c < p.d) dk[kvbase + j * kstride + c] = rows[e] * p.scale;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < ND; i++) simdgroup_store(dva[i], rows + i * 8, DP);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8 * DP; e += 32) {
        const uint r = e / DP, c = e % DP, j = j0 + sg * 8 + r;
        if (j < p.tk && c < p.d) dv[kvbase + j * kstride + c] = rows[e];
    }
}

// Simdgroups per threadgroup (8 rows each) and key/query block sizes per
// padded head dimension, within 32 KB of threadgroup memory.
#define ATTN_SG(DP) ((DP) <= 64 ? 8 : 4)
#define ATTN_BLOCK(DP) ((DP) <= 64 ? 32 : 16)
#define ATTN_DKV_BLOCK(DP) ((DP) <= 64 ? 16 : 16)
#define ATTN_MAX(A, B) ((A) > (B) ? (A) : (B))
#define ATTN_TILE(DP) ((DP) * ATTN_MAX(8 * ATTN_SG(DP), 2 * ATTN_BLOCK(DP)))
#define ATTN_DKV_TILE(DP) ((DP) * ATTN_MAX(8 * ATTN_SG(DP), 2 * ATTN_DKV_BLOCK(DP)))

#define ATTN_KERNELS(DP)                                                                                                    \
    kernel void attention_f32_d##DP(device float* out [[buffer(0)]], device const float* q [[buffer(1)]],                 \
                                    device const float* k [[buffer(2)]], device const float* v [[buffer(3)]],             \
                                    device float* lse [[buffer(4)]], constant AttnArgs& p [[buffer(5)]],                  \
                                    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup float tile[ATTN_TILE(DP)];                                                                              \
        threadgroup float scratch[ATTN_SG(DP) * 8 * ATTN_BLOCK(DP)];                                                        \
        threadgroup float diag[ATTN_SG(DP) * 64];                                                                           \
        attention_body<DP, ATTN_BLOCK(DP), ATTN_SG(DP)>(out, q, k, v, lse, p, tile, scratch, diag, tg, tid, sg, lane);      \
    }                                                                                                                       \
    kernel void attention_dq_f32_d##DP(device float* dq [[buffer(0)]], device const float* dout [[buffer(1)]],            \
                                       device const float* q [[buffer(2)]], device const float* k [[buffer(3)]],          \
                                       device const float* v [[buffer(4)]], device const float* lse [[buffer(5)]],        \
                                       device const float* delta [[buffer(6)]], constant AttnArgs& p [[buffer(7)]],       \
                                       uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup float tile[ATTN_TILE(DP)];                                                                              \
        threadgroup float scratch[ATTN_SG(DP) * 16 * ATTN_BLOCK(DP)];                                                       \
        attention_dq_body<DP, ATTN_BLOCK(DP), ATTN_SG(DP)>(dq, dout, q, k, v, lse, delta, p, tile, scratch, tg, tid, sg, lane); \
    }                                                                                                                       \
    kernel void attention_dkv_f32_d##DP(device float* dk [[buffer(0)]], device float* dv [[buffer(1)]],                   \
                                        device const float* dout [[buffer(2)]], device const float* q [[buffer(3)]],      \
                                        device const float* k [[buffer(4)]], device const float* v [[buffer(5)]],         \
                                        device const float* lse [[buffer(6)]], device const float* delta [[buffer(7)]],   \
                                        constant AttnArgs& p [[buffer(8)]],                                               \
                                        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup float tile[ATTN_DKV_TILE(DP)];                                                                          \
        threadgroup float scratch[ATTN_SG(DP) * 16 * ATTN_DKV_BLOCK(DP)];                                                   \
        threadgroup float stats[2 * ATTN_DKV_BLOCK(DP)];                                                                    \
        attention_dkv_body<DP, ATTN_DKV_BLOCK(DP), ATTN_SG(DP)>(dk, dv, dout, q, k, v, lse, delta, p, tile, scratch, stats, tg, tid, sg, lane); \
    }

ATTN_KERNELS(8)
ATTN_KERNELS(16)
ATTN_KERNELS(32)
ATTN_KERNELS(64)
ATTN_KERNELS(128)
