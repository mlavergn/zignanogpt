// Reductions and optimizer kernels of MetalBackend (src/metal/backend.zig),
// compiled without fast math: sums are carried as double-floats (hi + lo,
// TwoSum / TwoProduct) so they match the CPU backend's f64 accumulation,
// cancellations included, and the elementwise updates round as the CPU's do.
#include <metal_stdlib>
using namespace metal;

// Error-free a + b.
static float2 two_sum(float a, float b) {
    const float s = a + b;
    const float bp = s - a;
    return float2(s, (a - (s - bp)) + (b - bp));
}

// (hi, lo) + (hi, lo), renormalized.
static float2 add_df(float2 x, float2 y) {
    const float2 s = two_sum(x.x, y.x);
    const float lo = s.y + x.y + y.y;
    return two_sum(s.x, lo);
}

// Sums a double-float over the threadgroup (at most 1024 threads).
static float2 block_sum_df(float2 v, threadgroup float2* shared, uint lane, uint sg, uint groups) {
    for (ushort offset = 16; offset > 0; offset /= 2) {
        const float2 other = float2(simd_shuffle_down(v.x, offset), simd_shuffle_down(v.y, offset));
        v = add_df(v, other);
    }
    if (lane == 0) shared[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 total = float2(0.0f);
    for (uint g = 0; g < groups; g++) total = add_df(total, shared[g]);
    return total;
}

// partial[group] = sum over the group's slice of a[i] * b[i], as (hi, lo).
kernel void dot_partial_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float2* partial [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[32];
    const uint per = (n + groups - 1) / groups;
    const uint start = group * per;
    const uint end = min(start + per, n);
    float2 sum = float2(0.0f);
    for (uint i = start + tid; i < end; i += threads) {
        const float p = a[i] * b[i];
        const float e = fma(a[i], b[i], -p);
        sum = add_df(sum, float2(p, e));
    }
    const float2 total = block_sum_df(sum, shared, lane, sg, (threads + 31) / 32);
    if (tid == 0) partial[group] = total;
}

struct ReduceArgs {
    uint count;
    float factor;
    uint accumulate;
    uint pairs; // values are (hi, lo) pairs
};

// out[0] (+)= factor * sum(values), one threadgroup, fixed order.
kernel void reduce_sum_f32(device const float* values [[buffer(0)]], device float* out [[buffer(1)]], constant ReduceArgs& p [[buffer(2)]],
                           uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[32];
    float2 sum = float2(0.0f);
    for (uint i = tid; i < p.count; i += threads) {
        const float2 v = p.pairs ? float2(values[2 * i], values[2 * i + 1]) : float2(values[i], 0.0f);
        sum = add_df(sum, v);
    }
    const float2 total = block_sum_df(sum, shared, lane, sg, (threads + 31) / 32);
    if (tid == 0) {
        const float value = (total.x + total.y) * p.factor;
        out[0] = p.accumulate ? out[0] + value : value;
    }
}

// ---------------------------------------------------------------- double-float helpers

// sqrt(hi + lo), rounded to a float.
static float sqrt_df(float2 x) {
    const float s = sqrt(x.x);
    if (s == 0.0f) return 0.0f;
    const float e = fma(-s, s, x.x) + x.y;
    return s + e / (2.0f * s);
}

// (hi + lo) * y as a double-float.
static float2 mul_df_f(float2 x, float y) {
    const float p = x.x * y;
    const float e = fma(x.x, y, -p) + x.y * y;
    return two_sum(p, e);
}

// (hi + lo) / y, rounded to a float.
static float div_df_f(float2 x, float y) {
    const float q = x.x / y;
    const float r = fma(-q, y, x.x) + x.y;
    return q + r / y;
}

// torch's two-sided lerp (CpuBackend.lerp).
static float lerp_t(float a, float b, float w) {
    return fabs(w) < 0.5f ? a + w * (b - a) : b - (b - a) * (1.0f - w);
}

// out[r] = sum over row r of x^2, as (hi, lo); one threadgroup per row.
kernel void row_squares_f32(device const float* x [[buffer(0)]], device float2* out [[buffer(1)]], constant uint& cols [[buffer(2)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[32];
    device const float* row = x + r * cols;
    float2 sum = float2(0.0f);
    for (uint c = tid; c < cols; c += threads) {
        const float v = row[c];
        const float sq = v * v;
        sum = add_df(sum, float2(sq, fma(v, v, -sq)));
    }
    const float2 total = block_sum_df(sum, shared, lane, sg, (threads + 31) / 32);
    if (tid == 0) out[r] = total;
}

struct ColArgs {
    uint rows;
    uint cols;
};

// out[c] = sum over column c of x^2, as (hi, lo); one thread per column, rows in order.
kernel void col_squares_f32(device const float* x [[buffer(0)]], device float2* out [[buffer(1)]], constant ColArgs& a [[buffer(2)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= a.cols) return;
    float2 sum = float2(0.0f);
    for (uint r = 0; r < a.rows; r++) {
        const float v = x[r * a.cols + c];
        const float sq = v * v;
        sum = add_df(sum, float2(sq, fma(v, v, -sq)));
    }
    out[c] = sum;
}

// out[0] = sum of count (hi, lo) pairs, kept as a pair; one threadgroup.
kernel void pair_total_f32(device const float2* values [[buffer(0)]], device float2* out [[buffer(1)]], constant uint& count [[buffer(2)]],
                           uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[32];
    float2 sum = float2(0.0f);
    for (uint i = tid; i < count; i += threads) sum = add_df(sum, values[i]);
    const float2 total = block_sum_df(sum, shared, lane, sg, (threads + 31) / 32);
    if (tid == 0) out[0] = total;
}

// ---------------------------------------------------------------- optimizer

struct AdamArgs {
    float decay;
    float w1;
    float w2;
    float bias2;
    float step_size;
    float eps;
    uint n;
};

// CpuBackend.adamwStep, elementwise.
kernel void adamw_step_f32(device float* p [[buffer(0)]], device const float* g [[buffer(1)]], device float* m [[buffer(2)]],
                           device float* v [[buffer(3)]], constant AdamArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    float pv = p[i] * a.decay;
    const float gv = g[i];
    const float mv = lerp_t(m[i], gv, a.w1);
    const float vv = lerp_t(v[i], gv * gv, a.w2);
    const float denom = sqrt(vv / a.bias2) + a.eps;
    pv += -a.step_size * (mv / denom);
    p[i] = pv;
    m[i] = mv;
    v[i] = vv;
}

struct MomentumArgs {
    float momentum;
    uint n;
};

// CpuBackend.muonMomentum, elementwise.
kernel void muon_momentum_f32(device float* g [[buffer(0)]], device float* buf [[buffer(1)]], constant MomentumArgs& a [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float b = lerp_t(buf[i], g[i], 1.0f - a.momentum);
    buf[i] = b;
    g[i] = lerp_t(g[i], b, a.momentum);
}

struct ShapeArgs {
    uint rows;
    uint cols;
    uint n;
};

// MuonEq: every row to the mean row norm sqrt(total) / sqrt(rows).
kernel void muon_scale_rows_f32(device float* x [[buffer(0)]], device const float2* row_sq [[buffer(1)]], device const float2* total [[buffer(2)]],
                                constant ShapeArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float target = sqrt_df(total[0]) / sqrt(float(a.rows));
    const float row_norm = max(sqrt_df(row_sq[i / a.cols]), 1e-6f);
    x[i] *= target / row_norm;
}

// x /= ||x||_F * 1.01 + 1e-6.
kernel void muon_scale_all_f32(device float* x [[buffer(0)]], device const float2* total [[buffer(1)]], constant uint& n [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    x[i] /= sqrt_df(total[0]) * 1.01f + 1e-6f;
}

struct RenormArgs {
    float target;
    uint n;
};

// Muon+: g *= target / max(||g||_F, 1e-6).
kernel void muon_renorm_f32(device float* g [[buffer(0)]], device const float2* total [[buffer(1)]], constant RenormArgs& a [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    g[i] *= a.target / max(sqrt_df(total[0]), 1e-6f);
}

struct StatsArgs {
    uint count;
    float red;
    float beta2;
};

// NorMuon's second-moment EMA (in place) and the norm-preserving ratio; one threadgroup.
kernel void normuon_stats_f32(device const float2* sums [[buffer(0)]], device float* second [[buffer(1)]], device float* ratio [[buffer(2)]],
                              constant StatsArgs& a [[buffer(3)]],
                              uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared[32];
    float2 norm_sq = float2(0.0f);
    float2 new_sq = float2(0.0f);
    for (uint i = tid; i < a.count; i += threads) {
        const float mean = div_df_f(sums[i], a.red);
        norm_sq = add_df(norm_sq, float2(mean, 0.0f));
        const float s = lerp_t(second[i], mean, 1.0f - a.beta2);
        second[i] = s;
        const float step = 1.0f / sqrt(max(s, 1e-10f));
        new_sq = add_df(new_sq, float2(mean * a.red * step * step, 0.0f));
    }
    const uint groups = (threads + 31) / 32;
    const float2 v_norm_sq = block_sum_df(norm_sq, shared, lane, sg, groups);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float2 v_new_sq = block_sum_df(new_sq, shared, lane, sg, groups);
    if (tid == 0) {
        const float v_norm = sqrt_df(mul_df_f(v_norm_sq, a.red));
        ratio[0] = v_norm / max(sqrt_df(v_new_sq), 1e-10f);
    }
}

struct UpdateArgs {
    float lr;
    float lr_wd;
    uint cols;
    uint by_row;
    uint n;
};

// NorMuon scaling, then the cautious update p -= lr * g + lr * wd * p * [g * p >= 0].
kernel void muon_update_f32(device float* p [[buffer(0)]], device float* g [[buffer(1)]], device const float* second [[buffer(2)]],
                            device const float* ratio [[buffer(3)]], constant UpdateArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const uint idx = a.by_row ? i / a.cols : i % a.cols;
    const float gv = g[i] * ((1.0f / sqrt(max(second[idx], 1e-10f))) * ratio[0]);
    g[i] = gv;
    const float pv = p[i];
    const float mask = gv * pv >= 0.0f ? 1.0f : 0.0f;
    p[i] = pv - (a.lr * gv + a.lr_wd * pv * mask);
}

// ---------------------------------------------------------------- embedding backward

struct EmbedBackArgs {
    uint cols;
    uint unique;
    uint rows;
};

// dtable[id] += dout[r] for every row r with that id, rows in order (the CPU's
// sum order). `index` holds the rows grouped by id, then each id's start (and
// the end), then the ids. One thread per (column, id).
kernel void embedding_backward_f32(device float* dtable [[buffer(0)]], device const float* dout [[buffer(1)]], device const uint* index [[buffer(2)]],
                                   constant EmbedBackArgs& a [[buffer(3)]], uint2 pos [[thread_position_in_grid]]) {
    const uint c = pos.x;
    const uint u = pos.y;
    if (c >= a.cols || u >= a.unique) return;
    device const uint* order = index;
    device const uint* starts = index + a.rows;
    device const uint* ids = starts + a.unique + 1;
    device float* cell = dtable + ids[u] * a.cols + c;
    float acc = *cell;
    for (uint k = starts[u]; k < starts[u + 1]; k++) acc += dout[order[k] * a.cols + c];
    *cell = acc;
}
