// Reductions of MetalBackend (src/metal/backend.zig), compiled without fast
// math: sums are carried as double-floats (hi + lo, TwoSum / TwoProduct) so
// they match the CPU backend's f64 accumulation, cancellations included.
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
