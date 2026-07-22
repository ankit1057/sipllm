// simd.h — Task 10: ARM NEON kernels with scalar fallback.
//
// The single most-executed operation in the whole engine is a dot product of
// two fp32 vectors (one row of a weight matrix · the activation vector). Every
// matmul is thousands of these. We give it a hand-vectorized NEON path using
// four independent accumulators to hide FMA latency, plus fused
// multiply-add helpers used by RMSNorm / residual adds / SwiGLU.
//
// On this device `nproc` reports asimd + asimddp + i8mm + bf16 + sve2, so the
// NEON path is always taken; the scalar version exists for portability and as
// the correctness oracle the tests diff against.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LLM_HAVE_NEON 1
#else
#define LLM_HAVE_NEON 0
#endif

namespace llm {

// out = sum_i a[i]*b[i]
inline float dot_f32(const float* a, const float* b, int64_t n) {
#if LLM_HAVE_NEON
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f), acc3 = vdupq_n_f32(0.f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i +  0), vld1q_f32(b + i +  0));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.f;
    for (int64_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

// y += alpha * x
inline void axpy_f32(float* y, const float* x, float alpha, int64_t n) {
#if LLM_HAVE_NEON
    float32x4_t va = vdupq_n_f32(alpha);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    for (; i < n; ++i) y[i] += alpha * x[i];
#else
    for (int64_t i = 0; i < n; ++i) y[i] += alpha * x[i];
#endif
}

// out[i] = a[i] * b[i]
inline void mul_f32(float* out, const float* a, const float* b, int64_t n) {
#if LLM_HAVE_NEON
    int64_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; ++i) out[i] = a[i] * b[i];
#else
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

// sum of squares — used by RMSNorm.
inline float sumsq_f32(const float* x, int64_t n) {
    return dot_f32(x, x, n);
}

// scale in place: x[i] *= s
inline void scale_f32(float* x, float s, int64_t n) {
#if LLM_HAVE_NEON
    float32x4_t vs = vdupq_n_f32(s);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vs));
    for (; i < n; ++i) x[i] *= s;
#else
    for (int64_t i = 0; i < n; ++i) x[i] *= s;
#endif
}

} // namespace llm
