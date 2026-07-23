// ops.cpp — implementations of the math engine (see ops.h).
#include "llm/ops.h"
#include "llm/simd.h"

#include <cmath>

namespace llm {

void vec_add(float* out, const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void vec_add_inplace(float* x, const float* y, int64_t n) {
    axpy_f32(x, y, 1.0f, n);
}

void rmsnorm(float* out, const float* x, const float* weight, int64_t n,
             float eps) {
    // scale = 1 / sqrt( (1/n) * sum(x^2) + eps )
    float ss = sumsq_f32(x, n);
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] * scale * weight[i];
}

void softmax(float* x, int64_t n) {
    if (n <= 0) return;
    float maxv = x[0];
    for (int64_t i = 1; i < n; ++i) maxv = x[i] > maxv ? x[i] : maxv;
    float sum = 0.f;
    for (int64_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - maxv); sum += x[i]; }
    float inv = sum > 0.f ? 1.0f / sum : 0.f;
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

void silu_inplace(float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

void gelu_inplace(float* x, int64_t n) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;   // sqrt(2/pi)
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        float inner = kSqrt2OverPi * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

void softcap_inplace(float* x, int64_t n, float cap) {
    if (cap <= 0.f) return;
    const float inv = 1.0f / cap;
    for (int64_t i = 0; i < n; ++i) x[i] = cap * std::tanh(x[i] * inv);
}

void rmsnorm_gemma(float* out, const float* x, const float* weight, int64_t n,
                   float eps) {
    float ss = sumsq_f32(x, n);
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] * scale * (1.0f + weight[i]);
}

void matmul(float* y, const float* W, const float* x,
            int64_t n_out, int64_t n_in, ThreadPool* pool) {
    auto body = [&](int, int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o)
            y[o] = dot_f32(W + o * n_in, x, n_in);
    };
    if (pool && pool->size() > 1 && n_out >= 64)
        pool->parallel_for(n_out, body);
    else
        body(0, 0, n_out);
}

void matmul_batch(float* Y, const float* X, const float* W,
                  int64_t m, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    // Parallelize over output features; each worker handles all m rows for its
    // slice of outputs. Keeps W rows hot in cache across the m tokens.
    auto body = [&](int, int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) {
            const float* wrow = W + o * n_in;
            for (int64_t r = 0; r < m; ++r)
                Y[r * n_out + o] = dot_f32(wrow, X + r * n_in, n_in);
        }
    };
    if (pool && pool->size() > 1 && n_out >= 64)
        pool->parallel_for(n_out, body);
    else
        body(0, 0, n_out);
}

} // namespace llm
