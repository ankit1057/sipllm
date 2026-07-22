// ops.h — Task 2 + Phase 1: the math engine.
//
// Pure numeric kernels over fp32 buffers. No transformer concepts here — just
// linear algebra and the handful of nonlinearities a decoder needs. Everything
// is expressed against raw pointers so the same kernels serve owning Tensors,
// mmap'd views, and dequant scratch buffers alike.
//
// Weight-matrix convention (used everywhere): W is [n_out, n_in] row-major,
// i.e. one contiguous row per output feature. This matches PyTorch nn.Linear
// and GGUF tensor layout, and makes each output a single dot product over a
// contiguous row — ideal for streaming and for block-quantized formats.
#pragma once

#include "llm/tensor.h"
#include "llm/threadpool.h"

namespace llm {

// ---- vector ops -----------------------------------------------------------

// out[i] = a[i] + b[i]
void vec_add(float* out, const float* a, const float* b, int64_t n);

// in-place residual: x[i] += y[i]
void vec_add_inplace(float* x, const float* y, int64_t n);

// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// (Llama-style: no mean subtraction, per-channel learned scale.)
void rmsnorm(float* out, const float* x, const float* weight, int64_t n,
             float eps = 1e-5f);

// Numerically-stable softmax over n elements, in place.
void softmax(float* x, int64_t n);

// SiLU / swish: x * sigmoid(x), in place.
void silu_inplace(float* x, int64_t n);

// ---- matmul ---------------------------------------------------------------

// Single-vector (batch=1) linear: y = W @ x
//   W: [n_out, n_in] row-major, x: [n_in], y: [n_out]
// Row-parallel across the thread pool. This is the workhorse for token-by-token
// decoding where the "batch" is a single position.
void matmul(float* y, const float* W, const float* x,
            int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// Batched linear: Y = X @ W^T
//   X: [m, n_in] row-major, W: [n_out, n_in] row-major, Y: [m, n_out]
// Used for prompt prefill where m tokens are processed together.
void matmul_batch(float* Y, const float* X, const float* W,
                  int64_t m, int64_t n_out, int64_t n_in,
                  ThreadPool* pool = nullptr);

} // namespace llm
