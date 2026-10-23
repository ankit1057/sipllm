// linear.h — the single call the transformer uses for every projection.
//
// Dispatches on the resident weight's dtype: fp32 weights go to the plain
// matmul, block-quantized weights to the fused dequant-matmul. The transformer
// never branches on quantization — it just calls linear().
#pragma once

#include "llm/model.h"
#include "llm/ops.h"
#include "llm/quant.h"
#include "llm/neon.h"
#include "llm/kquant.h"

namespace llm {

// Opt-in (--fast) integer K-quant path: only where the AVX2 kernels exist, so
// ARM keeps the int8 SDOT kernels below and the default (fast off) is unchanged.
inline bool use_kquant_fast(const WeightRef& W) {
    return fast_quant_enabled() && kquant_simd_available() &&
           kquant_supported(W.dtype, W.n_in);
}

inline void linear(float* y, const WeightRef& W, const float* x,
                   ThreadPool* pool = nullptr) {
    if (W.dtype == DType::F32)
        matmul(y, static_cast<const float*>(W.data), x, W.n_out, W.n_in, pool);
    else if (W.dtype == DType::Q8_0 && fast_quant_enabled() && (W.n_in % 32 == 0))
        matmul_q8_0_i8(y, W.data, x, W.n_out, W.n_in, pool);   // int8 SDOT (opt-in --fast)
    else if (use_kquant_fast(W))   // x86 AVX2+FMA only; ARM keeps the SDOT kernels below
        matmul_kq_q8K(y, W.data, W.dtype, x, W.n_out, W.n_in, pool);  // AVX2 Q*_K x Q8_K (opt-in --fast)
    else if (W.dtype == DType::Q4_K && fast_quant_enabled() && (W.n_in % 256 == 0))
        matmul_q4_K_i8(y, W.data, x, W.n_out, W.n_in, pool);   // int8 SDOT on 4-bit (opt-in --fast)
    else if (W.dtype == DType::Q6_K && fast_quant_enabled() && (W.n_in % 256 == 0))
        matmul_q6_K_i8(y, W.data, x, W.n_out, W.n_in, pool);   // int8 SDOT on 6-bit (opt-in --fast)
    else
        matmul_quant(y, W.data, W.dtype, x, W.n_out, W.n_in, pool);
}

inline void linear_batch(float* Y, const WeightRef& W, const float* X, int64_t bs,
                         ThreadPool* pool = nullptr) {
    if (bs == 1) {
        linear(Y, W, X, pool);
        return;
    }
    if (W.dtype == DType::F32)
        matmul_batch(Y, X, static_cast<const float*>(W.data), bs, W.n_out, W.n_in, pool);
    else if (W.dtype == DType::Q8_0 && fast_quant_enabled() && (W.n_in % 32 == 0)) {
        for (int64_t b = 0; b < bs; ++b) {
            matmul_q8_0_i8(Y + b * W.n_out, W.data, X + b * W.n_in, W.n_out, W.n_in, pool);
        }
    } else if (use_kquant_fast(W)) {
        matmul_kq_q8K_batch(Y, W.data, W.dtype, X, bs, W.n_out, W.n_in, pool);
    } else if (W.dtype == DType::Q4_K && fast_quant_enabled() && (W.n_in % 256 == 0)) {
        for (int64_t b = 0; b < bs; ++b) {
            matmul_q4_K_i8(Y + b * W.n_out, W.data, X + b * W.n_in, W.n_out, W.n_in, pool);
        }
    } else if (W.dtype == DType::Q6_K && fast_quant_enabled() && (W.n_in % 256 == 0)) {
        for (int64_t b = 0; b < bs; ++b) {
            matmul_q6_K_i8(Y + b * W.n_out, W.data, X + b * W.n_in, W.n_out, W.n_in, pool);
        }
    } else {
        matmul_quant_batch(Y, W.data, W.dtype, X, bs, W.n_out, W.n_in, pool);
    }
}

} // namespace llm
