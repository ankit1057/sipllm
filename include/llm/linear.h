// linear.h — the single call the transformer uses for every projection.
//
// Dispatches on the resident weight's dtype: fp32 weights go to the plain
// matmul, block-quantized weights to the fused dequant-matmul. The transformer
// never branches on quantization — it just calls linear().
#pragma once

#include "llm/model.h"
#include "llm/ops.h"
#include "llm/quant.h"

namespace llm {

inline void linear(float* y, const WeightRef& W, const float* x,
                   ThreadPool* pool = nullptr) {
    if (W.dtype == DType::F32)
        matmul(y, static_cast<const float*>(W.data), x, W.n_out, W.n_in, pool);
    else
        matmul_quant(y, W.data, W.dtype, x, W.n_out, W.n_in, pool);
}

} // namespace llm
