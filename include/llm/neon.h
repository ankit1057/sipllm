// neon.h — Task 10: ARM SIMD kernels beyond the fp32 dot in simd.h.
//
// The mobile-inference win is doing the matmul in int8 with the SDOT
// instruction (asimddp) instead of dequantizing to fp32 first. We quantize the
// activation vector to int8 once per token, then each Q8_0 weight block becomes
// a 32-wide int8·int8 dot (two vdotq_s32) scaled by d_weight·d_activation.
//
// This is an *approximation* of the exact fp32 path (the activation is
// quantized), matching what llama.cpp does; the exact fp32-dequant path in
// quant.cpp remains the default/oracle. Availability is reported at runtime so
// callers/benchmarks can pick a path.
#pragma once

#include "llm/threadpool.h"

#include <cstdint>

namespace llm {

// True if built with SDOT (dot-product) support.
bool neon_dotprod_available();

// y = W @ x, W is Q8_0 [n_out, n_in]; activation quantized to int8 internally.
// n_in must be a multiple of 32. Falls back to the fp32-dequant path when SDOT
// is unavailable, so it is always safe to call.
void matmul_q8_0_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// Bulk fp16 -> fp32 using the hardware half-precision converter (asimdhp).
void fp16_to_fp32_bulk(const uint16_t* src, float* dst, int64_t n);

} // namespace llm
