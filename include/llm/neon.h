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

// True if THIS CPU implements SDOT (asimddp). Detected at runtime (getauxval /
// sysctl), not at compile time, so a baseline-ISA binary still uses the SDOT
// kernels on cores that have them. SIPLLM_CPU=generic forces false.
bool neon_dotprod_available();

// True if an int8 kernel exists for this CPU: SDOT on ARM (runtime), or a
// binary built with AVX2 on x86.
bool int8_kernels_available();

// Fast-quant kernel toggle (--fast). fast_quant_enabled() is true only when it
// was requested AND int8_kernels_available(); otherwise linear()/linear_batch()
// take the exact fp32-dequant path for the whole batch. Off by default so the
// exact path stays the correctness oracle; the int8 paths quantize the
// activation (like llama.cpp).
void set_fast_quant(bool on);
bool fast_quant_enabled();

// y = W @ x, W is Q8_0 [n_out, n_in]; activation quantized to int8 internally.
// n_in must be a multiple of 32. Falls back to the fp32-dequant path when SDOT
// is unavailable, so it is always safe to call.
void matmul_q8_0_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// y = W @ x, W is Q4_K [n_out, n_in] (256-weight super-blocks); activation
// quantized to int8 per 32-block internally, Σxq per block shared across rows.
// n_in must be a multiple of 256. Same opt-in gate and the same fp32-dequant
// fallback as matmul_q8_0_i8, so it is always safe to call.
void matmul_q4_K_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// y = W @ x, W is Q6_K [n_out, n_in] (256-weight super-blocks, per-16 scales).
// Same gate, activation quantization and fallback as matmul_q4_K_i8.
void matmul_q6_K_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// Batched (prefill) forms: Y[bs][n_out] = X[bs][n_in] . W^T. On ARM with SDOT
// the activations are quantized once for the batch and each weight row is
// unpacked once per small tile of tokens; per token the result is bit-identical
// to the single-vector kernel. Elsewhere they call the single-vector kernel per
// token (today's behavior).
void matmul_q8_0_i8_batch(float* Y, const void* W, const float* X, int64_t bs,
                          int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);
void matmul_q4_K_i8_batch(float* Y, const void* W, const float* X, int64_t bs,
                          int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);
void matmul_q6_K_i8_batch(float* Y, const void* W, const float* X, int64_t bs,
                          int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// Bulk fp16 -> fp32 using the hardware half-precision converter (asimdhp).
void fp16_to_fp32_bulk(const uint16_t* src, float* dst, int64_t n);

} // namespace llm
