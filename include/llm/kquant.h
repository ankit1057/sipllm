// kquant.h — integer-domain K-quant matmul (Q4_K / Q5_K / Q6_K x Q8_K).
//
// The default quantized matmul (quant.cpp: matmul_quant) dequantizes every
// weight row to fp32 and does an fp32 dot. That is exact but slow. This module
// does what llama.cpp does instead: quantize the activation vector ONCE to
// ggml's Q8_K format (256-element blocks, fp32 scale, int8 codes, int16 sums of
// every 16 codes) and dot the packed K-quant weight blocks against it in int8
// arithmetic (AVX2 maddubs/madd on x86-64). The weight never leaves its packed
// form, so the kernel is bound by weight bandwidth, not dequant ALU.
//
// Numerics: the activation is quantized to int8 per 256 elements, so results
// differ from the fp32-dequant oracle by the activation-quantization error
// (~1e-3..1e-2 relative), exactly like llama.cpp. That is why linear() only
// routes here when the opt-in --fast flag (fast_quant_enabled()) is set.
//
// Portability: the AVX2+FMA kernels are compiled only under
// `__AVX2__ && __FMA__`; every entry point also has a scalar reference (ported
// from ggml's *_generic functions) so the file compiles and works everywhere.
// kquant_simd_available() reports whether the fast kernels are live — callers
// should only *prefer* this path when it returns true (on ARM the existing
// NEON dequant path is faster than the scalar reference here).
//
// Block layouts are GGUF-standard and identical to the ones decoded by
// dequant_q4_K / dequant_q5_K / dequant_q6_K in quant.cpp.
#pragma once

#include "llm/dtype.h"
#include "llm/threadpool.h"

#include <cstdint>

namespace llm {

constexpr int kQK_K = 256;

struct block_q4_K {           // 144 bytes / 256 weights (4.5 bpw)
    uint16_t d;               // fp16 super-block scale for scales
    uint16_t dmin;            // fp16 super-block scale for mins
    uint8_t  scales[12];      // 8 x (6-bit scale, 6-bit min), packed
    uint8_t  qs[128];         // 4-bit quants
};
struct block_q5_K {           // 176 bytes / 256 weights (5.5 bpw)
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];          // high (5th) bit
    uint8_t  qs[128];         // low 4 bits
};
struct block_q6_K {           // 210 bytes / 256 weights (6.5625 bpw)
    uint8_t  ql[128];         // low 4 bits
    uint8_t  qh[64];          // high 2 bits
    int8_t   scales[16];      // 8-bit signed sub-block scales
    uint16_t d;               // fp16 super-block scale
};
struct block_q8_K {           // 292 bytes / 256 activations
    float    d;               // scale (fp32, NOT fp16)
    int8_t   qs[256];         // int8 codes
    int16_t  bsums[16];       // sum of each group of 16 codes
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K layout");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K layout");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K layout");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K layout");

// True when the AVX2+FMA kernels were compiled in.
bool kquant_simd_available();

// True if matmul_kq_q8K handles weight type `t` with this inner dimension.
inline bool kquant_supported(DType t, int64_t n_in) {
    return (t == DType::Q4_K || t == DType::Q5_K || t == DType::Q6_K) &&
           n_in > 0 && (n_in % kQK_K) == 0;
}

// Exactly ggml's quantize_row_q8_K_ref: per 256 elements, iscale = -127/max
// (max = the signed value of largest magnitude), round-half-even codes,
// d = 1/iscale, bsums over each 16. n must be a multiple of 256.
void quantize_row_q8_K(const float* x, block_q8_K* y, int64_t n);

// Dot of one K-quant weight row (n elements, n % 256 == 0) with a Q8_K row.
// Dispatches to AVX2 when available, else the scalar reference.
float vec_dot_q4_K_q8_K(int64_t n, const void* w, const block_q8_K* y);
float vec_dot_q5_K_q8_K(int64_t n, const void* w, const block_q8_K* y);
float vec_dot_q6_K_q8_K(int64_t n, const void* w, const block_q8_K* y);

// Scalar references (ports of ggml's *_generic) — always compiled; used as the
// fallback and as the tight test oracle for the SIMD kernels.
float vec_dot_q4_K_q8_K_ref(int64_t n, const void* w, const block_q8_K* y);
float vec_dot_q5_K_q8_K_ref(int64_t n, const void* w, const block_q8_K* y);
float vec_dot_q6_K_q8_K_ref(int64_t n, const void* w, const block_q8_K* y);

// y = W @ x. W: [n_out, n_in] of type t (Q4_K/Q5_K/Q6_K), n_in % 256 == 0.
// x is quantized to Q8_K once, then output rows are split across the pool.
void matmul_kq_q8K(float* y, const void* W, DType t, const float* x,
                   int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// Batched (prefill): Y = X @ W^T, X: [m, n_in], Y: [m, n_out] (same layout as
// matmul_quant_batch). All m activation rows are quantized once; each weight
// row is then dotted against a cache-sized tile of activation rows, 4 at a time
// with the weight block unpacked once per 4 rows. Every Y element is
// bit-identical to what matmul_kq_q8K produces for that row alone.
void matmul_kq_q8K_batch(float* Y, const void* W, DType t, const float* X,
                         int64_t m, int64_t n_out, int64_t n_in,
                         ThreadPool* pool = nullptr);

} // namespace llm
