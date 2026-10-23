// neon_internal.h — helpers shared by neon.cpp (baseline ISA) and
// neon_sdot.cpp (compiled with +dotprod). PRIVATE to those two files.
//
// Everything here has internal linkage (anonymous namespace) ON PURPOSE: the
// two files are compiled for different ISAs, and an inline function with
// external linkage would be merged by the linker into ONE copy — possibly the
// +dotprod one, which a baseline caller on an ARMv8.0 core would then execute
// (SIGILL). Internal linkage gives each file its own copy for its own ISA.
#pragma once

#include "llm/quant.h"
#include "llm/threadpool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace llm {

namespace {

// Quantize an fp32 activation vector to per-32-block int8 (symmetric), matching
// the Q8 activation format. Returns int8 codes + per-block scale.
[[maybe_unused]] void quantize_activation_q8(const float* x, int64_t n, int8_t* q, float* scale) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
        float d = amax / 127.f;
        float id = d > 0.f ? 1.f / d : 0.f;
        scale[b] = d;
        for (int i = 0; i < 32; ++i) {
            int v = (int)std::lround(xb[i] * id);
            q[b * 32 + i] = (int8_t)std::max(-127, std::min(127, v));
        }
    }
}

// Hardware fp16 -> fp32 for a Q8_0 block scale (one fcvt vs the software IEEE
// decode). Falls back to the software path where fp16 arithmetic is absent.
#if defined(__ARM_FP16_FORMAT_IEEE)
[[maybe_unused]] inline float f16_scale(const uint8_t* p) { __fp16 h; std::memcpy(&h, p, 2); return (float)h; }
#else
[[maybe_unused]] inline float f16_scale(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return fp16_to_fp32(h); }
#endif

} // namespace

// True when this core implements SDOT/UDOT (runtime check, cached; neon.cpp).
bool cpu_has_dotprod();

// Wave 15: Q4_K weights interleaved 4 rows at a time ("IL4"). One block holds
// the same bytes as four GGUF Q4_K blocks (one super-block of 256 weights for
// each of 4 consecutive rows), rearranged so a 16-byte vector carries 4
// consecutive quant bytes of each row (vdotq_laneq: one lane per row) and the
// 12 packed scale bytes are interleaved by byte across the 4 rows (all 32
// (sc, m) pairs decode with 3 vector loads). Same size as GGUF, so a resident
// layer can be repacked in place.
struct BlockQ4KIL4 {
    uint16_t d[4], dmin[4];   // fp16, per row
    uint8_t  scales[12][4];   // byte i of the packed 6-bit scales, rows 0..3
    uint8_t  qs[4][8][16];    // [64-wide chunk][vector][row*4 + byte]
};
static_assert(sizeof(BlockQ4KIL4) == 4 * 144, "IL4 must be exactly the size of 4 GGUF Q4_K blocks");

namespace detail {
// False when neon_sdot.cpp was compiled without dotprod (then the SDOT entry
// points are exact-path stubs and cpu_has_dotprod() reports false).
bool sdot_kernels_compiled();
// ARM SDOT kernels (src/neon_sdot.cpp). Only call when cpu_has_dotprod().
void matmul_q8_0_sdot(float* y, const void* W, const float* x,
                      int64_t n_out, int64_t n_in, ThreadPool* pool);
void matmul_q4_K_sdot(float* y, const void* W, const float* x,
                      int64_t n_out, int64_t n_in, ThreadPool* pool);
void matmul_q6_K_sdot(float* y, const void* W, const float* x,
                      int64_t n_out, int64_t n_in, ThreadPool* pool);
// Batched (prefill): Y[bs][n_out] = X[bs][n_in] . W^T, bit-identical per token
// to the single-vector kernels above.
void matmul_q8_0_sdot_batch(float* Y, const void* W, const float* X, int64_t bs,
                            int64_t n_out, int64_t n_in, ThreadPool* pool);
void matmul_q4_K_sdot_batch(float* Y, const void* W, const float* X, int64_t bs,
                            int64_t n_out, int64_t n_in, ThreadPool* pool);
void matmul_q6_K_sdot_batch(float* Y, const void* W, const float* X, int64_t bs,
                            int64_t n_out, int64_t n_in, ThreadPool* pool);
// Y[bs][n_out] from IL4-repacked Q4_K weights (n_out % 4 == 0); any bs >= 1.
void matmul_q4_K_il4_sdot(float* Y, const void* Wil4, const float* X, int64_t bs,
                          int64_t n_out, int64_t n_in, ThreadPool* pool);
} // namespace detail

} // namespace llm
