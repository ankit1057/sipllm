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
} // namespace detail

} // namespace llm
