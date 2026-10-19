// neon_sdot.cpp — the ARM int8 SDOT kernels, compiled with +dotprod (Issue #68).
//
// The rest of the engine is built for the baseline ISA so one binary runs on
// every 64-bit ARM core, including ARMv8.0 parts without SDOT (Cortex-A53).
// Only this file is built with dotprod enabled (Makefile / CMake per-file
// flag); neon.cpp calls into it only after cpu_has_dotprod() said yes at
// runtime. Bodies moved verbatim from neon.cpp's former
// `#if defined(__ARM_FEATURE_DOTPROD)` branches.
#include "neon_internal.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <vector>

namespace llm {
namespace detail {

bool sdot_kernels_compiled() { return true; }

void matmul_q8_0_sdot(float* y, const void* W, const float* x,
                      int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nb = n_in / 32;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nb * 34;   // Q8_0 block = fp16 d + 32 int8

    // Quantize the activation once; reused across all output rows.
    std::vector<int8_t> xq(n_in);
    std::vector<float>  xs(nb);
    quantize_activation_q8(x, n_in, xq.data(), xs.data());

    auto body = [&](int, int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            // Vector float accumulator hides FMA latency; each block's int dot
            // stays in an int32x4 (no per-block horizontal reduce) and the
            // weight scale uses the hardware fp16 converter.
            float32x4_t accv = vdupq_n_f32(0.f);
            for (int64_t b = 0; b < nb; ++b) {
                const int8_t* wq = reinterpret_cast<const int8_t*>(row + b * 34 + 2);
                const int8_t* aq = xq.data() + b * 32;
                int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                                        vld1q_s8(wq),      vld1q_s8(aq)),
                                        vld1q_s8(wq + 16), vld1q_s8(aq + 16));
                accv = vfmaq_n_f32(accv, vcvtq_f32_s32(p), f16_scale(row + b * 34) * xs[b]);
            }
            y[o] = vaddvq_f32(accv);
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

void matmul_q4_K_sdot(float* y, const void* W, const float* x,
                      int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nsb = n_in / 256;              // super-blocks per row
    const int64_t nb  = n_in / 32;               // activation blocks per row
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nsb * 144;

    std::vector<int8_t>  xq(n_in);
    std::vector<float>   xs(nb);
    std::vector<int32_t> xsum(nb);
    quantize_activation_q8(x, n_in, xq.data(), xs.data());
    for (int64_t b = 0; b < nb; ++b)
        xsum[b] = vaddlvq_s8(vld1q_s8(xq.data() + b * 32))
                + vaddlvq_s8(vld1q_s8(xq.data() + b * 32 + 16));

    auto body = [&](int, int64_t begin, int64_t end) {
        const uint8x16_t lomask = vdupq_n_u8(0x0F);
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            float32x4_t accv = vdupq_n_f32(0.f);  // Σ xs*d*sc*dot (lane-wise)
            float       accm = 0.f;               // Σ xs*dmin*m*Σxq (scalar)
            for (int64_t k = 0; k < nsb; ++k) {
                const uint8_t* p = row + k * 144;
                const float d = f16_scale(p), dmin = f16_scale(p + 2);
                const uint8_t* scales = p + 4;
                const uint8_t* q = p + 16;
                const int8_t*  aq = xq.data() + k * 256;
                const float*   bxs = xs.data() + k * 8;
                const int32_t* bsum = xsum.data() + k * 8;
                for (int j = 0; j < 4; ++j) {     // four 64-weight chunks
                    const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
                    const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, lomask));
                    const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, lomask));
                    const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                    const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                    const int32x4_t plo = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                                              lo0, vld1q_s8(aq)),      lo1, vld1q_s8(aq + 16));
                    const int32x4_t phi = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                                              hi0, vld1q_s8(aq + 32)), hi1, vld1q_s8(aq + 48));
                    uint8_t sc, m;
                    const int b0 = 2 * j;         // activation block of the low half
                    get_scale_min_k4(b0, scales, sc, m);
                    accv = vfmaq_n_f32(accv, vcvtq_f32_s32(plo), d * (float)sc * bxs[b0]);
                    accm += dmin * (float)m * bxs[b0] * (float)bsum[b0];
                    get_scale_min_k4(b0 + 1, scales, sc, m);
                    accv = vfmaq_n_f32(accv, vcvtq_f32_s32(phi), d * (float)sc * bxs[b0 + 1]);
                    accm += dmin * (float)m * bxs[b0 + 1] * (float)bsum[b0 + 1];
                    q += 32; aq += 64;
                }
            }
            y[o] = vaddvq_f32(accv) - accm;
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

void matmul_q6_K_sdot(float* y, const void* W, const float* x,
                      int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nsb = n_in / 256;
    const int64_t nb  = n_in / 32;
    const int64_t ng  = n_in / 16;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nsb * 210;

    std::vector<int8_t>  xq(n_in);
    std::vector<float>   xs(nb);
    std::vector<int32_t> xsum16(ng);
    quantize_activation_q8(x, n_in, xq.data(), xs.data());
    for (int64_t g = 0; g < ng; ++g) xsum16[g] = vaddlvq_s8(vld1q_s8(xq.data() + g * 16));

    auto body = [&](int, int64_t begin, int64_t end) {
        const uint8x16_t m0f = vdupq_n_u8(0x0F), m03 = vdupq_n_u8(0x03);
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            float32x4_t accv = vdupq_n_f32(0.f);
            float       accm = 0.f;
            for (int64_t k = 0; k < nsb; ++k) {
                const uint8_t* p  = row + k * 210;
                const uint8_t* ql = p;
                const uint8_t* qh = p + 128;
                const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192);
                const float d = f16_scale(p + 208);
                for (int h = 0; h < 2; ++h) {                 // two 128-weight halves
                    const int64_t e0 = k * 256 + h * 128;     // first element of this half
                    for (int off = 0; off < 32; off += 16) {  // two 16-wide groups per quarter
                        const int is = off / 16;
                        const uint8x16_t qlv  = vld1q_u8(ql + off);
                        const uint8x16_t qlv2 = vld1q_u8(ql + off + 32);
                        const uint8x16_t qhv  = vld1q_u8(qh + off);
                        const int8x16_t q1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlv,  m0f), vshlq_n_u8(vandq_u8(qhv, m03), 4)));
                        const int8x16_t q2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlv2, m0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), m03), 4)));
                        const int8x16_t q3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlv,  4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), m03), 4)));
                        const int8x16_t q4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlv2, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), m03), 4)));
                        const int8x16_t* qs[4] = {&q1, &q2, &q3, &q4};
                        for (int quarter = 0; quarter < 4; ++quarter) {
                            const int64_t e  = e0 + quarter * 32 + off;      // element index
                            const int32x4_t pd = vdotq_s32(vdupq_n_s32(0), *qs[quarter], vld1q_s8(xq.data() + e));
                            const float s = d * (float)sc[is + 2 * quarter] * xs[e / 32];
                            accv = vfmaq_n_f32(accv, vcvtq_f32_s32(pd), s);
                            accm += s * 32.f * (float)xsum16[e / 16];
                        }
                    }
                    ql += 64; qh += 32; sc += 8;
                }
            }
            y[o] = vaddvq_f32(accv) - accm;
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

} // namespace detail
} // namespace llm

#else  // not built with dotprod (non-ARM target, or the per-file flag is missing)

#if defined(__aarch64__)
#pragma message("src/neon_sdot.cpp built WITHOUT dotprod: --fast int8 kernels disabled on ARM. " \
                "Build this file with -march=armv8.2-a+dotprod (the Makefile and CMake do).")
#endif

namespace llm {
namespace detail {
// Reported to neon.cpp so the runtime check says "no SDOT" honestly; the stubs
// are never reached through the dispatcher but keep the link complete.
bool sdot_kernels_compiled() { return false; }
void matmul_q8_0_sdot(float* y, const void* W, const float* x, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    matmul_quant(y, W, DType::Q8_0, x, n_out, n_in, pool);
}
void matmul_q4_K_sdot(float* y, const void* W, const float* x, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    matmul_quant(y, W, DType::Q4_K, x, n_out, n_in, pool);
}
void matmul_q6_K_sdot(float* y, const void* W, const float* x, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    matmul_quant(y, W, DType::Q6_K, x, n_out, n_in, pool);
}
} // namespace detail
} // namespace llm

#endif
