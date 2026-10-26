// neon_sdot.cpp — the ARM int8 SDOT kernels, compiled with +dotprod (Issue #68).
//
// The rest of the engine is built for the baseline ISA so one binary runs on
// every 64-bit ARM core, including ARMv8.0 parts without SDOT (Cortex-A53).
// Only this file is built with dotprod enabled (Makefile / CMake per-file
// flag); neon.cpp calls into it only after cpu_has_dotprod() said yes at
// runtime. Bodies moved verbatim from neon.cpp's former
// `#if defined(__ARM_FEATURE_DOTPROD)` branches.
#include "neon_internal.h"
#include "llm/common.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <algorithm>
#include <type_traits>
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


// ---------------------------------------------------------------------------
// Batched variants for prefill (wave 14). linear_batch used to call the
// single-vector kernel once per prompt token, so a 32-token chunk re-read and
// re-unpacked every weight row 32 times and quantized the activations inside
// every call. Here the activations of the whole batch are quantized once and
// each weight row is loaded and unpacked once per tile of T tokens, then dotted
// against each token in the tile. For every (row, token) pair the float
// operations are the SAME sequence as the single-vector kernel above, so the
// results are bit-identical to calling it per token (test_neon proves it); only
// the redundant weight traffic is gone.
namespace {
constexpr int kTile = 4;   // tokens per weight-row pass (accumulators stay in registers)

// Per-token activation quantization for a batch, laid out [token][...].
struct BatchQ8 {
    std::vector<int8_t> xq;
    std::vector<float>  xs;
    BatchQ8(const float* X, int64_t bs, int64_t n_in) : xq((size_t)(bs * n_in)), xs((size_t)(bs * (n_in / 32))) {
        for (int64_t t = 0; t < bs; ++t)
            quantize_activation_q8(X + t * n_in, n_in, xq.data() + t * n_in, xs.data() + t * (n_in / 32));
    }
};
} // namespace

void matmul_q8_0_sdot_batch(float* Y, const void* W, const float* X, int64_t bs,
                            int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nb = n_in / 32;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nb * 34;
    const BatchQ8 a(X, bs, n_in);

    auto body = [&](int, int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            for (int64_t t0 = 0; t0 < bs; t0 += kTile) {
                const int nt = (int)std::min<int64_t>(kTile, bs - t0);
                float32x4_t accv[kTile];
                for (int t = 0; t < kTile; ++t) accv[t] = vdupq_n_f32(0.f);
                for (int64_t b = 0; b < nb; ++b) {
                    const int8_t* wq = reinterpret_cast<const int8_t*>(row + b * 34 + 2);
                    const int8x16_t w0 = vld1q_s8(wq), w1 = vld1q_s8(wq + 16);
                    const float wd = f16_scale(row + b * 34);
                    for (int t = 0; t < nt; ++t) {
                        const int8_t* aq = a.xq.data() + (t0 + t) * n_in + b * 32;
                        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(aq)),
                                                w1, vld1q_s8(aq + 16));
                        accv[t] = vfmaq_n_f32(accv[t], vcvtq_f32_s32(p), wd * a.xs[(t0 + t) * nb + b]);
                    }
                }
                for (int t = 0; t < nt; ++t) Y[(t0 + t) * n_out + o] = vaddvq_f32(accv[t]);
            }
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

void matmul_q4_K_sdot_batch(float* Y, const void* W, const float* X, int64_t bs,
                            int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nsb = n_in / 256;
    const int64_t nb  = n_in / 32;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nsb * 144;
    const BatchQ8 a(X, bs, n_in);
    std::vector<int32_t> xsum((size_t)(bs * nb));
    for (int64_t t = 0; t < bs; ++t)
        for (int64_t b = 0; b < nb; ++b) {
            const int8_t* q = a.xq.data() + t * n_in + b * 32;
            xsum[t * nb + b] = vaddlvq_s8(vld1q_s8(q)) + vaddlvq_s8(vld1q_s8(q + 16));
        }

    auto body = [&](int, int64_t begin, int64_t end) {
        const uint8x16_t lomask = vdupq_n_u8(0x0F);
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            for (int64_t t0 = 0; t0 < bs; t0 += kTile) {
                const int nt = (int)std::min<int64_t>(kTile, bs - t0);
                float32x4_t accv[kTile];
                float       accm[kTile];
                for (int t = 0; t < kTile; ++t) { accv[t] = vdupq_n_f32(0.f); accm[t] = 0.f; }
                for (int64_t k = 0; k < nsb; ++k) {
                    const uint8_t* p = row + k * 144;
                    const float d = f16_scale(p), dmin = f16_scale(p + 2);
                    const uint8_t* scales = p + 4;
                    const uint8_t* q = p + 16;
                    for (int j = 0; j < 4; ++j) {
                        const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
                        const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, lomask));
                        const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, lomask));
                        const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                        const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                        const int b0 = 2 * j;
                        uint8_t sc0, m0, sc1, m1;
                        get_scale_min_k4(b0, scales, sc0, m0);
                        get_scale_min_k4(b0 + 1, scales, sc1, m1);
                        for (int t = 0; t < nt; ++t) {
                            const int64_t tt = t0 + t;
                            const int8_t*  aq   = a.xq.data() + tt * n_in + k * 256 + j * 64;
                            const float*   bxs  = a.xs.data() + tt * nb + k * 8;
                            const int32_t* bsum = xsum.data() + tt * nb + k * 8;
                            const int32x4_t plo = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                                                      lo0, vld1q_s8(aq)),      lo1, vld1q_s8(aq + 16));
                            const int32x4_t phi = vdotq_s32(vdotq_s32(vdupq_n_s32(0),
                                                      hi0, vld1q_s8(aq + 32)), hi1, vld1q_s8(aq + 48));
                            accv[t] = vfmaq_n_f32(accv[t], vcvtq_f32_s32(plo), d * (float)sc0 * bxs[b0]);
                            accm[t] += dmin * (float)m0 * bxs[b0] * (float)bsum[b0];
                            accv[t] = vfmaq_n_f32(accv[t], vcvtq_f32_s32(phi), d * (float)sc1 * bxs[b0 + 1]);
                            accm[t] += dmin * (float)m1 * bxs[b0 + 1] * (float)bsum[b0 + 1];
                        }
                        q += 32;
                    }
                }
                for (int t = 0; t < nt; ++t) Y[(t0 + t) * n_out + o] = vaddvq_f32(accv[t]) - accm[t];
            }
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

void matmul_q6_K_sdot_batch(float* Y, const void* W, const float* X, int64_t bs,
                            int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nsb = n_in / 256;
    const int64_t nb  = n_in / 32;
    const int64_t ng  = n_in / 16;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nsb * 210;
    const BatchQ8 a(X, bs, n_in);
    std::vector<int32_t> xsum16((size_t)(bs * ng));
    for (int64_t t = 0; t < bs; ++t)
        for (int64_t g = 0; g < ng; ++g)
            xsum16[t * ng + g] = vaddlvq_s8(vld1q_s8(a.xq.data() + t * n_in + g * 16));

    auto body = [&](int, int64_t begin, int64_t end) {
        const uint8x16_t m0f = vdupq_n_u8(0x0F), m03 = vdupq_n_u8(0x03);
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            for (int64_t t0 = 0; t0 < bs; t0 += kTile) {
                const int nt = (int)std::min<int64_t>(kTile, bs - t0);
                float32x4_t accv[kTile];
                float       accm[kTile];
                for (int t = 0; t < kTile; ++t) { accv[t] = vdupq_n_f32(0.f); accm[t] = 0.f; }
                for (int64_t k = 0; k < nsb; ++k) {
                    const uint8_t* p  = row + k * 210;
                    const uint8_t* ql = p;
                    const uint8_t* qh = p + 128;
                    const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192);
                    const float d = f16_scale(p + 208);
                    for (int h = 0; h < 2; ++h) {
                        const int64_t e0 = k * 256 + h * 128;
                        for (int off = 0; off < 32; off += 16) {
                            const int is = off / 16;
                            const uint8x16_t qlv  = vld1q_u8(ql + off);
                            const uint8x16_t qlv2 = vld1q_u8(ql + off + 32);
                            const uint8x16_t qhv  = vld1q_u8(qh + off);
                            const int8x16_t q1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlv,  m0f), vshlq_n_u8(vandq_u8(qhv, m03), 4)));
                            const int8x16_t q2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlv2, m0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), m03), 4)));
                            const int8x16_t q3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlv,  4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), m03), 4)));
                            const int8x16_t q4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlv2, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), m03), 4)));
                            const int8x16_t qv[4] = {q1, q2, q3, q4};
                            for (int quarter = 0; quarter < 4; ++quarter) {
                                const int64_t e = e0 + quarter * 32 + off;
                                const float dsc = d * (float)sc[is + 2 * quarter];
                                for (int t = 0; t < nt; ++t) {
                                    const int64_t tt = t0 + t;
                                    const int32x4_t pd = vdotq_s32(vdupq_n_s32(0), qv[quarter],
                                                                   vld1q_s8(a.xq.data() + tt * n_in + e));
                                    const float s = dsc * a.xs[tt * nb + e / 32];
                                    accv[t] = vfmaq_n_f32(accv[t], vcvtq_f32_s32(pd), s);
                                    accm[t] += s * 32.f * (float)xsum16[tt * ng + e / 16];
                                }
                            }
                        }
                        ql += 64; qh += 32; sc += 8;
                    }
                }
                for (int t = 0; t < nt; ++t) Y[(t0 + t) * n_out + o] = vaddvq_f32(accv[t]) - accm[t];
            }
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}


// ---------------------------------------------------------------------------
// Wave 15: Q4_K x int8 on IL4-interleaved weights (see BlockQ4KIL4). Same
// activation quantization as matmul_q4_K_sdot (int8 per 32 values); the four
// rows of a group live in the four lanes of every accumulator, so the per-32
// float scaling runs once per 4 rows instead of once per row. Only the float
// summation order differs from the row-major kernels (results agree to ~1e-6
// relative; test_neon bounds it).
void matmul_q4_K_il4_sdot(float* Y, const void* Wil4, const float* X, int64_t bs,
                          int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nsb = n_in / 256, nb = n_in / 32, ng = n_out / 4;
    const BlockQ4KIL4* Wp = static_cast<const BlockQ4KIL4*>(Wil4);
    const BatchQ8 a(X, bs, n_in);
    // Per sub-block activation term for the mins: xs * sum(q). Same float product
    // the inner loop used to form inline, now computed once per token.
    std::vector<float> xm((size_t)(bs * nb));
    for (int64_t t = 0; t < bs; ++t)
        for (int64_t b = 0; b < nb; ++b) {
            const int8_t* q = a.xq.data() + t * n_in + b * 32;
            const int32_t s = vaddlvq_s8(vld1q_s8(q)) + vaddlvq_s8(vld1q_s8(q + 16));
            xm[t * nb + b] = a.xs[t * nb + b] * (float)s;
        }
    auto bytes_to_f32 = [](uint8x16_t v, float32x4_t* o) {   // 16 bytes -> 4 x float32x4
        const uint16x8_t l = vmovl_u8(vget_low_u8(v)), h = vmovl_u8(vget_high_u8(v));
        o[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(l))); o[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(l)));
        o[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h))); o[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h)));
    };
    auto body = [&](int, int64_t g0, int64_t g1) {
        const uint8x16_t m4 = vdupq_n_u8(0x0F), c63 = vdupq_n_u8(63), c15 = vdupq_n_u8(15);
        // Decoded scales of one group, all super-blocks: [k][0..7] = d*sc, [k][8..15] = dmin*m
        // per sub-block, one row per lane. Decoded once per group, shared by every token tile.
        // Per-thread and reused: body() runs once per scheduler chunk, many times per token.
        thread_local std::vector<float32x4_t> scl;
        if (scl.size() < (size_t)nsb * 16) scl.resize((size_t)nsb * 16);
        for (int64_t g = g0; g < g1; ++g) {
            for (int64_t k = 0; k < nsb; ++k) {
                const BlockQ4KIL4& b = Wp[g * nsb + k];
                // get_scale_min_k4 for 8 sub-blocks x 4 rows from 3 vector loads;
                // lanes are [sub-block (0..3 | 4..7)][row].
                const uint8x16_t q0 = vld1q_u8(b.scales[0]), q4 = vld1q_u8(b.scales[4]), q8 = vld1q_u8(b.scales[8]);
                float32x4_t sf[8], mf[8];
                bytes_to_f32(vandq_u8(q0, c63), sf);
                bytes_to_f32(vorrq_u8(vandq_u8(q8, c15), vshlq_n_u8(vshrq_n_u8(q0, 6), 4)), sf + 4);
                bytes_to_f32(vandq_u8(q4, c63), mf);
                bytes_to_f32(vorrq_u8(vshrq_n_u8(q8, 4), vshlq_n_u8(vshrq_n_u8(q4, 6), 4)), mf + 4);
                // fp16 -> fp32 is exact; one vector convert covers the 4 rows.
                const float32x4_t d4  = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b.d)));
                const float32x4_t dm4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b.dmin)));
                float32x4_t* s = scl.data() + k * 16;
                for (int i = 0; i < 8; ++i) { s[i] = vmulq_f32(d4, sf[i]); s[8 + i] = vmulq_f32(dm4, mf[i]); }
            }
            // NT tokens per tile as a compile-time constant: the accumulators stay
            // in registers (a runtime count put them on the stack, and every FMA
            // then waited on a store-to-load round trip).
            auto tile = [&](auto NTc, int64_t t0) {
                constexpr int nt = decltype(NTc)::value;
                float32x4_t accv[nt], accm[nt];
                for (int t = 0; t < nt; ++t) { accv[t] = vdupq_n_f32(0.f); accm[t] = vdupq_n_f32(0.f); }
                for (int64_t k = 0; k < nsb; ++k) {
                    const BlockQ4KIL4& b = Wp[g * nsb + k];
                    const float32x4_t* s = scl.data() + k * 16;
                    for (int j = 0; j < 4; ++j) {
                        uint8x16_t w[8];
                        for (int v = 0; v < 8; ++v) w[v] = vld1q_u8(b.qs[j][v]);
                        for (int t = 0; t < nt; ++t) {
                            const int64_t tt = t0 + t;
                            const int8_t* aq = a.xq.data() + tt * n_in + k * 256 + j * 64;
                            const int8x16_t a0 = vld1q_s8(aq), a1 = vld1q_s8(aq + 16);
                            const int8x16_t a2 = vld1q_s8(aq + 32), a3 = vld1q_s8(aq + 48);
                            // Two independent chains per half (integer, so the split is exact):
                            // one token alone no longer waits on 8 dependent SDOTs.
                            // One token alone: two chains per half, so it does not wait on
                            // 8 dependent SDOTs. Two or more tokens already give independent
                            // chains, so skip the extra adds (integer either way: exact).
                            constexpr bool split = nt == 1;
                            int32x4_t lo0 = vdupq_n_s32(0), lo1 = vdupq_n_s32(0), hi0 = vdupq_n_s32(0), hi1 = vdupq_n_s32(0);
#define LLM_IL4_LO(acc, v, A, L) (acc) = vdotq_laneq_s32((acc), vreinterpretq_s8_u8(vandq_u8(w[v], m4)), A, L)
#define LLM_IL4_HI(acc, v, A, L) (acc) = vdotq_laneq_s32((acc), vreinterpretq_s8_u8(vshrq_n_u8(w[v], 4)), A, L)
                            LLM_IL4_LO(lo0, 0, a0, 0); LLM_IL4_LO(split ? lo1 : lo0, 1, a0, 1); LLM_IL4_LO(lo0, 2, a0, 2); LLM_IL4_LO(split ? lo1 : lo0, 3, a0, 3);
                            LLM_IL4_LO(lo0, 4, a1, 0); LLM_IL4_LO(split ? lo1 : lo0, 5, a1, 1); LLM_IL4_LO(lo0, 6, a1, 2); LLM_IL4_LO(split ? lo1 : lo0, 7, a1, 3);
                            LLM_IL4_HI(hi0, 0, a2, 0); LLM_IL4_HI(split ? hi1 : hi0, 1, a2, 1); LLM_IL4_HI(hi0, 2, a2, 2); LLM_IL4_HI(split ? hi1 : hi0, 3, a2, 3);
                            LLM_IL4_HI(hi0, 4, a3, 0); LLM_IL4_HI(split ? hi1 : hi0, 5, a3, 1); LLM_IL4_HI(hi0, 6, a3, 2); LLM_IL4_HI(split ? hi1 : hi0, 7, a3, 3);
#undef LLM_IL4_LO
#undef LLM_IL4_HI
                            int32x4_t lo = lo0, hi = hi0;
                            if constexpr (split) { lo = vaddq_s32(lo0, lo1); hi = vaddq_s32(hi0, hi1); }
                            const int64_t sb = tt * nb + k * 8 + 2 * j;
                            accv[t] = vfmaq_f32(accv[t], vcvtq_f32_s32(lo), vmulq_n_f32(s[2 * j], a.xs[sb]));
                            accv[t] = vfmaq_f32(accv[t], vcvtq_f32_s32(hi), vmulq_n_f32(s[2 * j + 1], a.xs[sb + 1]));
                            accm[t] = vfmaq_n_f32(accm[t], s[8 + 2 * j], xm[sb]);
                            accm[t] = vfmaq_n_f32(accm[t], s[8 + 2 * j + 1], xm[sb + 1]);
                        }
                    }
                }
                for (int t = 0; t < nt; ++t) {
                    float r[4];
                    vst1q_f32(r, vsubq_f32(accv[t], accm[t]));
                    float* y = Y + (t0 + t) * n_out + g * 4;
                    y[0] = r[0]; y[1] = r[1]; y[2] = r[2]; y[3] = r[3];
                }
            };
            int64_t t0 = 0;
            for (; t0 + kTile <= bs; t0 += kTile) tile(std::integral_constant<int, kTile>{}, t0);
            switch (bs - t0) {
                case 1: tile(std::integral_constant<int, 1>{}, t0); break;
                case 2: tile(std::integral_constant<int, 2>{}, t0); break;
                case 3: tile(std::integral_constant<int, 3>{}, t0); break;
                default: break;
            }
        }
    };
    if (pool && pool->size() > 1 && ng >= 8) pool->parallel_for(ng, body);
    else body(0, 0, ng);
}

// ---------------------------------------------------------------------------
// Wave 16: Q6_K x int8 on IL4-interleaved weights (see BlockQ6KIL4). Same
// activation quantization as matmul_q6_K_sdot (int8 per 32 values). Q6_K's -32
// offset is the starting value of each SDOT accumulator (-32 * sum of the 16
// activations) and the int8 sub-block scales are applied in integer, so float
// work is one convert + one FMA per 32 values per 4 rows. Exact integer up to
// that point; only the float summation order differs from the row-major
// kernel (test_neon bounds it).
void matmul_q6_K_il4_sdot(float* Y, const void* Wil4, const float* X, int64_t bs,
                          int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nsb = n_in / 256, nb = n_in / 32, n16 = n_in / 16, ng = n_out / 4;
    const BlockQ6KIL4* Wp = static_cast<const BlockQ6KIL4*>(Wil4);
    const BatchQ8 a(X, bs, n_in);
    std::vector<int32_t> x32((size_t)(bs * n16));   // -32 * sum(q) per 16 activations
    for (int64_t t = 0; t < bs; ++t)
        for (int64_t i = 0; i < n16; ++i)
            x32[t * n16 + i] = -32 * (int32_t)vaddlvq_s8(vld1q_s8(a.xq.data() + t * n_in + i * 16));
    auto body = [&](int, int64_t g0, int64_t g1) {
        const uint8x16_t m03 = vdupq_n_u8(0x03), m30 = vdupq_n_u8(0x30);
        // Per group, all super-blocks: 16 sub-block scales (int32, row per lane)
        // and d (fp32, row per lane). Decoded once, shared by every token tile.
        thread_local std::vector<int32x4_t> scl;     // per-thread, reused (see the Q4_K kernel)
        thread_local std::vector<float32x4_t> dl;
        if (scl.size() < (size_t)nsb * 16) scl.resize((size_t)nsb * 16);
        if (dl.size() < (size_t)nsb) dl.resize((size_t)nsb);
        for (int64_t g = g0; g < g1; ++g) {
            for (int64_t k = 0; k < nsb; ++k) {
                const BlockQ6KIL4& b = Wp[g * nsb + k];
                dl[k] = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b.d)));
                for (int i = 0; i < 16; i += 4) {
                    const int8x16_t s = vld1q_s8(b.scales[i]);
                    const int16x8_t l = vmovl_s8(vget_low_s8(s)), h = vmovl_s8(vget_high_s8(s));
                    int32x4_t* o = scl.data() + k * 16 + i;
                    o[0] = vmovl_s16(vget_low_s16(l)); o[1] = vmovl_s16(vget_high_s16(l));
                    o[2] = vmovl_s16(vget_low_s16(h)); o[3] = vmovl_s16(vget_high_s16(h));
                }
            }
            auto tile = [&](auto NTc, int64_t t0) {   // compile-time NT, as in the Q4_K kernel
                constexpr int nt = decltype(NTc)::value;
                float32x4_t accv[nt];
                for (int t = 0; t < nt; ++t) accv[t] = vdupq_n_f32(0.f);
                for (int64_t k = 0; k < nsb; ++k) {
                    const BlockQ6KIL4& b = Wp[g * nsb + k];
                    const int32x4_t* sc = scl.data() + k * 16;
                    const float32x4_t d4 = dl[k];
                    if constexpr (nt == 1) {
                        // One token (decode): two passes per half, quarters {0,2} then
                        // {1,3}, so each pass reads one ql vector and one qh vector per
                        // step (the quarter loop below reads qh four times and ql twice)
                        // while about 16 vector registers stay live: no spills, which an
                        // in-order core (Cortex-A55) cannot hide. Integer sums are exact
                        // and the float FMAs run in quarter order, so results are identical.
                        const int8_t* xrow = a.xq.data() + t0 * n_in;
                        const int32_t* xi = x32.data() + t0 * n16;
                        const float* xsr = a.xs.data() + t0 * nb;
                        for (int h = 0; h < 2; ++h) {
                            const int64_t b0 = k * 8 + h * 4;   // activation block of quarter 0
                            int32x4_t sv[4];
                            auto pass = [&](auto Pc) {           // P = 0: quarters 0,2; P = 1: quarters 1,3
                                constexpr int P = decltype(Pc)::value, Qa = P, Qb = P + 2;
                                const int8x16_t aA0 = vld1q_s8(xrow + (b0 + Qa) * 32), aA1 = vld1q_s8(xrow + (b0 + Qa) * 32 + 16);
                                const int8x16_t aB0 = vld1q_s8(xrow + (b0 + Qb) * 32), aB1 = vld1q_s8(xrow + (b0 + Qb) * 32 + 16);
                                int32x4_t dA0 = vdupq_n_s32(xi[2 * (b0 + Qa)]), dA1 = vdupq_n_s32(xi[2 * (b0 + Qa) + 1]);
                                int32x4_t dB0 = vdupq_n_s32(xi[2 * (b0 + Qb)]), dB1 = vdupq_n_s32(xi[2 * (b0 + Qb) + 1]);
                                auto step = [&](auto Vc) {
                                    constexpr int v = decltype(Vc)::value, lane = v % 4;
                                    const uint8x16_t L = vld1q_u8(b.ql[h][v + 8 * P]), H = vld1q_u8(b.qh[h][v]);
                                    int8x16_t qa, qb;
                                    if constexpr (P == 0) {
                                        qa = vreinterpretq_s8_u8(vsliq_n_u8(L, vandq_u8(H, m03), 4));
                                        qb = vreinterpretq_s8_u8(vsriq_n_u8(vandq_u8(H, m30), L, 4));
                                    } else {
                                        qa = vreinterpretq_s8_u8(vsliq_n_u8(L, vandq_u8(vshrq_n_u8(H, 2), m03), 4));
                                        qb = vreinterpretq_s8_u8(vsriq_n_u8(vshrq_n_u8(H, 2), L, 4));
                                    }
                                    if constexpr (v < 4) { dA0 = vdotq_laneq_s32(dA0, qa, aA0, lane); dB0 = vdotq_laneq_s32(dB0, qb, aB0, lane); }
                                    else                 { dA1 = vdotq_laneq_s32(dA1, qa, aA1, lane); dB1 = vdotq_laneq_s32(dB1, qb, aB1, lane); }
                                };
                                step(std::integral_constant<int, 0>{}); step(std::integral_constant<int, 1>{});
                                step(std::integral_constant<int, 2>{}); step(std::integral_constant<int, 3>{});
                                step(std::integral_constant<int, 4>{}); step(std::integral_constant<int, 5>{});
                                step(std::integral_constant<int, 6>{}); step(std::integral_constant<int, 7>{});
                                sv[Qa] = vmlaq_s32(vmulq_s32(dA0, sc[8 * h + 2 * Qa]), dA1, sc[8 * h + 2 * Qa + 1]);
                                sv[Qb] = vmlaq_s32(vmulq_s32(dB0, sc[8 * h + 2 * Qb]), dB1, sc[8 * h + 2 * Qb + 1]);
                            };
                            pass(std::integral_constant<int, 0>{});
                            pass(std::integral_constant<int, 1>{});
                            for (int Q = 0; Q < 4; ++Q)
                                accv[0] = vfmaq_f32(accv[0], vcvtq_f32_s32(sv[Q]), vmulq_n_f32(d4, xsr[b0 + Q]));
                        }
                        continue;
                    }
                    for (int h = 0; h < 2; ++h) {
                        // Quarter Q of this half: 32 values, ql low (Q 0,1) or high
                        // (Q 2,3) nibble, qh bits 2Q..2Q+1, scales 8h+2Q and 8h+2Q+1.
                        auto quarter = [&](auto Qc) {
                            constexpr int Q = decltype(Qc)::value;
                            int8x16_t q[8];
                            for (int v = 0; v < 8; ++v) {
                                const uint8x16_t L = vld1q_u8(b.ql[h][v + ((Q & 1) ? 8 : 0)]);
                                const uint8x16_t H = vld1q_u8(b.qh[h][v]);
                                uint8x16_t u;
                                if constexpr (Q == 0) u = vsliq_n_u8(L, vandq_u8(H, m03), 4);
                                else if constexpr (Q == 1) u = vsliq_n_u8(L, vandq_u8(vshrq_n_u8(H, 2), m03), 4);
                                else if constexpr (Q == 2) u = vsriq_n_u8(vandq_u8(H, m30), L, 4);
                                else u = vsriq_n_u8(vshrq_n_u8(H, 2), L, 4);
                                q[v] = vreinterpretq_s8_u8(u);   // 0..63
                            }
                            const int64_t bi = k * 8 + h * 4 + Q;      // 32-wide activation block
                            const int32x4_t sA = sc[8 * h + 2 * Q], sB = sc[8 * h + 2 * Q + 1];
                            for (int t = 0; t < nt; ++t) {
                                const int64_t tt = t0 + t;
                                const int8_t* aq = a.xq.data() + tt * n_in + bi * 32;
                                const int8x16_t a0 = vld1q_s8(aq), a1 = vld1q_s8(aq + 16);
                                // Two chains per token here (nt >= 2 in practice: nt == 1 takes
                                // the two-pass path above), so several tokens interleave.
                                int32x4_t dA = vdupq_n_s32(x32[tt * n16 + 2 * bi]);
                                int32x4_t dB = vdupq_n_s32(x32[tt * n16 + 2 * bi + 1]);
                                dA = vdotq_laneq_s32(dA, q[0], a0, 0); dB = vdotq_laneq_s32(dB, q[4], a1, 0);
                                dA = vdotq_laneq_s32(dA, q[1], a0, 1); dB = vdotq_laneq_s32(dB, q[5], a1, 1);
                                dA = vdotq_laneq_s32(dA, q[2], a0, 2); dB = vdotq_laneq_s32(dB, q[6], a1, 2);
                                dA = vdotq_laneq_s32(dA, q[3], a0, 3); dB = vdotq_laneq_s32(dB, q[7], a1, 3);
                                const int32x4_t s = vmlaq_s32(vmulq_s32(dA, sA), dB, sB);
                                accv[t] = vfmaq_f32(accv[t], vcvtq_f32_s32(s), vmulq_n_f32(d4, a.xs[tt * nb + bi]));
                            }
                        };
                        quarter(std::integral_constant<int, 0>{});
                        quarter(std::integral_constant<int, 1>{});
                        quarter(std::integral_constant<int, 2>{});
                        quarter(std::integral_constant<int, 3>{});
                    }
                }
                for (int t = 0; t < nt; ++t) vst1q_f32(Y + (t0 + t) * n_out + g * 4, accv[t]);
            };
            int64_t t0 = 0;
            for (; t0 + kTile <= bs; t0 += kTile) tile(std::integral_constant<int, kTile>{}, t0);
            switch (bs - t0) {
                case 1: tile(std::integral_constant<int, 1>{}, t0); break;
                case 2: tile(std::integral_constant<int, 2>{}, t0); break;
                case 3: tile(std::integral_constant<int, 3>{}, t0); break;
                default: break;
            }
        }
    };
    if (pool && pool->size() > 1 && ng >= 8) pool->parallel_for(ng, body);
    else body(0, 0, ng);
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
void matmul_q8_0_sdot_batch(float* Y, const void* W, const float* X, int64_t bs, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    matmul_quant_batch(Y, W, DType::Q8_0, X, bs, n_out, n_in, pool);
}
void matmul_q4_K_sdot_batch(float* Y, const void* W, const float* X, int64_t bs, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    matmul_quant_batch(Y, W, DType::Q4_K, X, bs, n_out, n_in, pool);
}
void matmul_q6_K_sdot_batch(float* Y, const void* W, const float* X, int64_t bs, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    matmul_quant_batch(Y, W, DType::Q6_K, X, bs, n_out, n_in, pool);
}
void matmul_q4_K_il4_sdot(float*, const void*, const float*, int64_t, int64_t, int64_t, ThreadPool*) {
    // Unreachable: weights are only repacked when cpu_has_dotprod() is true,
    // which requires this file to have been built with dotprod.
    LLM_CHECK(false, "IL4 weights without the SDOT kernel compiled in");
}
void matmul_q6_K_il4_sdot(float*, const void*, const float*, int64_t, int64_t, int64_t, ThreadPool*) {
    LLM_CHECK(false, "IL4 weights without the SDOT kernel compiled in");
}
} // namespace detail
} // namespace llm

#endif
