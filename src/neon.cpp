// neon.cpp — ARM int8 SDOT quantized matmul + fp16 bulk convert (see neon.h).
#include "llm/neon.h"
#include "llm/quant.h"
#include "llm/simd.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace llm {

bool neon_dotprod_available() {
#if defined(__ARM_FEATURE_DOTPROD)
    return true;
#else
    return false;
#endif
}

static bool g_fast_quant = false;
void set_fast_quant(bool on) { g_fast_quant = on; }
bool fast_quant_enabled() { return g_fast_quant; }

void fp16_to_fp32_bulk(const uint16_t* src, float* dst, int64_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(src + i));
        vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
#else
    for (int64_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
#endif
}

namespace {

// Quantize an fp32 activation vector to per-32-block int8 (symmetric), matching
// the Q8 activation format. Returns int8 codes + per-block scale.
void quantize_activation_q8(const float* x, int64_t n, int8_t* q, float* scale) {
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
inline float f16_scale(const uint8_t* p) { __fp16 h; std::memcpy(&h, p, 2); return (float)h; }
#else
inline float f16_scale(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return fp16_to_fp32(h); }
#endif

} // namespace

void matmul_q8_0_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t nb = n_in / 32;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nb * 34;   // Q8_0 block = fp16 d + 32 int8

#if defined(__ARM_FEATURE_DOTPROD)
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
#else
    // No SDOT: defer to the exact fp32-dequant path.
    (void)base; (void)row_bytes;
    matmul_quant(y, W, DType::Q8_0, x, n_out, n_in, pool);
#endif
}

} // namespace llm

// ---------------------------------------------------------------------------
// Q4_K × int8-activation kernel (opt-in via --fast, like matmul_q8_0_i8).
//
// Q4_K super-block (256 weights, 144 bytes): d fp16 | dmin fp16 | scales[12]
// (eight 6-bit (scale, min) pairs) | q[128] nibbles. Chunk j (64 weights) holds
// its low nibbles as weights 64j+0..31 (pair 2j) and its high nibbles as
// 64j+32..63 (pair 2j+1); w = d*sc*q - dmin*m with q in [0,15].
//
// With the activation quantized per 32-block (x ≈ xs[b]*xq), each 32-wide
// sub-block contributes
//     xs[b] * ( d*sc * Σ q*xq  -  dmin*m * Σ xq )
// so the whole row is int8 dot products (SDOT on the 0..15 nibbles, which are
// valid int8) plus two scalar fused-multiply-adds per sub-block. Σxq per
// activation block is computed once and reused across every output row.
// Numerics: same int8-activation approximation as the Q8_0 kernel (and as
// llama.cpp's Q8_K activations); the exact fp32 dequant path stays the default.
namespace llm {

void matmul_q4_K_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool) {
#if defined(__ARM_FEATURE_DOTPROD)
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
#elif LLM_HAVE_AVX2
    // x86-64 (AMD/Intel, Haswell+/Zen+): the same arithmetic with
    // _mm256_maddubs_epi16, whose operand contract is exactly (u8, s8) — the
    // 0..15 nibbles are the unsigned side, the int8 activations the signed
    // side. Pair sums are at most 2*15*127 = 3810, far below int16 saturation.
    // NOT yet run on real x86 hardware (cross-compiled only on 2026-10-19).
    const int64_t nsb = n_in / 256;
    const int64_t nb  = n_in / 32;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nsb * 144;

    std::vector<int8_t>  xq(n_in);
    std::vector<float>   xs(nb);
    std::vector<int32_t> xsum(nb);
    quantize_activation_q8(x, n_in, xq.data(), xs.data());
    for (int64_t b = 0; b < nb; ++b) {
        int32_t t = 0;
        for (int i = 0; i < 32; ++i) t += xq[b * 32 + i];
        xsum[b] = t;
    }

    auto hsum256 = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    };

    auto body = [&](int, int64_t begin, int64_t end) {
        const __m256i lomask = _mm256_set1_epi8(0x0F);
        const __m256i ones16 = _mm256_set1_epi16(1);
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            __m256 accv = _mm256_setzero_ps();
            float  accm = 0.f;
            for (int64_t k = 0; k < nsb; ++k) {
                const uint8_t* p = row + k * 144;
                const float d = f16_scale(p), dmin = f16_scale(p + 2);
                const uint8_t* scales = p + 4;
                const uint8_t* q = p + 16;
                const int8_t*  aq = xq.data() + k * 256;
                const float*   bxs = xs.data() + k * 8;
                const int32_t* bsum = xsum.data() + k * 8;
                for (int j = 0; j < 4; ++j) {
                    const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
                    const __m256i lo = _mm256_and_si256(qv, lomask);
                    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), lomask);
                    const __m256i alo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq));
                    const __m256i ahi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + 32));
                    const __m256i plo = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, alo), ones16);
                    const __m256i phi = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, ahi), ones16);
                    uint8_t sc, m;
                    const int b0 = 2 * j;
                    get_scale_min_k4(b0, scales, sc, m);
                    accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(plo),
                                           _mm256_set1_ps(d * (float)sc * bxs[b0]), accv);
                    accm += dmin * (float)m * bxs[b0] * (float)bsum[b0];
                    get_scale_min_k4(b0 + 1, scales, sc, m);
                    accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(phi),
                                           _mm256_set1_ps(d * (float)sc * bxs[b0 + 1]), accv);
                    accm += dmin * (float)m * bxs[b0 + 1] * (float)bsum[b0 + 1];
                    q += 32; aq += 64;
                }
            }
            y[o] = hsum256(accv) - accm;
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
#else
    // No SDOT / AVX2: defer to the exact fp32-dequant path.
    matmul_quant(y, W, DType::Q4_K, x, n_out, n_in, pool);
#endif
}

} // namespace llm
