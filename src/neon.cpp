// neon.cpp — int8 quantized matmul dispatch + fp16 bulk convert (see neon.h).
// Built for the baseline ISA. The ARM SDOT kernel bodies are in neon_sdot.cpp
// (built with +dotprod) and are only entered after a runtime CPU check; the x86
// AVX2 kernels below stay compile-time (x86 ships separate baseline/AVX2
// binaries, picked by the `sipllm` launcher).
#include "llm/neon.h"
#include "llm/quant.h"
#include "llm/simd.h"
#include "neon_internal.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)   // Linux arm64 uapi: ASIMD dot product (SDOT/UDOT)
#endif
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace llm {

// Runtime CPU feature detection (Issue #68). The SDOT kernels live in
// neon_sdot.cpp, built with +dotprod; everything else is baseline ISA, so the
// binary must ASK the CPU before calling them. Cached after the first call.
// SIPLLM_CPU=generic forces the no-SDOT path (CI and tests use it to exercise
// the fallback on SDOT-capable hardware).
static bool detect_dotprod() {
    if (const char* e = std::getenv("SIPLLM_CPU"))
        if (std::strcmp(e, "generic") == 0) return false;
    if (!detail::sdot_kernels_compiled()) return false;
#if defined(__aarch64__) && defined(__APPLE__)
    int v = 0; size_t n = sizeof(v);
    return sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, nullptr, 0) == 0 && v != 0;
#elif defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

bool cpu_has_dotprod() {
    static const bool has = detect_dotprod();
    return has;
}

bool neon_dotprod_available() { return cpu_has_dotprod(); }

bool int8_kernels_available() {
#if defined(__aarch64__)
    return cpu_has_dotprod();
#elif LLM_HAVE_AVX2
    return true;
#else
    return false;
#endif
}

static bool g_fast_quant = false;
void set_fast_quant(bool on) { g_fast_quant = on; }
// --fast requested AND this CPU has an int8 kernel. When it does not, linear()
// and linear_batch() take the exact path per batch (one weight pass for the
// whole prompt) instead of falling back one row at a time inside each kernel.
bool fast_quant_enabled() { return g_fast_quant && int8_kernels_available(); }

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


void matmul_q8_0_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool) {
#if defined(__aarch64__)
    if (cpu_has_dotprod()) { detail::matmul_q8_0_sdot(y, W, x, n_out, n_in, pool); return; }
#endif
    // No SDOT on this core (and no AVX2 Q8_0 kernel): the exact fp32-dequant path.
    matmul_quant(y, W, DType::Q8_0, x, n_out, n_in, pool);
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
#if defined(__aarch64__)
    if (cpu_has_dotprod()) { detail::matmul_q4_K_sdot(y, W, x, n_out, n_in, pool); return; }
    matmul_quant(y, W, DType::Q4_K, x, n_out, n_in, pool);   // no SDOT on this core: exact path
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

// ---------------------------------------------------------------------------
// Q6_K × int8-activation kernel (opt-in via --fast). Q6_K is the lm_head and
// a few attention/FFN tensors in a Q4_K_M file (21 of 201 in TinyLlama).
//
// Q6_K super-block (256 weights, 210 bytes): ql[128] low nibbles | qh[64] high
// 2-bit pairs | sc[16] int8 (one per 16 weights) | d fp16. Each 128-weight half
// unpacks, for l in [0,32): q1 = ql[l]&0xF | (qh[l]&3)<<4 at l, q2 at l+32,
// q3 = ql[l]>>4 | ... at l+64, q4 at l+96; w = d * sc[g] * (q - 32), q in [0,63].
//
// Scales are per 16 weights, so the dot is taken per 16-wide group (one SDOT
// per group gives exactly that) and the "-32" is folded into a scalar term:
//     xs[b] * d * sc[g] * ( Σ q*xq  -  32 * Σ xq )       (Σ over the 16 group)
// Σxq per 16 is precomputed once per activation and shared across rows.
namespace llm {

void matmul_q6_K_i8(float* y, const void* W, const float* x,
                    int64_t n_out, int64_t n_in, ThreadPool* pool) {
#if defined(__aarch64__)
    if (cpu_has_dotprod()) { detail::matmul_q6_K_sdot(y, W, x, n_out, n_in, pool); return; }
    matmul_quant(y, W, DType::Q6_K, x, n_out, n_in, pool);   // no SDOT on this core: exact path
#elif LLM_HAVE_AVX2
    // Same arithmetic; 32 bytes of ql at once give two 16-wide groups (two
    // scales) per quarter, so the per-lane scale vector carries sc[is] in
    // lanes 0-3 and sc[is+1] in lanes 4-7. Pair sums <= 2*63*127 = 16002.
    // NOT yet run on real x86 hardware (cross-compiled only on 2026-10-19).
    const int64_t nsb = n_in / 256;
    const int64_t nb  = n_in / 32;
    const int64_t ng  = n_in / 16;
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const int64_t row_bytes = nsb * 210;

    std::vector<int8_t>  xq(n_in);
    std::vector<float>   xs(nb);
    std::vector<int32_t> xsum16(ng);
    quantize_activation_q8(x, n_in, xq.data(), xs.data());
    for (int64_t g = 0; g < ng; ++g) {
        int32_t t = 0;
        for (int i = 0; i < 16; ++i) t += xq[g * 16 + i];
        xsum16[g] = t;
    }
    auto hsum256 = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    };
    auto body = [&](int, int64_t begin, int64_t end) {
        const __m256i m0f = _mm256_set1_epi8(0x0F), m03 = _mm256_set1_epi8(0x03);
        const __m256i ones16 = _mm256_set1_epi16(1);
        for (int64_t o = begin; o < end; ++o) {
            const uint8_t* row = base + o * row_bytes;
            __m256 accv = _mm256_setzero_ps();
            float  accm = 0.f;
            for (int64_t k = 0; k < nsb; ++k) {
                const uint8_t* p  = row + k * 210;
                const uint8_t* ql = p;
                const uint8_t* qh = p + 128;
                const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192);
                const float d = f16_scale(p + 208);
                for (int h = 0; h < 2; ++h) {
                    const int64_t e0 = k * 256 + h * 128;
                    const __m256i qlv  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
                    const __m256i qlv2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
                    const __m256i qhv  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
                    const __m256i q[4] = {
                        _mm256_or_si256(_mm256_and_si256(qlv,  m0f), _mm256_slli_epi16(_mm256_and_si256(qhv, m03), 4)),
                        _mm256_or_si256(_mm256_and_si256(qlv2, m0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m03), 4)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlv,  4), m0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m03), 4)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlv2, 4), m0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m03), 4)),
                    };
                    for (int quarter = 0; quarter < 4; ++quarter) {
                        const int64_t e = e0 + quarter * 32;          // 32 elements: groups e/16, e/16+1
                        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq.data() + e));
                        const __m256i pd = _mm256_madd_epi16(_mm256_maddubs_epi16(q[quarter], a), ones16);
                        const float sb = d * xs[e / 32];
                        const float s0 = sb * (float)sc[0 + 2 * quarter], s1 = sb * (float)sc[1 + 2 * quarter];
                        const __m256 sv = _mm256_set_ps(s1, s1, s1, s1, s0, s0, s0, s0);
                        accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pd), sv, accv);
                        accm += 32.f * (s0 * (float)xsum16[e / 16] + s1 * (float)xsum16[e / 16 + 1]);
                    }
                    ql += 64; qh += 32; sc += 8;
                }
            }
            y[o] = hsum256(accv) - accm;
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
#else
    matmul_quant(y, W, DType::Q6_K, x, n_out, n_in, pool);
#endif
}

} // namespace llm
