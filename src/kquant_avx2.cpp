// kquant_avx2.cpp — Q4_K / Q5_K / Q6_K x Q8_K integer dot products (see kquant.h).
//
// The AVX2 kernels are ports of ggml-cpu's x86 implementations
// (ggml/src/ggml-cpu/arch/x86/quants.c, `#if defined __AVX2__` branches), with
// one structural change: each kernel is a template over NR activation rows so
// the prefill path can unpack a weight block once and reuse it for NR rows.
// The per-row instruction sequence is identical for every NR, so batched
// results are bit-identical to single-row results.
//
// The scalar *_ref functions are ports of ggml's *_generic versions and are
// always compiled (fallback on non-AVX2 hosts, e.g. ARM, and test oracle).
#include "llm/kquant.h"
#include "llm/common.h"
#include "llm/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define LLM_KQ_AVX2 1
#else
#define LLM_KQ_AVX2 0
#endif

namespace llm {

bool kquant_simd_available() { return LLM_KQ_AVX2 != 0; }

namespace {

constexpr uint32_t kmask1 = 0x3f3f3f3f;
constexpr uint32_t kmask2 = 0x0f0f0f0f;
constexpr uint32_t kmask3 = 0x03030303;

// Unpack Q4_K/Q5_K's 12 packed bytes into utmp[0..1] = 8 scales (u8),
// utmp[2..3] = 8 mins (u8). Verbatim from ggml.
inline void unpack_scales_mins(const uint8_t* packed, uint32_t utmp[4]) {
    std::memcpy(utmp, packed, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// ggml's nearest_int: round-half-to-even via the 1.5*2^23 magic number.
// Valid for |fval| <= 4194303. Kept as its own function/statement so the
// preceding multiply is never contracted into an FMA.
inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Force `v` to be materialized (rounded to fp32) before its next use. GCC
// defaults to -ffp-contract=fast and, after inlining nearest_int, fuses
// `iscale*x + 12582912.f` into one FMA — which rounds differently from ggml's
// unfused product on halfway cases. An empty asm with a register in/out
// operand is an optimization barrier that blocks the contraction.
inline void no_contract(float& v) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __asm__("" : "+x"(v));
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__("" : "+w"(v));
#elif defined(__GNUC__)
    __asm__("" : "+m"(v));
#else
    volatile float t = v; v = t;
#endif
}

} // namespace

// ---- Q8_K activation quantizer (ggml quantize_row_q8_K_ref) ---------------
void quantize_row_q8_K(const float* x, block_q8_K* y, int64_t n) {
    LLM_CHECK(n % kQK_K == 0, "quantize_row_q8_K: n must be a multiple of 256");
    const int64_t nb = n / kQK_K;
    for (int64_t i = 0; i < nb; ++i) {
        float max = 0, amax = 0;
        for (int j = 0; j < kQK_K; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, sizeof(y[i].qs));
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));   // ggml leaves these unset
            x += kQK_K;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < kQK_K; ++j) {
            float s = iscale * x[j];
            no_contract(s);
            const int v = nearest_int(s);
            y[i].qs[j] = (int8_t)std::min(127, v);
        }
        for (int j = 0; j < kQK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1 / iscale;
        x += kQK_K;
    }
}

// ---- scalar references (ggml *_generic ports) -----------------------------
float vec_dot_q4_K_q8_K_ref(int64_t n, const void* vw, const block_q8_K* y) {
    const block_q4_K* x = static_cast<const block_q4_K*>(vw);
    const int64_t nb = n / kQK_K;
    uint32_t utmp[4];
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
    const uint8_t* mins   = reinterpret_cast<const uint8_t*>(&utmp[2]);
    int8_t aux8[kQK_K];
    int16_t aux16[8];
    float sums[8] = {0};
    int32_t aux32[8];
    float sumf = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        int8_t* a = aux8;
        for (int j = 0; j < kQK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        unpack_scales_mins(x[i].scales, utmp);
        int sumi = 0;
        for (int j = 0; j < kQK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < kQK_K / 32; ++j) {
            const int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

float vec_dot_q5_K_q8_K_ref(int64_t n, const void* vw, const block_q8_K* y) {
    const block_q5_K* x = static_cast<const block_q5_K*>(vw);
    const int64_t nb = n / kQK_K;
    uint32_t utmp[4];
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
    const uint8_t* mins   = reinterpret_cast<const uint8_t*>(&utmp[2]);
    int8_t aux8[kQK_K];
    int16_t aux16[8];
    float sums[8] = {0};
    int32_t aux32[8];
    float sumf = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].qs;
        const uint8_t* hm = x[i].qh;
        const int8_t* q8 = y[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        int8_t* a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < kQK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        unpack_scales_mins(x[i].scales, utmp);
        int sumi = 0;
        for (int j = 0; j < kQK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < kQK_K / 32; ++j) {
            const int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

float vec_dot_q6_K_q8_K_ref(int64_t n, const void* vw, const block_q8_K* y) {
    const block_q6_K* x = static_cast<const block_q6_K*>(vw);
    const int64_t nb = n / kQK_K;
    int8_t aux8[kQK_K];
    int16_t aux16[8];
    float sums[8] = {0};
    int32_t aux32[8];
    float sumf = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        int8_t* a = aux8;
        for (int j = 0; j < kQK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128; q4 += 64; qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < kQK_K / 16; ++j) {
            const int scale = x[i].scales[is++];
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

// ---- multi-row kernels ----------------------------------------------------
// kernel<NR>(w, y[NR], nb, out[NR]): out[r] = dot(weight row w, Q8_K row y[r]).
namespace {

#if LLM_KQ_AVX2

inline float f16s(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_fp32(h);
#endif
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

alignas(32) const uint8_t k_shuffle_k4[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};
inline __m256i get_scale_shuffle_k4(int i) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_shuffle_k4) + i);
}

alignas(16) const uint8_t k_shuffle_q6[128] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,
};
inline __m128i get_scale_shuffle(int i) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(k_shuffle_q6) + i);
}

template <int NR>
void kernel_q4_K(const void* vw, const block_q8_K* const* y, int64_t nb, float* out) {
    const block_q4_K* x = static_cast<const block_q4_K*>(vw);
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[NR];
    __m128 acc_m[NR];
    for (int r = 0; r < NR; ++r) { acc[r] = _mm256_setzero_ps(); acc_m[r] = _mm_setzero_ps(); }
    uint32_t utmp[4];

    for (int64_t i = 0; i < nb; ++i) {
        const float dw  = f16s(x[i].d);
        const float dmw = f16s(x[i].dmin);
        unpack_scales_mins(x[i].scales, utmp);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m256i scales  = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(mins_and_scales));

        float d[NR];
        for (int r = 0; r < NR; ++r) {
            d[r] = y[r][i].d * dw;
            const float dmin = -y[r][i].d * dmw;
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[r][i].bsums));
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                               _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins128, q8s);
            acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[r]);
        }

        const uint8_t* q4 = x[i].qs;
        __m256i sumi[NR];
        for (int r = 0; r < NR; ++r) sumi[r] = _mm256_setzero_si256();
        for (int j = 0; j < kQK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4)); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            for (int r = 0; r < NR; ++r) {
                const int8_t* q8 = y[r][i].qs + j * 64;
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                const __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
                const __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16l, p16h));
            }
        }
        for (int r = 0; r < NR; ++r)
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d[r]), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
    }
    for (int r = 0; r < NR; ++r) {
        __m128 am = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        am = _mm_add_ss(am, _mm_movehdup_ps(am));
        out[r] = hsum_float_8(acc[r]) + _mm_cvtss_f32(am);
    }
}

template <int NR>
void kernel_q5_K(const void* vw, const block_q8_K* const* y, int64_t nb, float* out) {
    const block_q5_K* x = static_cast<const block_q5_K*>(vw);
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[NR];
    float summs[NR];
    for (int r = 0; r < NR; ++r) { acc[r] = _mm256_setzero_ps(); summs[r] = 0.f; }
    uint32_t utmp[4];

    for (int64_t i = 0; i < nb; ++i) {
        const float dw  = f16s(x[i].d);
        const float dmw = f16s(x[i].dmin);
        unpack_scales_mins(x[i].scales, utmp);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m256i scales  = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(mins_and_scales));

        float d[NR];
        for (int r = 0; r < NR; ++r) {
            d[r] = y[r][i].d * dw;
            const float dmin = -y[r][i].d * dmw;
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[r][i].bsums));
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                               _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins128, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[r] += dmin * (float)_mm_extract_epi32(hsum, 0);
        }

        const uint8_t* q5 = x[i].qs;
        const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qh));
        __m256i hmask = mone;
        int bit = 0;
        __m256i sumi[NR];
        for (int r = 0; r < NR; ++r) sumi[r] = _mm256_setzero_si256();
        for (int j = 0; j < kQK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q5)); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(
                _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit++)), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(
                _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit++)), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            for (int r = 0; r < NR; ++r) {
                const int8_t* q8 = y[r][i].qs + j * 64;
                const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                const __m256i p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
                const __m256i p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_1));
            }
        }
        for (int r = 0; r < NR; ++r)
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d[r]), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
    }
    for (int r = 0; r < NR; ++r) out[r] = hsum_float_8(acc[r]) + summs[r];
}

template <int NR>
void kernel_q6_K(const void* vw, const block_q8_K* const* y, int64_t nb, float* out) {
    const block_q6_K* x = static_cast<const block_q6_K*>(vw);
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc[NR];
    for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_ps();

    for (int64_t i = 0; i < nb; ++i) {
        const float dw = f16s(x[i].d);
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;

        const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].scales));
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        __m256i q8sclsub[NR];
        __m256i sumi[NR];
        for (int r = 0; r < NR; ++r) {
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[r][i].bsums));
            q8sclsub[r] = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            sumi[r] = _mm256_setzero_si256();
        }

        int is = 0;
        for (int j = 0; j < kQK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4)); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4)); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh)); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i sc0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 0)));
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 1)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 2)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 3)));
            is += 4;

            for (int r = 0; r < NR; ++r) {
                const int8_t* q8 = y[r][i].qs + j * 128;
                const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 96));
                const __m256i p16_0 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q4_0, q8_0));
                const __m256i p16_1 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q4_1, q8_1));
                const __m256i p16_2 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q4_2, q8_2));
                const __m256i p16_3 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q4_3, q8_3));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_1));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_2, p16_3));
            }
        }
        for (int r = 0; r < NR; ++r) {
            const float d = y[r][i].d * dw;
            const __m256i s = _mm256_sub_epi32(sumi[r], q8sclsub[r]);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = hsum_float_8(acc[r]);
}

#else  // !LLM_KQ_AVX2 — scalar reference per row (still bit-consistent batch vs single)

template <int NR>
void kernel_q4_K(const void* w, const block_q8_K* const* y, int64_t nb, float* out) {
    for (int r = 0; r < NR; ++r) out[r] = vec_dot_q4_K_q8_K_ref(nb * kQK_K, w, y[r]);
}
template <int NR>
void kernel_q5_K(const void* w, const block_q8_K* const* y, int64_t nb, float* out) {
    for (int r = 0; r < NR; ++r) out[r] = vec_dot_q5_K_q8_K_ref(nb * kQK_K, w, y[r]);
}
template <int NR>
void kernel_q6_K(const void* w, const block_q8_K* const* y, int64_t nb, float* out) {
    for (int r = 0; r < NR; ++r) out[r] = vec_dot_q6_K_q8_K_ref(nb * kQK_K, w, y[r]);
}

#endif

using KernelFn = void (*)(const void*, const block_q8_K* const*, int64_t, float*);

constexpr int kNR = 4;   // activation rows per weight-block unpack (prefill)

struct Kernels { KernelFn one; KernelFn many; };

Kernels kernels_for(DType t) {
    switch (t) {
        case DType::Q4_K: return {kernel_q4_K<1>, kernel_q4_K<kNR>};
        case DType::Q5_K: return {kernel_q5_K<1>, kernel_q5_K<kNR>};
        case DType::Q6_K: return {kernel_q6_K<1>, kernel_q6_K<kNR>};
        default: throw Error(std::string("matmul_kq_q8K: unsupported type ") + dtype_name(t));
    }
}

} // namespace

float vec_dot_q4_K_q8_K(int64_t n, const void* w, const block_q8_K* y) {
    float r; kernel_q4_K<1>(w, &y, n / kQK_K, &r); return r;
}
float vec_dot_q5_K_q8_K(int64_t n, const void* w, const block_q8_K* y) {
    float r; kernel_q5_K<1>(w, &y, n / kQK_K, &r); return r;
}
float vec_dot_q6_K_q8_K(int64_t n, const void* w, const block_q8_K* y) {
    float r; kernel_q6_K<1>(w, &y, n / kQK_K, &r); return r;
}

// ---- matmuls --------------------------------------------------------------
void matmul_kq_q8K(float* y, const void* W, DType t, const float* x,
                   int64_t n_out, int64_t n_in, ThreadPool* pool) {
    LLM_CHECK(kquant_supported(t, n_in), "matmul_kq_q8K: unsupported type or n_in % 256 != 0");
    const KernelFn k1 = kernels_for(t).one;
    const int64_t nb = n_in / kQK_K;
    const int64_t row_bytes = type_nbytes(t, n_in);
    const uint8_t* base = static_cast<const uint8_t*>(W);

    std::vector<block_q8_K> xq(nb);
    quantize_row_q8_K(x, xq.data(), n_in);
    const block_q8_K* yq = xq.data();

    auto body = [&](int, int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) k1(base + o * row_bytes, &yq, nb, y + o);
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

void matmul_kq_q8K_batch(float* Y, const void* W, DType t, const float* X,
                         int64_t m, int64_t n_out, int64_t n_in, ThreadPool* pool) {
    LLM_CHECK(kquant_supported(t, n_in), "matmul_kq_q8K_batch: unsupported type or n_in % 256 != 0");
    if (m <= 0) return;
    if (m == 1) { matmul_kq_q8K(Y, W, t, X, n_out, n_in, pool); return; }
    const Kernels ks = kernels_for(t);
    const int64_t nb = n_in / kQK_K;
    const int64_t row_bytes = type_nbytes(t, n_in);
    const uint8_t* base = static_cast<const uint8_t*>(W);
    const bool par = pool && pool->size() > 1;

    // Quantize all activation rows once (parallel over rows when worthwhile).
    std::vector<block_q8_K> xq(m * nb);
    auto qbody = [&](int, int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) quantize_row_q8_K(X + r * n_in, xq.data() + r * nb, n_in);
    };
    if (par && m >= 8) pool->parallel_for(m, qbody);
    else qbody(0, 0, m);

    // Activation tile sized to sit in L2 (~256 KB) while a chunk of weight rows
    // streams past it; each weight row (<= ~10 KB) stays in L1 across the tile.
    const int64_t act_row_bytes = nb * (int64_t)sizeof(block_q8_K);
    int64_t mt = std::max<int64_t>(kNR, (256 * 1024) / act_row_bytes);
    mt = std::max<int64_t>(kNR, mt / kNR * kNR);

    auto body = [&](int, int64_t begin, int64_t end) {
        for (int64_t r0 = 0; r0 < m; r0 += mt) {
            const int64_t r1 = std::min(m, r0 + mt);
            for (int64_t o = begin; o < end; ++o) {
                const uint8_t* w = base + o * row_bytes;
                int64_t r = r0;
                for (; r + kNR <= r1; r += kNR) {
                    const block_q8_K* ys[kNR];
                    for (int k = 0; k < kNR; ++k) ys[k] = xq.data() + (r + k) * nb;
                    float out[kNR];
                    ks.many(w, ys, nb, out);
                    for (int k = 0; k < kNR; ++k) Y[(r + k) * n_out + o] = out[k];
                }
                for (; r < r1; ++r) {
                    const block_q8_K* ys = xq.data() + r * nb;
                    ks.one(w, &ys, nb, Y + r * n_out + o);
                }
            }
        }
    };
    if (par && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
}

} // namespace llm
