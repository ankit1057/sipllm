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

#if defined(__ARM_FEATURE_DOTPROD)
inline int32_t sdot32(const int8_t* a, const int8_t* b) {
    int8x16_t a0 = vld1q_s8(a),      b0 = vld1q_s8(b);
    int8x16_t a1 = vld1q_s8(a + 16), b1 = vld1q_s8(b + 16);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, a0, b0);
    acc = vdotq_s32(acc, a1, b1);
    return vaddvq_s32(acc);
}
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
            float acc = 0.f;
            for (int64_t b = 0; b < nb; ++b) {
                uint16_t dh; std::memcpy(&dh, row + b * 34, 2);
                const int8_t* wq = reinterpret_cast<const int8_t*>(row + b * 34 + 2);
                int32_t s = sdot32(wq, xq.data() + b * 32);
                acc += fp16_to_fp32(dh) * xs[b] * (float)s;
            }
            y[o] = acc;
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32) pool->parallel_for(n_out, body);
    else body(0, 0, n_out);
#else
    // No SDOT: defer to the exact fp32-dequant path.
    matmul_quant(y, W, DType::Q8_0, x, n_out, n_in, pool);
#endif
}

} // namespace llm
