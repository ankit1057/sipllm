// quant.cpp — dequantization + reference quantization (see quant.h).
//
// Every block layout and formula here matches ggml-quants.c so the engine can
// consume unmodified GGUF files. QK_K = 256 for K-quants; the legacy quants use
// 32-element blocks.
#include "llm/quant.h"
#include "llm/common.h"
#include "llm/simd.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace llm {

// ---- fp16 <-> fp32 --------------------------------------------------------
float fp16_to_fp32(uint16_t h) {
    // Full IEEE-754 half decode incl. subnormals/inf/nan.
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                     // +/- zero
        } else {
            // subnormal: normalize
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);   // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;         // too small -> zero
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = (mant >> shift);
        // round to nearest even
        if ((mant >> (shift - 1)) & 1) half += 1;
        return (uint16_t)(sign | half);
    } else if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00);             // overflow -> inf
    }
    uint16_t half = (uint16_t)(sign | (exp << 10) | (mant >> 13));
    if (mant & 0x1000) half += 1;                     // round to nearest even
    return half;
}

float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
}

// ---- legacy 32-block quants ----------------------------------------------
namespace {
constexpr int QK = 32;      // legacy block
constexpr int QK_K = 256;   // K-quant block

// scale/min unpacking shared by Q4_K and Q5_K (6-bit packed into 12 bytes).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

void dequant_q8_0(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        uint16_t dh; std::memcpy(&dh, p, 2);
        float d = fp16_to_fp32(dh);
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
        for (int i = 0; i < QK; ++i) y[b * QK + i] = d * qs[i];
        p += 34;
    }
}

void dequant_q4_0(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        uint16_t dh; std::memcpy(&dh, p, 2);
        float d = fp16_to_fp32(dh);
        const uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j) {
            int x0 = (qs[j] & 0x0F) - 8;
            int x1 = (qs[j] >> 4) - 8;
            y[b * QK + j]      = x0 * d;
            y[b * QK + j + 16] = x1 * d;
        }
        p += 18;
    }
}

// IQ4_NL: non-linear 4-bit. Per 32-element block: fp16 scale d + 16 bytes of
// 4-bit indices into a fixed 16-entry non-uniform codebook. Nibble j maps to
// output j (low) and j+16 (high) — same interleave as ggml's IQ4_NL.
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
void dequant_iq4_nl(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        uint16_t dh; std::memcpy(&dh, p, 2);
        float d = fp16_to_fp32(dh);
        const uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j) {
            y[b * QK + j]      = d * kvalues_iq4nl[qs[j] & 0x0F];
            y[b * QK + j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
        p += 18;
    }
}

void dequant_q4_1(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        float d = fp16_to_fp32(dh), m = fp16_to_fp32(mh);
        const uint8_t* qs = p + 4;
        for (int j = 0; j < 16; ++j) {
            y[b * QK + j]      = (qs[j] & 0x0F) * d + m;
            y[b * QK + j + 16] = (qs[j] >> 4)   * d + m;
        }
        p += 20;
    }
}

void dequant_q5_0(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        uint16_t dh; std::memcpy(&dh, p, 2);
        float d = fp16_to_fp32(dh);
        uint32_t qh; std::memcpy(&qh, p + 2, 4);
        const uint8_t* qs = p + 6;
        for (int j = 0; j < 16; ++j) {
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = ((qs[j] & 0x0F) | xh0) - 16;
            int x1 = ((qs[j] >>   4) | xh1) - 16;
            y[b * QK + j]      = x0 * d;
            y[b * QK + j + 16] = x1 * d;
        }
        p += 22;
    }
}

void dequant_q5_1(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK; ++b) {
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        float d = fp16_to_fp32(dh), m = fp16_to_fp32(mh);
        uint32_t qh; std::memcpy(&qh, p + 4, 4);
        const uint8_t* qs = p + 8;
        for (int j = 0; j < 16; ++j) {
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = (qs[j] & 0x0F) | xh0;
            int x1 = (qs[j] >>   4) | xh1;
            y[b * QK + j]      = x0 * d + m;
            y[b * QK + j + 16] = x1 * d + m;
        }
        p += 24;
    }
}

// ---- K-quants (256-block) -------------------------------------------------
void dequant_q4_K(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; ++b) {
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(mh);
        const uint8_t* scales = p + 4;         // 12 bytes
        const uint8_t* q = p + 16;             // 128 bytes
        float* yy = y + b * QK_K;
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *yy++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *yy++ = d2 * (q[l] >>  4) - m2;
            q += 32; is += 2;
        }
        p += 144;
    }
}

void dequant_q5_K(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; ++b) {
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(mh);
        const uint8_t* scales = p + 4;   // 12
        const uint8_t* qh = p + 16;      // 32
        const uint8_t* ql = p + 48;      // 128
        float* yy = y + b * QK_K;
        int is = 0; uint8_t sc, m, u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l)
                *yy++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *yy++ = d2 * ((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
        p += 176;
    }
}

void dequant_q6_K(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; ++b) {
        const uint8_t* ql = p;             // 128
        const uint8_t* qh = p + 128;       // 64
        const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192); // 16
        uint16_t dh; std::memcpy(&dh, p + 208, 2);
        float d = fp16_to_fp32(dh);
        float* yy = y + b * QK_K;
        for (int k = 0; k < QK_K; k += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                yy[l +  0] = d * sc[is + 0] * q1;
                yy[l + 32] = d * sc[is + 2] * q2;
                yy[l + 64] = d * sc[is + 4] * q3;
                yy[l + 96] = d * sc[is + 6] * q4;
            }
            yy += 128; ql += 64; qh += 32; sc += 8;
        }
        p += 210;
    }
}

void dequant_q2_K(const uint8_t* p, float* y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; ++b) {
        const uint8_t* scales = p;         // 16
        const uint8_t* q = p + 16;         // 64
        uint16_t dh, mh; std::memcpy(&dh, p + 80, 2); std::memcpy(&mh, p + 82, 2);
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(mh);
        float* yy = y + b * QK_K;
        int is = 0;
        for (int k = 0; k < QK_K; k += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t s = scales[is++];
                float dl = d * (s & 0xF), ml = dmin * (s >> 4);
                for (int l = 0; l < 16; ++l) *yy++ = dl * ((q[l] >> shift) & 3) - ml;
                s = scales[is++];
                dl = d * (s & 0xF); ml = dmin * (s >> 4);
                for (int l = 0; l < 16; ++l) *yy++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
        p += 84;
    }
}

void dequant_q3_K(const uint8_t* p, float* y, int64_t n) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    for (int64_t b = 0; b < n / QK_K; ++b) {
        const uint8_t* hm = p;             // hmask 32
        const uint8_t* q = p + 32;         // qs 64
        uint16_t dh; std::memcpy(&dh, p + 108, 2);
        float d_all = fp16_to_fp32(dh);
        uint32_t aux[4];
        std::memcpy(aux, p + 96, 12);      // scales 12
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        float* yy = y + b * QK_K;
        uint8_t m = 1; int is = 0;
        const uint8_t* qq = q;
        for (int k = 0; k < QK_K; k += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *yy++ = dl * ((int)((qq[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *yy++ = dl * ((int)((qq[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2; m <<= 1;
            }
            qq += 32;
        }
        p += 110;
    }
}

void dequant_f16(const uint8_t* p, float* y, int64_t n) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    for (int64_t i = 0; i < n; ++i) y[i] = fp16_to_fp32(h[i]);
}
void dequant_bf16(const uint8_t* p, float* y, int64_t n) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    for (int64_t i = 0; i < n; ++i) y[i] = bf16_to_fp32(h[i]);
}

} // namespace

// ---- public dequantize dispatch ------------------------------------------
void dequantize_row(DType t, const void* src, float* dst, int64_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    switch (t) {
        case DType::F32:  std::memcpy(dst, p, n * 4); break;
        case DType::F16:  dequant_f16(p, dst, n); break;
        case DType::BF16: dequant_bf16(p, dst, n); break;
        case DType::Q8_0: dequant_q8_0(p, dst, n); break;
        case DType::Q4_0: dequant_q4_0(p, dst, n); break;
        case DType::Q4_1: dequant_q4_1(p, dst, n); break;
        case DType::Q5_0: dequant_q5_0(p, dst, n); break;
        case DType::Q5_1: dequant_q5_1(p, dst, n); break;
        case DType::Q2_K: dequant_q2_K(p, dst, n); break;
        case DType::Q3_K: dequant_q3_K(p, dst, n); break;
        case DType::Q4_K: dequant_q4_K(p, dst, n); break;
        case DType::Q5_K: dequant_q5_K(p, dst, n); break;
        case DType::Q6_K: dequant_q6_K(p, dst, n); break;
        case DType::IQ4_NL: dequant_iq4_nl(p, dst, n); break;
        default:
            throw Error(std::string("dequantize_row: unsupported type ") + dtype_name(t));
    }
}

// ---- fused quantized matmul ----------------------------------------------
void matmul_quant(float* y, const void* W, DType t, const float* x,
                  int64_t n_out, int64_t n_in, ThreadPool* pool) {
    const int64_t row_bytes = type_nbytes(t, n_in);
    const uint8_t* base = static_cast<const uint8_t*>(W);

    auto body = [&](int, int64_t begin, int64_t end) {
        // Per-worker scratch: one dequantized row of fp32. Sized to n_in.
        std::vector<float> row(n_in);
        for (int64_t o = begin; o < end; ++o) {
            dequantize_row(t, base + o * row_bytes, row.data(), n_in);
            y[o] = dot_f32(row.data(), x, n_in);
        }
    };
    if (pool && pool->size() > 1 && n_out >= 32)
        pool->parallel_for(n_out, body);
    else
        body(0, 0, n_out);
}

// ---- reference quantizers -------------------------------------------------
void quantize_q8_0(const float* src, void* dst, int64_t n) {
    LLM_CHECK(n % QK == 0, "quantize_q8_0: n must be multiple of 32");
    uint8_t* p = static_cast<uint8_t*>(dst);
    for (int64_t b = 0; b < n / QK; ++b) {
        const float* xb = src + b * QK;
        float amax = 0.f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(xb[i]));
        float d = amax / 127.f;
        float id = d ? 1.f / d : 0.f;
        uint16_t dh = fp32_to_fp16(d);
        std::memcpy(p, &dh, 2);
        int8_t* qs = reinterpret_cast<int8_t*>(p + 2);
        for (int i = 0; i < QK; ++i) {
            int q = (int)std::lround(xb[i] * id);
            qs[i] = (int8_t)std::max(-127, std::min(127, q));
        }
        p += 34;
    }
}

void quantize_q4_0(const float* src, void* dst, int64_t n) {
    LLM_CHECK(n % QK == 0, "quantize_q4_0: n must be multiple of 32");
    uint8_t* p = static_cast<uint8_t*>(dst);
    for (int64_t b = 0; b < n / QK; ++b) {
        const float* xb = src + b * QK;
        // Symmetric max-abs scale, 4-bit signed range [-8, 7] centered at 8.
        float amax = 0.f, max = 0.f;
        for (int i = 0; i < QK; ++i) {
            if (std::fabs(xb[i]) > amax) { amax = std::fabs(xb[i]); max = xb[i]; }
        }
        float d = max / -8.f;
        float id = d ? 1.f / d : 0.f;
        uint16_t dh = fp32_to_fp16(d);
        std::memcpy(p, &dh, 2);
        uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j) {
            int q0 = std::min(15, (int)(xb[j]      * id + 8.5f));
            int q1 = std::min(15, (int)(xb[j + 16] * id + 8.5f));
            qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
        p += 18;
    }
}

} // namespace llm
