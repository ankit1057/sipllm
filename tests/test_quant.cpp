// test_quant.cpp — Task 9 verification.
//
// Strategy:
//  * fp16 conversion: exact known bit patterns + round-trip.
//  * Q8_0 / Q4_0: quantize a random vector, dequantize, bound the error by the
//    format's theoretical step size — proves both directions are consistent.
//  * Q4_K / Q6_K: hand-assemble a block with chosen scale/quant bytes and check
//    the first outputs against the formula computed independently here. This is
//    the unit-level guard; end-to-end correctness is proven when a real GGUF
//    produces coherent text.
//  * matmul_quant == dequantize-then-matmul.
#include "llm/quant.h"
#include "llm/ops.h"
#include "llm/dtype.h"
#include "tests/test_util.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace llm;

TEST(fp16_known_values) {
    APPROX(fp16_to_fp32(0x0000), 0.0, 0);       // +0
    APPROX(fp16_to_fp32(0x3C00), 1.0, 0);       // 1.0
    APPROX(fp16_to_fp32(0x4000), 2.0, 0);       // 2.0
    APPROX(fp16_to_fp32(0xC000), -2.0, 0);      // -2.0
    APPROX(fp16_to_fp32(0x3555), 0.333251953125, 1e-6);  // ~1/3
    CHECK(std::isinf(fp16_to_fp32(0x7C00)));    // +inf
}

TEST(fp16_roundtrip) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-10.f, 10.f);
    for (int i = 0; i < 2000; ++i) {
        float f = d(rng);
        float back = fp16_to_fp32(fp32_to_fp16(f));
        // fp16 has ~3-4 significant digits; relative error under 2^-10.
        APPROX(back, f, 1e-2 + std::fabs(f) * 1e-3);
    }
}

TEST(q8_0_quantize_dequantize) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-3.f, 3.f);
    const int n = 256;
    std::vector<float> x(n), y(n);
    for (auto& v : x) v = d(rng);
    std::vector<uint8_t> q(type_nbytes(DType::Q8_0, n));
    quantize_q8_0(x.data(), q.data(), n);
    dequantize_row(DType::Q8_0, q.data(), y.data(), n);
    // Step size ~ amax/127; per-block amax <= 3, so error <= ~0.024.
    for (int i = 0; i < n; ++i) APPROX(y[i], x[i], 0.03);
}

TEST(q4_0_quantize_dequantize) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    const int n = 256;
    std::vector<float> x(n), y(n);
    for (auto& v : x) v = d(rng);
    std::vector<uint8_t> q(type_nbytes(DType::Q4_0, n));
    quantize_q4_0(x.data(), q.data(), n);
    dequantize_row(DType::Q4_0, q.data(), y.data(), n);
    // 4-bit: 16 levels over [-amax, amax]; coarse. Bound generously but finite.
    double mse = 0; for (int i = 0; i < n; ++i) mse += (y[i]-x[i])*(y[i]-x[i]);
    mse /= n;
    CHECK_MSG(mse < 0.05, "q4_0 mse too high: " + std::to_string(mse));
}

TEST(q4_K_hand_block) {
    // Build one Q4_K block (256 elems) with dmin=0 so outputs are d*sc*(q&0xF).
    // scales[0] encodes sub-block-0 scale=5 (6-bit) via get_scale_min_k4(0).
    // With d chosen, first 32 outputs = d*5*nibble.
    std::vector<uint8_t> blk(144, 0);
    float d = 0.5f, dmin = 0.f;
    uint16_t dh = fp32_to_fp16(d), mh = fp32_to_fp16(dmin);
    std::memcpy(&blk[0], &dh, 2);
    std::memcpy(&blk[2], &mh, 2);
    // scales region (12 bytes) at offset 4. get_scale_min_k4(0): d = q[0]&63.
    blk[4] = 5;         // sub-block 0 scale = 5, min index 0
    // qs at offset 16: set first byte low nibble = 3, high nibble = 7.
    blk[16] = (7 << 4) | 3;
    std::vector<float> y(256);
    dequantize_row(DType::Q4_K, blk.data(), y.data(), 256);
    // y[0] uses q[0]&0xF = 3 -> d*5*3 = 7.5
    APPROX(y[0], 0.5 * 5 * 3, 1e-3);
    // y[32] uses q[0]>>4 = 7 with sub-block-1 scale (=0 here) -> 0.
    APPROX(y[32], 0.0, 1e-3);
}

TEST(q6_K_hand_block) {
    // Q6_K block: 128B ql, 64B qh, 16B int8 scales, fp16 d.
    std::vector<uint8_t> blk(210, 0);
    float d = 0.25f;
    // ql[0] low nibble = 5, qh[0] low 2 bits = 1 -> q = (5 | (1<<4)) - 32 = 21-32 = -11
    blk[0] = 5;
    blk[128] = 1;                 // qh[0] bits0-1 = 1
    int8_t* sc = reinterpret_cast<int8_t*>(&blk[192]);
    sc[0] = 4;                    // scales[is=0] for l=0
    uint16_t dh = fp32_to_fp16(d);
    std::memcpy(&blk[208], &dh, 2);
    std::vector<float> y(256);
    dequantize_row(DType::Q6_K, blk.data(), y.data(), 256);
    // y[0] = d * sc[0] * ((5 | 16) - 32) = 0.25 * 4 * (-11) = -11.0
    APPROX(y[0], 0.25 * 4 * (-11.0), 1e-3);
}

TEST(matmul_quant_matches_dequant_then_matmul) {
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 40, n_in = 128;   // n_in multiple of 32
    std::vector<float> Wf(n_out * n_in), x(n_in), y1(n_out), y2(n_out);
    for (auto& v : Wf) v = d(rng);
    for (auto& v : x) v = d(rng);

    // Quantize W row-by-row to Q8_0.
    std::vector<uint8_t> Wq((size_t)n_out * type_nbytes(DType::Q8_0, n_in));
    int64_t rb = type_nbytes(DType::Q8_0, n_in);
    std::vector<float> deq(n_out * n_in);
    for (int o = 0; o < n_out; ++o) {
        quantize_q8_0(Wf.data() + o * n_in, Wq.data() + o * rb, n_in);
        dequantize_row(DType::Q8_0, Wq.data() + o * rb, deq.data() + o * n_in, n_in);
    }
    matmul(y1.data(), deq.data(), x.data(), n_out, n_in);          // dequant then matmul
    matmul_quant(y2.data(), Wq.data(), DType::Q8_0, x.data(), n_out, n_in); // fused
    for (int o = 0; o < n_out; ++o) APPROX(y2[o], y1[o], 1e-4);
}

// Guards the NEON Q4_K/Q6_K dequant paths against the scalar ggml spec: random
// blocks dequantized via dequantize_row (NEON when built for ARM) must match the
// reference formula recomputed here. fp16 scales are set to controlled values so
// no inf/nan enters; quant/scale bytes are random for lane/mask coverage.
TEST(kquant_dequant_matches_scalar_spec) {
    std::mt19937 rng(1234);
    auto rb = [&]{ return (uint8_t)std::uniform_int_distribution<int>(0, 255)(rng); };
    std::uniform_real_distribution<float> rf(-1.f, 1.f);
    auto gsmk4 = [](int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
        if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
        else { d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
               m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
    };
    const int K = 256;
    for (int t = 0; t < 4; ++t) {
        // ---- Q4_K ----
        {
            std::vector<uint8_t> blk(144);
            for (auto& b : blk) b = rb();
            uint16_t dh = fp32_to_fp16(rf(rng)), mh = fp32_to_fp16(std::fabs(rf(rng)));
            std::memcpy(&blk[0], &dh, 2); std::memcpy(&blk[2], &mh, 2);
            float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(mh);
            const uint8_t* scales = blk.data() + 4; const uint8_t* q = blk.data() + 16;
            std::vector<float> ref(K); float* yy = ref.data(); int is = 0; uint8_t sc, m;
            for (int j = 0; j < K; j += 64) {
                gsmk4(is + 0, scales, sc, m); float d1 = d * sc, m1 = dmin * m;
                gsmk4(is + 1, scales, sc, m); float d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32; ++l) *yy++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *yy++ = d2 * (q[l] >>  4) - m2;
                q += 32; is += 2;
            }
            std::vector<float> got(K);
            dequantize_row(DType::Q4_K, blk.data(), got.data(), K);
            for (int i = 0; i < K; ++i) APPROX(got[i], ref[i], 1e-2);
        }
        // ---- Q6_K ----
        {
            std::vector<uint8_t> blk(210);
            for (auto& b : blk) b = rb();
            uint16_t dh = fp32_to_fp16(rf(rng));
            std::memcpy(&blk[208], &dh, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t* ql = blk.data(); const uint8_t* qh = blk.data() + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk.data() + 192);
            std::vector<float> ref(K); float* yy = ref.data();
            for (int k = 0; k < K; k += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    yy[l +  0] = d * sc[is + 0] * q1; yy[l + 32] = d * sc[is + 2] * q2;
                    yy[l + 64] = d * sc[is + 4] * q3; yy[l + 96] = d * sc[is + 6] * q4;
                }
                yy += 128; ql += 64; qh += 32; sc += 8;
            }
            std::vector<float> got(K);
            dequantize_row(DType::Q6_K, blk.data(), got.data(), K);
            for (int i = 0; i < K; ++i) APPROX(got[i], ref[i], 1e-2);
        }
    }
}

TEST(iq4_nl_hand_block) {
    // One IQ4_NL block (32 elems): fp16 d, then 16 nibble-pairs indexing the
    // fixed codebook. kvalues[0]=-127, kvalues[3]=-65, kvalues[8]=1.
    std::vector<uint8_t> blk(18, 0);
    float d = 0.5f;
    uint16_t dh = fp32_to_fp16(d);
    std::memcpy(&blk[0], &dh, 2);
    blk[2] = (8 << 4) | 3;          // qs[0]: low nibble 3, high nibble 8
    std::vector<float> y(32);
    dequantize_row(DType::IQ4_NL, blk.data(), y.data(), 32);
    APPROX(y[0],  0.5 * -65.0,  1e-3);   // low nibble -> position 0
    APPROX(y[16], 0.5 *   1.0,  1e-3);   // high nibble -> position 16
    APPROX(y[1],  0.5 * -127.0, 1e-3);   // untouched -> kvalues[0]
    APPROX(y[17], 0.5 * -127.0, 1e-3);
}

TEST(matmul_quant_iq4_nl_matches_dequant) {
    // No reference IQ4_NL quantizer exists; synthesize valid block bytes (fp16
    // scale + random nibbles), then confirm the fused matmul equals
    // dequantize-then-matmul for the same bytes.
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> nib(0, 255);
    const int n_out = 20, n_in = 64;                 // n_in multiple of 32
    const int64_t rb = type_nbytes(DType::IQ4_NL, n_in);
    std::vector<uint8_t> Wq((size_t)n_out * rb);
    for (int o = 0; o < n_out; ++o)
        for (int blk = 0; blk < n_in / 32; ++blk) {
            uint8_t* p = Wq.data() + o * rb + blk * 18;
            uint16_t dh = fp32_to_fp16(0.1f + 0.01f * blk);
            std::memcpy(p, &dh, 2);
            for (int j = 0; j < 16; ++j) p[2 + j] = (uint8_t)nib(rng);
        }
    std::vector<float> x(n_in), deq(n_out * n_in), y1(n_out), y2(n_out);
    std::uniform_real_distribution<float> rf(-1.f, 1.f);
    for (auto& v : x) v = rf(rng);
    for (int o = 0; o < n_out; ++o)
        dequantize_row(DType::IQ4_NL, Wq.data() + o * rb, deq.data() + o * n_in, n_in);
    matmul(y1.data(), deq.data(), x.data(), n_out, n_in);
    matmul_quant(y2.data(), Wq.data(), DType::IQ4_NL, x.data(), n_out, n_in);
    for (int o = 0; o < n_out; ++o) APPROX(y2[o], y1[o], 1e-4);
}

int main() {
    printf("== test_quant ==\n");
    return llmtest::run_all();
}
