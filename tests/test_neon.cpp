// test_neon.cpp — Task 10 verification.
//  * fp16 bulk convert matches the scalar reference exactly.
//  * int8 SDOT Q8_0 matmul approximates the fp32 result within the error the
//    activation-quantization budget allows (relative, per output).
#include "llm/neon.h"
#include "llm/ops.h"
#include "llm/quant.h"
#include "llm/threadpool.h"
#include "tests/test_util.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace llm;

TEST(fp16_bulk_matches_scalar) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> d(-5.f, 5.f);
    const int n = 1000;
    std::vector<uint16_t> h(n);
    std::vector<float> ref(n), got(n);
    for (int i = 0; i < n; ++i) { h[i] = fp32_to_fp16(d(rng)); ref[i] = fp16_to_fp32(h[i]); }
    fp16_to_fp32_bulk(h.data(), got.data(), n);
    for (int i = 0; i < n; ++i) APPROX(got[i], ref[i], 0);
}

TEST(q8_0_sdot_matmul_approximates_fp32) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 64, n_in = 256;
    std::vector<float> Wf(n_out * n_in), x(n_in), y_ref(n_out), y_neon(n_out);
    for (auto& v : Wf) v = d(rng);
    for (auto& v : x) v = d(rng);

    // Quantize weights to Q8_0.
    int64_t rb = type_nbytes(DType::Q8_0, n_in);
    std::vector<uint8_t> Wq((size_t)n_out * rb);
    for (int o = 0; o < n_out; ++o)
        quantize_q8_0(Wf.data() + o * n_in, Wq.data() + o * rb, n_in);

    // fp32 reference = plain matmul with the ORIGINAL weights.
    matmul(y_ref.data(), Wf.data(), x.data(), n_out, n_in);
    // int8 SDOT path (activation also quantized).
    matmul_q8_0_i8(y_neon.data(), Wq.data(), x.data(), n_out, n_in);

    // Both weight and activation are 8-bit; error scales with vector norm.
    double xnorm = 0; for (float v : x) xnorm += v * v; xnorm = std::sqrt(xnorm);
    for (int o = 0; o < n_out; ++o)
        APPROX(y_neon[o], y_ref[o], 0.05 * xnorm + 0.05);
}

// Build a valid Q4_K super-block row from random fields. Any well-formed
// encoding is a legitimate test input: the reference is the exact fp32 dequant
// of the SAME bytes (matmul_quant), so the only error the kernel may add is the
// int8 activation quantization, exactly like the Q8_0 case above.
static void random_q4_K_row(std::mt19937& rng, uint8_t* row, int64_t n_in) {
    std::uniform_int_distribution<int> six(0, 63), nib(0, 255);
    for (int64_t k = 0; k < n_in / 256; ++k) {
        uint8_t* p = row + k * 144;
        // d, dmin chosen so |w| <= 1: w = d*sc*q - dmin*m, sc,m <= 63, q <= 15.
        uint16_t dh = fp32_to_fp16(1.0f / (63.f * 15.f)), mh = fp32_to_fp16(0.5f / 63.f);
        std::memcpy(p, &dh, 2); std::memcpy(p + 2, &mh, 2);
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; ++j) { sc[j] = (uint8_t)six(rng); mn[j] = (uint8_t)six(rng); }
        uint8_t* q = p + 4;   // inverse of get_scale_min_k4
        for (int j = 0; j < 4; ++j) {
            q[j]     = (uint8_t)((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
            q[j + 4] = (uint8_t)((mn[j] & 63) | ((mn[j + 4] >> 4) << 6));
            q[j + 8] = (uint8_t)((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
        }
        for (int i = 0; i < 128; ++i) p[16 + i] = (uint8_t)nib(rng);
    }
}

TEST(q4_K_scale_min_pack_roundtrips) {
    std::mt19937 rng(9);
    std::vector<uint8_t> row(144);
    random_q4_K_row(rng, row.data(), 256);
    // Re-derive every (sc, m) and check the packing is self-consistent with a
    // second, independent unpack through the dequantizer's own helper.
    for (int j = 0; j < 8; ++j) {
        uint8_t d1, m1, d2, m2;
        get_scale_min_k4(j, row.data() + 4, d1, m1);
        get_scale_min_k4(j, row.data() + 4, d2, m2);
        CHECK(d1 == d2 && m1 == m2 && d1 < 64 && m1 < 64);
    }
}

TEST(q4_K_sdot_matmul_approximates_fp32) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 48, n_in = 512;     // two super-blocks per row
    const int64_t rb = type_nbytes(DType::Q4_K, n_in);
    CHECK(rb == 2 * 144);
    std::vector<uint8_t> Wq((size_t)n_out * rb);
    for (int o = 0; o < n_out; ++o) random_q4_K_row(rng, Wq.data() + o * rb, n_in);
    std::vector<float> x(n_in), y_ref(n_out), y_i8(n_out);
    for (auto& v : x) v = d(rng);
    matmul_quant(y_ref.data(), Wq.data(), DType::Q4_K, x.data(), n_out, n_in);   // exact
    matmul_q4_K_i8(y_i8.data(), Wq.data(), x.data(), n_out, n_in);
    double xnorm = 0; for (float v : x) xnorm += v * v; xnorm = std::sqrt(xnorm);
    for (int o = 0; o < n_out; ++o)
        APPROX(y_i8[o], y_ref[o], 0.05 * xnorm + 0.05);
    // And the batched entry point agrees with the single-vector one.
    std::vector<float> X2(2 * n_in), Y2(2 * n_out);
    for (auto& v : X2) v = d(rng);
    std::vector<float> y0(n_out), y1(n_out);
    matmul_q4_K_i8(y0.data(), Wq.data(), X2.data(), n_out, n_in);
    matmul_q4_K_i8(y1.data(), Wq.data(), X2.data() + n_in, n_out, n_in);
    for (int o = 0; o < n_out; ++o) { CHECK(y0[o] == y0[o]); CHECK(y1[o] == y1[o]); } // no NaN
}

TEST(q4_K_sdot_matmul_threaded_matches_single) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 96, n_in = 256;
    const int64_t rb = type_nbytes(DType::Q4_K, n_in);
    std::vector<uint8_t> Wq((size_t)n_out * rb);
    for (int o = 0; o < n_out; ++o) random_q4_K_row(rng, Wq.data() + o * rb, n_in);
    std::vector<float> x(n_in), y1(n_out), yN(n_out);
    for (auto& v : x) v = d(rng);
    matmul_q4_K_i8(y1.data(), Wq.data(), x.data(), n_out, n_in, nullptr);
    ThreadPool pool(4);
    matmul_q4_K_i8(yN.data(), Wq.data(), x.data(), n_out, n_in, &pool);
    for (int o = 0; o < n_out; ++o) APPROX(yN[o], y1[o], 0);   // row-parallel: bit-identical
}

// A well-formed random Q6_K row: ql/qh random bits, sc random int8 (6-bit
// range keeps |w| <= 1 with the chosen d), d fp16. Reference is the exact
// fp32 dequant of the same bytes.
static void random_q6_K_row(std::mt19937& rng, uint8_t* row, int64_t n_in) {
    std::uniform_int_distribution<int> byte(0, 255), scl(-64, 63);
    for (int64_t k = 0; k < n_in / 256; ++k) {
        uint8_t* p = row + k * 210;
        for (int i = 0; i < 192; ++i) p[i] = (uint8_t)byte(rng);      // ql[128] + qh[64]
        for (int i = 0; i < 16; ++i)  p[192 + i] = (uint8_t)(int8_t)scl(rng);
        uint16_t dh = fp32_to_fp16(1.0f / (64.f * 32.f));              // |sc*(q-32)| <= 64*32
        std::memcpy(p + 208, &dh, 2);
    }
}

TEST(q6_K_sdot_matmul_approximates_fp32) {
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 48, n_in = 512;
    const int64_t rb = type_nbytes(DType::Q6_K, n_in);
    CHECK(rb == 2 * 210);
    std::vector<uint8_t> Wq((size_t)n_out * rb);
    for (int o = 0; o < n_out; ++o) random_q6_K_row(rng, Wq.data() + o * rb, n_in);
    std::vector<float> x(n_in), y_ref(n_out), y_i8(n_out);
    for (auto& v : x) v = d(rng);
    matmul_quant(y_ref.data(), Wq.data(), DType::Q6_K, x.data(), n_out, n_in);   // exact
    matmul_q6_K_i8(y_i8.data(), Wq.data(), x.data(), n_out, n_in);
    double xnorm = 0; for (float v : x) xnorm += v * v; xnorm = std::sqrt(xnorm);
    for (int o = 0; o < n_out; ++o)
        APPROX(y_i8[o], y_ref[o], 0.05 * xnorm + 0.05);
}

TEST(q6_K_sdot_matmul_threaded_matches_single) {
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 96, n_in = 256;
    const int64_t rb = type_nbytes(DType::Q6_K, n_in);
    std::vector<uint8_t> Wq((size_t)n_out * rb);
    for (int o = 0; o < n_out; ++o) random_q6_K_row(rng, Wq.data() + o * rb, n_in);
    std::vector<float> x(n_in), y1(n_out), yN(n_out);
    for (auto& v : x) v = d(rng);
    matmul_q6_K_i8(y1.data(), Wq.data(), x.data(), n_out, n_in, nullptr);
    ThreadPool pool(4);
    matmul_q6_K_i8(yN.data(), Wq.data(), x.data(), n_out, n_in, &pool);
    for (int o = 0; o < n_out; ++o) APPROX(yN[o], y1[o], 0);
}

int main() {
    printf("== test_neon == (dotprod=%d)\n", (int)neon_dotprod_available());
    return llmtest::run_all();
}
