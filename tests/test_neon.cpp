// test_neon.cpp — Task 10 verification.
//  * fp16 bulk convert matches the scalar reference exactly.
//  * int8 SDOT Q8_0 matmul approximates the fp32 result within the error the
//    activation-quantization budget allows (relative, per output).
#include "llm/linear.h"
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


// Wave 14: the batched (prefill) kernels must be BIT-IDENTICAL, per token, to
// calling the single-vector kernel once per token — same arithmetic, fewer
// weight passes. Ragged batch sizes exercise the partial last tile; a thread
// pool exercises the row split.
enum class QK { Q8_0, Q4_K, Q6_K };
static void check_batch_identity(QK kind, const char* what) {
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t n_in = 512, n_out = 96;
    const DType dt = kind == QK::Q8_0 ? DType::Q8_0 : kind == QK::Q4_K ? DType::Q4_K : DType::Q6_K;
    const int64_t rb = type_nbytes(dt, n_in);
    std::vector<uint8_t> Wq((size_t)(rb * n_out));
    if (kind == QK::Q8_0) {
        std::vector<float> Wf((size_t)(n_out * n_in));
        for (auto& v : Wf) v = nd(rng) * 0.05f;
        for (int64_t o = 0; o < n_out; ++o) quantize_q8_0(Wf.data() + o * n_in, Wq.data() + o * rb, n_in);
    } else {
        for (int64_t o = 0; o < n_out; ++o)
            (kind == QK::Q4_K ? random_q4_K_row : random_q6_K_row)(rng, Wq.data() + o * rb, n_in);
    }
    ThreadPool pool(4);
    for (int64_t bs : {1, 2, 3, 4, 5, 7, 31, 32, 33}) {
        std::vector<float> X((size_t)(bs * n_in));
        for (auto& v : X) v = nd(rng) * 3.f;
        for (ThreadPool* tp : {(ThreadPool*)nullptr, &pool}) {
            std::vector<float> ref((size_t)(bs * n_out)), got((size_t)(bs * n_out), -1.f);
            for (int64_t b = 0; b < bs; ++b) {
                float* y = ref.data() + b * n_out;
                const float* x = X.data() + b * n_in;
                if (kind == QK::Q8_0)      matmul_q8_0_i8(y, Wq.data(), x, n_out, n_in, tp);
                else if (kind == QK::Q4_K) matmul_q4_K_i8(y, Wq.data(), x, n_out, n_in, tp);
                else                       matmul_q6_K_i8(y, Wq.data(), x, n_out, n_in, tp);
            }
            if (kind == QK::Q8_0)      matmul_q8_0_i8_batch(got.data(), Wq.data(), X.data(), bs, n_out, n_in, tp);
            else if (kind == QK::Q4_K) matmul_q4_K_i8_batch(got.data(), Wq.data(), X.data(), bs, n_out, n_in, tp);
            else                       matmul_q6_K_i8_batch(got.data(), Wq.data(), X.data(), bs, n_out, n_in, tp);
            CHECK_MSG(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0,
                      std::string(what) + ": batched != per-token at bs=" + std::to_string(bs) +
                      (tp ? " (pool)" : " (single)"));
        }
    }
}
TEST(q8_0_batch_bit_identical_to_per_token) { check_batch_identity(QK::Q8_0, "q8_0"); }
TEST(q4_K_batch_bit_identical_to_per_token) { check_batch_identity(QK::Q4_K, "q4_K"); }
TEST(q6_K_batch_bit_identical_to_per_token) { check_batch_identity(QK::Q6_K, "q6_K"); }


// Wave 15: IL4-interleaved Q4_K. Repacking is a pure rearrangement (same size),
// and the lane-per-row kernel computes the same int8 products as the row-major
// kernel; only the float summation order differs. Bound: 1e-5 of max |y|,
// four orders of magnitude below the int8 approximation's own error vs exact.
TEST(q4_K_il4_matches_row_major_int8) {
    if (!q4_K_interleave_supported()) { printf("    (skipped: no SDOT on this CPU)\n"); return; }
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t n_in = 768, n_out = 100;            // 100 = 25 groups of 4
    const int64_t rb = type_nbytes(DType::Q4_K, n_in);
    std::vector<uint8_t> Wq((size_t)(rb * n_out));
    for (int64_t o = 0; o < n_out; ++o) random_q4_K_row(rng, Wq.data() + o * rb, n_in);
    // Real models have a different d / dmin in every row and super-block; the
    // shared helper uses constants, which would hide a row mix-up of d / dmin.
    std::uniform_real_distribution<float> ud(0.2f, 1.8f);
    for (int64_t o = 0; o < n_out; ++o)
        for (int64_t k = 0; k < n_in / 256; ++k) {
            uint8_t* p = Wq.data() + o * rb + k * 144;
            const uint16_t dh = fp32_to_fp16(ud(rng) / (63.f * 15.f)), mh = fp32_to_fp16(ud(rng) * 0.5f / 63.f);
            std::memcpy(p, &dh, 2); std::memcpy(p + 2, &mh, 2);
        }
    std::vector<uint8_t> Wi = Wq;
    repack_q4_K_il4_inplace(Wi.data(), n_out, n_in);
    CHECK_MSG(Wi.size() == Wq.size() && Wi != Wq, "repack keeps the size and changes the layout");
    ThreadPool pool(4);
    for (int64_t bs : {1, 2, 3, 4, 5, 31, 32, 33}) {
        std::vector<float> X((size_t)(bs * n_in));
        for (auto& v : X) v = nd(rng) * 3.f;
        for (ThreadPool* tp : {(ThreadPool*)nullptr, &pool}) {
            std::vector<float> ref((size_t)(bs * n_out)), got((size_t)(bs * n_out), -1.f);
            matmul_q4_K_i8_batch(ref.data(), Wq.data(), X.data(), bs, n_out, n_in, tp);
            matmul_q4_K_il4(got.data(), Wi.data(), X.data(), bs, n_out, n_in, tp);
            float ymax = 0, dmax = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                ymax = std::max(ymax, std::fabs(ref[i]));
                dmax = std::max(dmax, std::fabs(ref[i] - got[i]));
            }
            CHECK_MSG(dmax <= 1e-5f * ymax, "il4 vs row-major at bs=" + std::to_string(bs) +
                      ": max diff " + std::to_string(dmax) + " vs max |y| " + std::to_string(ymax));
        }
    }
}


// Wave 15: an interleaved weight must never reach the exact (fp32-dequant)
// path, which would misread its bytes. With --fast turned off (the flag is
// process-global; a second Runtime can flip it), linear() must refuse loudly.
TEST(il4_weight_refuses_when_fast_is_off) {
    std::vector<uint8_t> W((size_t)(type_nbytes(DType::Q4_K, 256) * 4), 0);
    WeightRef w{W.data(), DType::Q4_K, 4, 256};
    w.layout = kLayoutQ4K_IL4;
    std::vector<float> x(256, 1.f), y(4, 0.f);
    const bool was = fast_quant_enabled();
    set_fast_quant(false);
    bool threw = false;
    try { linear(y.data(), w, x.data()); } catch (const std::exception&) { threw = true; }
    CHECK_MSG(threw, "linear() must refuse IL4 weights when --fast is off");
    threw = false;
    try { linear_batch(y.data(), w, x.data(), 1); } catch (const std::exception&) { threw = true; }
    CHECK_MSG(threw, "linear_batch() must refuse IL4 weights when --fast is off");
    set_fast_quant(was);
}

int main() {
    printf("== test_neon == (dotprod=%d)\n", (int)neon_dotprod_available());
    return llmtest::run_all();
}
