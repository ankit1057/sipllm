// test_neon.cpp — Task 10 verification.
//  * fp16 bulk convert matches the scalar reference exactly.
//  * int8 SDOT Q8_0 matmul approximates the fp32 result within the error the
//    activation-quantization budget allows (relative, per output).
#include "llm/neon.h"
#include "llm/ops.h"
#include "llm/quant.h"
#include "tests/test_util.h"

#include <cmath>
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

int main() {
    printf("== test_neon == (dotprod=%d)\n", (int)neon_dotprod_available());
    return llmtest::run_all();
}
