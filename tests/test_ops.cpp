// test_ops.cpp — Phase 1 verification: tensor, matmul, vec_add, rmsnorm,
// softmax, silu. The NEON dot product is diffed against a plain scalar loop so
// the vectorized path is proven equivalent, not just "fast".
#include "llm/ops.h"
#include "llm/simd.h"
#include "llm/tensor.h"
#include "tests/test_util.h"

#include <random>
#include <vector>

using namespace llm;

TEST(tensor_alloc_and_index) {
    Tensor t({2, 3});
    CHECK(t.numel() == 6);
    CHECK(t.ndim() == 2);
    t.at(0, 0) = 1.f; t.at(1, 2) = 9.f;
    CHECK(t[0] == 1.f);
    CHECK(t[5] == 9.f);
    t.fill(2.5f);
    CHECK(t[3] == 2.5f);
    t.zero();
    CHECK(t[3] == 0.f);
}

TEST(tensor_move_semantics) {
    Tensor a({4});
    a.fill(7.f);
    float* p = a.data();
    Tensor b = std::move(a);
    CHECK(b.data() == p);       // buffer transferred, not copied
    CHECK(a.numel() == 0);      // source emptied
    CHECK(b[2] == 7.f);
}

TEST(vec_add_basic) {
    float a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40}, out[4];
    vec_add(out, a, b, 4);
    for (int i = 0; i < 4; ++i) APPROX(out[i], (i + 1) * 11.0, 1e-6);
    vec_add_inplace(a, b, 4);
    APPROX(a[3], 44.0, 1e-6);
}

TEST(dot_neon_matches_scalar) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (int n : {1, 3, 4, 7, 15, 16, 17, 63, 128, 4096, 4097}) {
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; ++i) { a[i] = d(rng); b[i] = d(rng); }
        double ref = 0.0;
        for (int i = 0; i < n; ++i) ref += (double)a[i] * b[i];
        float got = dot_f32(a.data(), b.data(), n);
        // fp32 accumulation drift grows with n; scale tolerance accordingly.
        APPROX(got, ref, 1e-3 + 1e-5 * n);
    }
}

// Equivalence of the vectorized simd.h kernels (NEON on ARM, AVX2 on x86_64,
// scalar elsewhere) against a plain scalar oracle — same guard as #4 asks for
// dot_f32, extended to axpy/mul/scale so every AVX2 lane is exercised in CI.
TEST(simd_kernels_match_scalar) {
    std::mt19937 rng(321);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (int n : {1, 3, 7, 8, 9, 16, 31, 32, 33, 129, 256, 1000}) {
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; ++i) { a[i] = d(rng); b[i] = d(rng); }
        const float alpha = 0.75f, s = -1.5f;

        // axpy: y += alpha*x
        std::vector<float> y1(b), y2(b);
        axpy_f32(y1.data(), a.data(), alpha, n);
        for (int i = 0; i < n; ++i) y2[i] += alpha * a[i];
        for (int i = 0; i < n; ++i) APPROX(y1[i], y2[i], 1e-5);

        // mul: out = a*b
        std::vector<float> m(n);
        mul_f32(m.data(), a.data(), b.data(), n);
        for (int i = 0; i < n; ++i) APPROX(m[i], a[i] * b[i], 1e-6);

        // scale: x *= s
        std::vector<float> sc(a);
        scale_f32(sc.data(), s, n);
        for (int i = 0; i < n; ++i) APPROX(sc[i], a[i] * s, 1e-6);
    }
}

TEST(matmul_identity) {
    // Identity weight matrix must reproduce the input vector.
    const int n = 5;
    std::vector<float> W(n * n, 0.f), x(n), y(n);
    for (int i = 0; i < n; ++i) { W[i * n + i] = 1.f; x[i] = i + 1; }
    matmul(y.data(), W.data(), x.data(), n, n);
    for (int i = 0; i < n; ++i) APPROX(y[i], i + 1, 1e-6);
}

TEST(matmul_known_values) {
    // W = [[1,2,3],[4,5,6]], x = [1,0,-1] -> y = [1-3, 4-6] = [-2,-2]
    float W[6] = {1, 2, 3, 4, 5, 6};
    float x[3] = {1, 0, -1};
    float y[2];
    matmul(y, W, x, 2, 3);
    APPROX(y[0], -2.0, 1e-6);
    APPROX(y[1], -2.0, 1e-6);
}

TEST(matmul_threaded_matches_serial) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int n_out = 300, n_in = 200;
    std::vector<float> W(n_out * n_in), x(n_in), y1(n_out), y2(n_out);
    for (auto& v : W) v = d(rng);
    for (auto& v : x) v = d(rng);
    matmul(y1.data(), W.data(), x.data(), n_out, n_in, nullptr);      // serial
    ThreadPool pool(4);
    matmul(y2.data(), W.data(), x.data(), n_out, n_in, &pool);        // threaded
    for (int i = 0; i < n_out; ++i) APPROX(y2[i], y1[i], 1e-4);
}

TEST(matmul_batch_matches_single) {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int m = 3, n_out = 40, n_in = 64;
    std::vector<float> W(n_out * n_in), X(m * n_in), Y(m * n_out);
    for (auto& v : W) v = d(rng);
    for (auto& v : X) v = d(rng);
    matmul_batch(Y.data(), X.data(), W.data(), m, n_out, n_in);
    for (int r = 0; r < m; ++r) {
        std::vector<float> yr(n_out);
        matmul(yr.data(), W.data(), X.data() + r * n_in, n_out, n_in);
        for (int o = 0; o < n_out; ++o) APPROX(Y[r * n_out + o], yr[o], 1e-4);
    }
}

TEST(rmsnorm_math) {
    // x = [1,2,3,4]; ms = mean(x^2) = (1+4+9+16)/4 = 7.5
    // scale = 1/sqrt(7.5 + eps); weight all ones.
    float x[4] = {1, 2, 3, 4}, w[4] = {1, 1, 1, 1}, out[4];
    rmsnorm(out, x, w, 4, 0.0f);
    double scale = 1.0 / std::sqrt(7.5);
    for (int i = 0; i < 4; ++i) APPROX(out[i], (i + 1) * scale, 1e-5);
    // Output RMS should be ~1.
    double ss = 0; for (int i = 0; i < 4; ++i) ss += out[i] * out[i];
    APPROX(std::sqrt(ss / 4), 1.0, 1e-5);
}

TEST(softmax_math) {
    float x[3] = {1, 2, 3};
    softmax(x, 3);
    double sum = x[0] + x[1] + x[2];
    APPROX(sum, 1.0, 1e-6);
    CHECK(x[2] > x[1] && x[1] > x[0]);   // monotone preserved
    // Known reference: exp offsets from max.
    double e0 = std::exp(-2.0), e1 = std::exp(-1.0), e2 = 1.0;
    double z = e0 + e1 + e2;
    APPROX(x[0], e0 / z, 1e-6);
}

TEST(softmax_large_values_stable) {
    // Would overflow without the max-subtraction trick.
    float x[3] = {1000.f, 1001.f, 1002.f};
    softmax(x, 3);
    double sum = x[0] + x[1] + x[2];
    APPROX(sum, 1.0, 1e-6);
    CHECK(!std::isnan(x[0]));
}

TEST(silu_math) {
    float x[3] = {0.f, 1.f, -1.f};
    silu_inplace(x, 3);
    APPROX(x[0], 0.0, 1e-6);                          // silu(0)=0
    APPROX(x[1], 1.0 / (1.0 + std::exp(-1.0)), 1e-6); // silu(1)
    APPROX(x[2], -1.0 / (1.0 + std::exp(1.0)), 1e-6); // silu(-1)
}

TEST(gelu_math) {
    float x[4] = {0.f, 1.f, -1.f, 50.f};
    gelu_inplace(x, 4);
    APPROX(x[0], 0.0, 1e-6);          // gelu(0)=0
    APPROX(x[1], 0.8411920, 1e-4);    // tanh-approx gelu(1)
    APPROX(x[2], -0.1588080, 1e-4);   // gelu(-1)
    APPROX(x[3], 50.0, 1e-3);         // saturates to x for large x
}

TEST(softcap_math) {
    float v[4] = {0.f, 1.f, 100.f, -100.f};
    softcap_inplace(v, 4, 5.f);
    APPROX(v[0], 0.0, 1e-6);
    APPROX(v[1], 5.0 * std::tanh(0.2), 1e-5);
    APPROX(v[2], 5.0, 1e-3);          // saturates to +cap
    APPROX(v[3], -5.0, 1e-3);         // saturates to -cap
    float w[2] = {7.f, -3.f};
    softcap_inplace(w, 2, 0.f);       // cap<=0 is a no-op
    APPROX(w[0], 7.0, 0); APPROX(w[1], -3.0, 0);
}

TEST(rmsnorm_gemma_math) {
    float x[4] = {3.f, 0.f, 0.f, 0.f};   // rms = sqrt(9/4) = 1.5
    float out[4];
    float wz[4] = {0.f, 0.f, 0.f, 0.f};
    rmsnorm_gemma(out, x, wz, 4, 0.f);
    APPROX(out[0], 2.0, 1e-4);           // 3/1.5 * (1+0)
    float w1[4] = {1.f, 1.f, 1.f, 1.f};
    rmsnorm_gemma(out, x, w1, 4, 0.f);
    APPROX(out[0], 4.0, 1e-4);           // 3/1.5 * (1+1)
}

int main() {
    printf("== test_ops ==\n");
    return llmtest::run_all();
}
