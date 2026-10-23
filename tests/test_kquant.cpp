// test_kquant.cpp — K-quant x Q8_K integer kernels (kquant.h).
//  * quantize_row_q8_K matches ggml's Q8_K semantics (scale, round-half-even,
//    bsums, zero block).
//  * SIMD vec_dot == scalar ggml-generic port (tight: integer part is exact,
//    only fp summation order differs).
//  * vec_dot / matmul_kq vs fp32 dequant oracle within the int8-activation
//    budget (norm-relative 1e-2).
//  * batched == per-row single, bit-exact (incl. NR remainder + thread pool).
//  * linear(): fast off is byte-identical to matmul_quant; fast on routes here.
//  * microbenchmark: 15360x3840 Q4_K matmul, old (dequant+fp32) vs new, 1 & 8
//    threads, decode (m=1) and prefill (m=64).
#include "llm/kquant.h"
#include "llm/linear.h"
#include "llm/neon.h"
#include "llm/quant.h"
#include "llm/threadpool.h"
#include "tests/test_util.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace llm;

namespace {

// Synthesize a valid K-quant weight matrix [rows, n]: random payload bytes
// (cover every packing case) with sane fp16 super-block scales.
std::vector<uint8_t> make_weights(DType t, int64_t rows, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    const int64_t nb = n / kQK_K;
    const int64_t bsz = type_traits(t).type_size;
    std::vector<uint8_t> w(rows * nb * bsz);
    for (auto& b : w) b = (uint8_t)byte(rng);
    for (int64_t i = 0; i < rows * nb; ++i) {
        uint8_t* blk = w.data() + i * bsz;
        if (t == DType::Q4_K || t == DType::Q5_K) {
            uint16_t d = fp32_to_fp16(2e-3f * u(rng)), dm = fp32_to_fp16(2e-3f * u(rng));
            std::memcpy(blk, &d, 2); std::memcpy(blk + 2, &dm, 2);
        } else {  // Q6_K: d at offset 208
            uint16_t d = fp32_to_fp16(1e-4f * u(rng));
            std::memcpy(blk + 208, &d, 2);
        }
    }
    return w;
}

std::vector<float> rand_vec(int64_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

using DotFn = float (*)(int64_t, const void*, const block_q8_K*);
DotFn simd_dot(DType t) {
    return t == DType::Q4_K ? vec_dot_q4_K_q8_K : t == DType::Q5_K ? vec_dot_q5_K_q8_K : vec_dot_q6_K_q8_K;
}
DotFn ref_dot(DType t) {
    return t == DType::Q4_K ? vec_dot_q4_K_q8_K_ref
         : t == DType::Q5_K ? vec_dot_q5_K_q8_K_ref : vec_dot_q6_K_q8_K_ref;
}

const DType kTypes[] = {DType::Q4_K, DType::Q5_K, DType::Q6_K};

double norm_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double)(a[i] - b[i]) * (a[i] - b[i]);
        den += (double)b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

} // namespace

TEST(q8K_quantize_matches_ggml_semantics) {
    const int64_t n = 512;
    auto x = rand_vec(n, 11, -3.f, 3.f);
    x[300] = -4.f;                       // block 1's max is negative
    std::vector<block_q8_K> q(n / kQK_K);
    quantize_row_q8_K(x.data(), q.data(), n);
    for (int64_t b = 0; b < n / kQK_K; ++b) {
        float max = 0, amax = 0;
        for (int j = 0; j < kQK_K; ++j) {
            float v = x[b * kQK_K + j];
            if (std::fabs(v) > amax) { amax = std::fabs(v); max = v; }
        }
        APPROX(q[b].d, max / -127.f, 1e-7);
        for (int j = 0; j < kQK_K; ++j) {
            float v = x[b * kQK_K + j];
            CHECK(q[b].qs[j] >= -127 && q[b].qs[j] <= 127);
            CHECK(std::fabs(v - q[b].d * q[b].qs[j]) <= std::fabs(q[b].d) * 0.5f + 1e-6f);
        }
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int k = 0; k < 16; ++k) s += q[b].qs[g * 16 + k];
            CHECK(q[b].bsums[g] == s);
        }
    }
    CHECK(q[1].qs[300 - 256] == -127);   // the max-magnitude element always maps to -127
}

TEST(q8K_round_half_even_and_zero_block) {
    // max = -127 -> iscale = 1 exactly; halves round to even like ggml nearest_int.
    std::vector<float> x(512, 0.f);
    x[0] = -127.f;
    x[1] = 0.5f; x[2] = 1.5f; x[3] = 2.5f; x[4] = -0.5f; x[5] = -1.5f;
    std::vector<block_q8_K> q(2);
    std::memset(q.data(), 0x5A, sizeof(block_q8_K) * 2);   // poison
    quantize_row_q8_K(x.data(), q.data(), 512);
    CHECK(q[0].d == 1.f);                                  // iscale = -127/max = 1
    CHECK(q[0].qs[0] == -127);
    CHECK(q[0].qs[1] == 0);  CHECK(q[0].qs[2] == 2); CHECK(q[0].qs[3] == 2);
    CHECK(q[0].qs[4] == 0);  CHECK(q[0].qs[5] == -2);
    CHECK(q[1].d == 0.f);                                  // all-zero block
    for (int j = 0; j < 256; ++j) CHECK(q[1].qs[j] == 0);
    for (int g = 0; g < 16; ++g) CHECK(q[1].bsums[g] == 0);
}

TEST(q8K_codes_bit_match_unfused_ggml_rounding) {
    // ggml rounds the *fp32-rounded* product iscale*x (no FMA). Recompute each
    // code with the product forced through a volatile, round-half-even, and
    // require exact equality (catches compiler FMA contraction).
    const int64_t n = 256 * 64;
    auto x = rand_vec(n, 4242, -7.f, 7.f);
    std::vector<block_q8_K> q(n / kQK_K);
    quantize_row_q8_K(x.data(), q.data(), n);
    for (int64_t b = 0; b < n / kQK_K; ++b) {
        float max = 0, amax = 0;
        for (int j = 0; j < kQK_K; ++j) {
            float v = x[b * kQK_K + j];
            if (std::fabs(v) > amax) { amax = std::fabs(v); max = v; }
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < kQK_K; ++j) {
            volatile float prod = iscale * x[b * kQK_K + j];
            const int want = std::min(127, (int)std::nearbyint((float)prod));
            CHECK_MSG(q[b].qs[j] == want, "Q8_K code differs from unfused ggml rounding");
        }
    }
}

TEST(vec_dot_simd_matches_ggml_generic_ref) {
    printf("        (kquant_simd_available = %d)\n", (int)kquant_simd_available());
    const int64_t n = 3840;
    for (DType t : kTypes) {
        auto w = make_weights(t, 16, n, 100 + (int)t);
        const int64_t rb = type_nbytes(t, n);
        std::vector<float> row(n);
        for (int rep = 0; rep < 16; ++rep) {
            auto x = rand_vec(n, 7 + rep);
            std::vector<block_q8_K> q(n / kQK_K);
            quantize_row_q8_K(x.data(), q.data(), n);
            const void* wr = w.data() + rep * rb;
            dequantize_row(t, wr, row.data(), n);
            double scale = 0;
            for (int64_t i = 0; i < n; ++i) scale += std::fabs(row[i] * x[i]);
            const float a = simd_dot(t)(n, wr, q.data());
            const float b = ref_dot(t)(n, wr, q.data());
            CHECK_MSG(std::fabs(a - b) <= 1e-5 * scale + 1e-7,
                      std::string(dtype_name(t)) + " simd vs ref mismatch");
        }
    }
}

TEST(vec_dot_matches_fp32_dequant_oracle) {
    const int64_t n = 4096;
    for (DType t : kTypes) {
        auto w = make_weights(t, 1, n, 200 + (int)t);
        std::vector<float> row(n);
        dequantize_row(t, w.data(), row.data(), n);
        for (int rep = 0; rep < 8; ++rep) {
            auto x = rand_vec(n, 50 + rep);
            std::vector<block_q8_K> q(n / kQK_K);
            quantize_row_q8_K(x.data(), q.data(), n);
            double exact = 0, scale = 0;
            for (int64_t i = 0; i < n; ++i) { exact += (double)row[i] * x[i]; scale += std::fabs(row[i] * x[i]); }
            const float a = simd_dot(t)(n, w.data(), q.data());
            // int8 activation quantization: per-term error <= |w| * d/2, d ~ 1/127.
            CHECK_MSG(std::fabs(a - exact) <= 1e-2 * scale,
                      std::string(dtype_name(t)) + " vs fp32 oracle");
        }
    }
}

TEST(matmul_kq_vs_matmul_quant) {
    const int64_t n_in = 2048, n_out = 96;
    ThreadPool pool(4);
    for (DType t : kTypes) {
        auto w = make_weights(t, n_out, n_in, 300 + (int)t);
        auto x = rand_vec(n_in, 9);
        std::vector<float> ya(n_out), yb(n_out), yc(n_out);
        matmul_quant(yb.data(), w.data(), t, x.data(), n_out, n_in, nullptr);
        matmul_kq_q8K(ya.data(), w.data(), t, x.data(), n_out, n_in, nullptr);
        matmul_kq_q8K(yc.data(), w.data(), t, x.data(), n_out, n_in, &pool);
        const double rel = norm_rel(ya, yb);
        printf("        %s matmul_kq vs fp32 oracle: norm-rel err %.2e\n", dtype_name(t), rel);
        CHECK_MSG(rel < 1e-2, std::string(dtype_name(t)) + " norm-rel error too large");
        CHECK_MSG(ya == yc, "threaded matmul_kq not bit-identical to single-thread");
    }
}

TEST(batch_bit_identical_to_single) {
    const int64_t n_in = 1536, n_out = 64;
    ThreadPool pool(4);
    for (DType t : kTypes) {
        auto w = make_weights(t, n_out, n_in, 400 + (int)t);
        for (int64_t m : {2, 5, 7, 9}) {   // covers the NR=4 remainder path
            auto X = rand_vec(m * n_in, 77 + m);
            std::vector<float> Yb(m * n_out), Yp(m * n_out), Ys(m * n_out);
            matmul_kq_q8K_batch(Yb.data(), w.data(), t, X.data(), m, n_out, n_in, nullptr);
            matmul_kq_q8K_batch(Yp.data(), w.data(), t, X.data(), m, n_out, n_in, &pool);
            for (int64_t r = 0; r < m; ++r)
                matmul_kq_q8K(Ys.data() + r * n_out, w.data(), t, X.data() + r * n_in, n_out, n_in, nullptr);
            CHECK_MSG(Yb == Ys, std::string(dtype_name(t)) + " batch != single");
            CHECK_MSG(Yp == Ys, std::string(dtype_name(t)) + " threaded batch != single");
        }
    }
}

TEST(linear_dispatch_default_unchanged_fast_routes) {
    const int64_t n_in = 1024, n_out = 48, m = 3;
    for (DType t : kTypes) {
        auto w = make_weights(t, n_out, n_in, 500 + (int)t);
        WeightRef W;
        W.data = w.data(); W.dtype = t; W.n_out = n_out; W.n_in = n_in;
        auto X = rand_vec(m * n_in, 5);
        std::vector<float> a(n_out), b(n_out), A(m * n_out), B(m * n_out);

        set_fast_quant(false);
        linear(a.data(), W, X.data());
        matmul_quant(b.data(), w.data(), t, X.data(), n_out, n_in);
        CHECK_MSG(std::memcmp(a.data(), b.data(), n_out * 4) == 0, "fast-off linear changed");
        linear_batch(A.data(), W, X.data(), m);
        matmul_quant_batch(B.data(), w.data(), t, X.data(), m, n_out, n_in);
        CHECK_MSG(std::memcmp(A.data(), B.data(), m * n_out * 4) == 0, "fast-off linear_batch changed");

        set_fast_quant(true);
        linear(a.data(), W, X.data());
        linear_batch(A.data(), W, X.data(), m);
        set_fast_quant(false);
        if (kquant_simd_available()) {
            matmul_kq_q8K(b.data(), w.data(), t, X.data(), n_out, n_in);
            matmul_kq_q8K_batch(B.data(), w.data(), t, X.data(), m, n_out, n_in);
        }  // else: fast still uses matmul_quant (b/B already hold it)
        CHECK_MSG(a == b, "fast linear did not route as expected");
        CHECK_MSG(A == B, "fast linear_batch did not route as expected");
    }
}

TEST(microbench_q4K_15360x3840) {
    using clk = std::chrono::steady_clock;
    const int64_t n_out = 15360, n_in = 3840;   // Gemma-class FFN gate/up shape
    const DType t = DType::Q4_K;
    auto w = make_weights(t, n_out, n_in, 999);
    auto best_ms = [&](int reps, auto&& fn) {
        double best = 1e30;
        for (int i = 0; i < reps; ++i) {
            auto t0 = clk::now(); fn();
            best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        }
        return best;
    };
    double load = -1;
    if (FILE* f = std::fopen("/proc/loadavg", "r")) { if (std::fscanf(f, "%lf", &load) != 1) load = -1; std::fclose(f); }
    printf("        microbench %s %lldx%lld (%.1f MB weights), loadavg(1m)=%.2f, simd=%d\n",
           dtype_name(t), (long long)n_out, (long long)n_in, w.size() / 1e6, load,
           (int)kquant_simd_available());

    ThreadPool p1(1), p8(8);
    auto x = rand_vec(n_in, 1);
    std::vector<float> y(n_out);
    const double gflop1 = 2.0 * n_out * n_in / 1e9;
    for (ThreadPool* p : {&p1, &p8}) {
        double old_ms = best_ms(3, [&] { matmul_quant(y.data(), w.data(), t, x.data(), n_out, n_in, p); });
        double new_ms = best_ms(5, [&] { matmul_kq_q8K(y.data(), w.data(), t, x.data(), n_out, n_in, p); });
        printf("        m=1  threads=%d  old %8.2f ms (%5.1f GFLOP/s)  new %8.2f ms (%5.1f GFLOP/s)  x%.1f\n",
               p->size(), old_ms, gflop1 / old_ms * 1e3, new_ms, gflop1 / new_ms * 1e3, old_ms / new_ms);
    }
    const int64_t m = 64;
    auto X = rand_vec(m * n_in, 2);
    std::vector<float> Y(m * n_out);
    double old_ms = best_ms(2, [&] { matmul_quant_batch(Y.data(), w.data(), t, X.data(), m, n_out, n_in, &p8); });
    double new_ms = best_ms(3, [&] { matmul_kq_q8K_batch(Y.data(), w.data(), t, X.data(), m, n_out, n_in, &p8); });
    printf("        m=%lld threads=8  old %8.2f ms (%5.1f GFLOP/s)  new %8.2f ms (%5.1f GFLOP/s)  x%.1f\n",
           (long long)m, old_ms, gflop1 * m / old_ms * 1e3, new_ms, gflop1 * m / new_ms * 1e3, old_ms / new_ms);
}

int main() { return llmtest::run_all(); }
