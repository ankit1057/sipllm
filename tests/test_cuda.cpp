// test_cuda.cpp — optional CUDA backend (cuda_backend.h, --gpu-layers).
//
// Without an NVIDIA driver (no libcuda.so.1) or with SIPLLM_NO_CUDA=1, every
// GPU case prints SKIP and passes: the default build/test must stay green on
// machines with no GPU. With a GPU present:
//  * GPU matvec (m=1) vs the CPU fp32-dequant oracle for Q4_K/Q5_K/Q6_K/F16/F32
//    on random blocks: norm-relative error < 1e-3 (expected ~1e-6, only the
//    summation order differs).
//  * GPU small-batch matmul (prefill) for m spanning the 16-token CTA tile and
//    the 64-token host chunk, every row vs the oracle.
//  * linear()/linear_batch() dispatch on WeightRef::dev.
//  * LayerLoader --gpu-layers on a toy model: logits match the CPU run, also
//    combined with RAM-budget pinning and the GPU LM head.
#include "llm/cuda_backend.h"
#include "llm/kquant.h"
#include "llm/linear.h"
#include "llm/quant.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "llm/sampler.h"
#include "tests/test_util.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace llm;

namespace {

bool gpu_or_skip(const char* what) {
    if (cuda::available()) return true;
    printf("  SKIP  %s: %s\n", what, cuda::unavailable_reason().c_str());
    return false;
}

// Random weight matrix [rows, n] in GGUF layout: random payload bytes (every
// packing case) with sane fp16 super-block scales (no NaN/Inf).
std::vector<uint8_t> make_weights(DType t, int64_t rows, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<uint8_t> w((size_t)type_nbytes(t, n) * rows);
    if (t == DType::F32) {
        float* f = reinterpret_cast<float*>(w.data());
        for (int64_t i = 0; i < rows * n; ++i) f[i] = nd(rng);
        return w;
    }
    if (t == DType::F16) {
        uint16_t* h = reinterpret_cast<uint16_t*>(w.data());
        for (int64_t i = 0; i < rows * n; ++i) h[i] = fp32_to_fp16(nd(rng));
        return w;
    }
    for (auto& b : w) b = (uint8_t)byte(rng);
    const int64_t bsz = type_traits(t).type_size;
    const int64_t nblk = rows * (n / kQK_K);
    for (int64_t i = 0; i < nblk; ++i) {
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

std::vector<float> rand_vec(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// CPU oracle in double: dequantize each row to fp32, dot with x.
std::vector<float> ref_matmul(const std::vector<uint8_t>& W, DType t, int64_t rows,
                              int64_t n, const float* X, int64_t m) {
    std::vector<float> row(n), Y((size_t)(m * rows));
    const int64_t rb = type_nbytes(t, n);
    for (int64_t r = 0; r < rows; ++r) {
        dequantize_row(t, W.data() + r * rb, row.data(), n);
        for (int64_t b = 0; b < m; ++b) {
            double s = 0;
            for (int64_t i = 0; i < n; ++i) s += (double)row[i] * X[b * n + i];
            Y[(size_t)(b * rows + r)] = (float)s;
        }
    }
    return Y;
}

double norm_rel(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        den += (double)b[i] * b[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1));
}

const DType kTypes[] = {DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::F16, DType::F32};

struct DevWeight {
    WeightRef ref;
    explicit DevWeight(const std::vector<uint8_t>& host, DType t, int64_t rows, int64_t n) {
        ref.dtype = t; ref.n_out = rows; ref.n_in = n;
        ref.dev = cuda::upload_weight(host.data(), t, rows, n);
    }
    ~DevWeight() { cuda::free_weight(ref.dev); }
};

} // namespace

TEST(cuda_supports_and_sizes) {
    // Pure host-side logic: valid with or without a GPU.
    CHECK(cuda::supports(DType::Q4_K, 3840));
    CHECK(cuda::supports(DType::Q6_K, 15360));
    CHECK(!cuda::supports(DType::Q4_K, 100));     // not a multiple of 256
    CHECK(!cuda::supports(DType::Q8_0, 4096));    // no kernel
    CHECK(cuda::weight_bytes(DType::Q4_K, 2, 512) == 2 * 2 * 144);
    CHECK(cuda::weight_bytes(DType::Q6_K, 3, 256) == 3 * 212);   // padded block
    CHECK(cuda::parse_gpu_layers("max") == -1);
    CHECK(cuda::parse_gpu_layers("12") == 12);
    CHECK(cuda::parse_gpu_layers("0") == 0);
    WeightRef w; CHECK(!w.valid()); CHECK(!w.on_gpu());
    w.dev = 1; CHECK(w.valid()); CHECK(w.on_gpu());
    if (!cuda::available()) printf("  (CUDA unavailable: %s)\n", cuda::unavailable_reason().c_str());
    else printf("  (CUDA device: %s)\n", cuda::device_name().c_str());
}

TEST(cuda_matvec_vs_cpu_all_types) {
    if (!gpu_or_skip("cuda_matvec_vs_cpu_all_types")) return;
    struct Shape { int64_t rows, n; };
    const Shape shapes[] = {{37, 768}, {2048, 3840}, {130, 15360}};
    uint32_t seed = 1;
    for (DType t : kTypes)
        for (const Shape& sh : shapes) {
            auto W = make_weights(t, sh.rows, sh.n, seed++);
            auto x = rand_vec(sh.n, seed++);
            auto ref = ref_matmul(W, t, sh.rows, sh.n, x.data(), 1);
            DevWeight dw(W, t, sh.rows, sh.n);
            CHECK_MSG(dw.ref.dev != 0, "upload failed");
            std::vector<float> y(sh.rows, -999.f);
            cuda::linear(y.data(), dw.ref, x.data(), 1);
            const double e = norm_rel(y.data(), ref.data(), y.size());
            char msg[160];
            snprintf(msg, sizeof(msg), "%s %lldx%lld matvec rel err %.3g", dtype_name(t),
                     (long long)sh.rows, (long long)sh.n, e);
            CHECK_MSG(e < 1e-3, msg);
            if (sh.rows == 2048) printf("        %s\n", msg);
        }
}

TEST(cuda_batched_matmul_vs_cpu) {
    if (!gpu_or_skip("cuda_batched_matmul_vs_cpu")) return;
    const int64_t rows = 53, n = 1024;          // rows not a multiple of the 16-row CTA
    const int64_t ms[] = {2, 5, 16, 17, 64, 70};  // CTA tile (16) and host chunk (64) edges
    uint32_t seed = 100;
    for (DType t : kTypes) {
        auto W = make_weights(t, rows, n, seed++);
        DevWeight dw(W, t, rows, n);
        CHECK_MSG(dw.ref.dev != 0, "upload failed");
        for (int64_t m : ms) {
            auto X = rand_vec(m * n, seed++);
            auto ref = ref_matmul(W, t, rows, n, X.data(), m);
            std::vector<float> Y((size_t)(m * rows), -999.f);
            cuda::linear(Y.data(), dw.ref, X.data(), m);
            for (int64_t b = 0; b < m; ++b) {
                const double e = norm_rel(Y.data() + b * rows, ref.data() + b * rows, rows);
                char msg[160];
                snprintf(msg, sizeof(msg), "%s m=%lld row %lld rel err %.3g", dtype_name(t),
                         (long long)m, (long long)b, e);
                CHECK_MSG(e < 1e-3, msg);
            }
        }
    }
}

TEST(cuda_linear_dispatch) {
    if (!gpu_or_skip("cuda_linear_dispatch")) return;
    const int64_t rows = 64, n = 512, m = 3;
    auto W = make_weights(DType::Q4_K, rows, n, 7);
    auto X = rand_vec(m * n, 8);
    std::vector<float> cpu1(rows), gpu1(rows), cpuB(m * rows), gpuB(m * rows);
    WeightRef host{W.data(), DType::Q4_K, rows, n};
    linear(cpu1.data(), host, X.data());
    linear_batch(cpuB.data(), host, X.data(), m);
    DevWeight dw(W, DType::Q4_K, rows, n);
    CHECK(dw.ref.data == nullptr && dw.ref.valid());
    linear(gpu1.data(), dw.ref, X.data());
    linear_batch(gpuB.data(), dw.ref, X.data(), m);
    CHECK(norm_rel(gpu1.data(), cpu1.data(), rows) < 1e-4);
    CHECK(norm_rel(gpuB.data(), cpuB.data(), m * rows) < 1e-4);
}

namespace {
struct ToyRun { std::vector<float> logits; int gpu = 0; int pinned = 0; bool head = false; };

ToyRun run_toy(const std::string& path, int gpu_layers, size_t budget, bool gpu_output,
               bool batched) {
    ModelFile f(path, false);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader::Options opt;
    opt.gpu_layers = gpu_layers;
    opt.gpu_output = gpu_output;
    opt.ram_budget_bytes = budget;
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(2);
    Transformer tf(&loader, &kv, &pool);
    const std::vector<int64_t> toks = {3, 1, 4, 1, 5, 9, 2, 6};
    const float* lg = nullptr;
    if (batched) {
        lg = tf.prefill(toks.data(), (int64_t)toks.size(), 0);
        lg = tf.forward(7, (int64_t)toks.size());
    } else {
        for (int64_t p = 0; p < (int64_t)toks.size(); ++p) lg = tf.forward(toks[p], p);
    }
    ToyRun r;
    r.logits.assign(lg, lg + cfg.vocab_size);
    r.gpu = loader.gpu_layers();
    r.pinned = loader.pinned_layers();
    r.head = loader.gpu_output();
    return r;
}
} // namespace

TEST(cuda_loader_gpu_layers_toy_model) {
    if (!gpu_or_skip("cuda_loader_gpu_layers_toy_model")) return;
    ToyConfig tc; tc.n_layers = 4; tc.dim = 256; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 512; tc.vocab_size = 64; tc.seed = 99; tc.tied = true;
    const std::string path = llmtest::scratch_path("toy_cuda.llmw");
    write_toy_model(path, tc);
    for (bool batched : {false, true}) {
        const ToyRun cpu = run_toy(path, 0, 0, true, batched);
        CHECK(cpu.gpu == 0 && !cpu.head);
        const ToyRun g2 = run_toy(path, 2, 0, false, batched);         // partial offload
        CHECK_MSG(g2.gpu == 2 && !g2.head, "2 layers on GPU, head on CPU");
        const ToyRun gall = run_toy(path, -1, 0, true, batched);       // everything + head
        CHECK_MSG(gall.gpu == 4 && gall.head, "all layers + head on GPU");
        const ToyRun gpin = run_toy(path, 1, (size_t)1 << 30, false, batched);  // + RAM pinning
        CHECK_MSG(gpin.gpu == 1 && gpin.pinned == 3, "1 GPU layer + 3 RAM-pinned layers");
        for (const ToyRun* r : {&g2, &gall, &gpin}) {
            const double e = norm_rel(r->logits.data(), cpu.logits.data(), cpu.logits.size());
            char msg[128];
            snprintf(msg, sizeof(msg), "toy logits rel err %.3g (gpu=%d head=%d batched=%d)", e,
                     r->gpu, (int)r->head, (int)batched);
            CHECK_MSG(e < 1e-4, msg);
            CHECK_MSG(Sampler::argmax(r->logits.data(), (int64_t)r->logits.size()) ==
                      Sampler::argmax(cpu.logits.data(), (int64_t)cpu.logits.size()), msg);
        }
    }
}

int main() { return llmtest::run_all(); }
