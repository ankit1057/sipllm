// test_prefill.cpp — RFC-007: single-pass batched prefill.
//
// Proves the batched prefill path is a bit-for-bit substitute for the old
// per-token loop: for the SAME multi-token prompt it must produce
//   * identical final-position logits,
//   * an identical KV cache (every layer, every filled position),
//   * and therefore the identical (greedy) sampled first token.
//
// This is the exact-match regression the RFC calls for. Batching only changes
// how often the weights are streamed (once per layer instead of once per
// token), never the arithmetic, so equality is EXACT (not tolerance-based).
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/sampler.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

// A full snapshot of a prefill run: the final-position logits and the entire
// filled KV cache (k and v for every layer x position).
struct Snapshot {
    std::vector<float> logits;
    std::vector<float> kv_k;
    std::vector<float> kv_v;
};

// Flatten the [0, n) filled region of the KV cache across all layers.
static void snapshot_kv(const KVCache& kv, int64_t n, Snapshot& s) {
    const int64_t nl = kv.n_layers(), kvd = kv.kv_dim();
    s.kv_k.reserve((size_t)nl * n * kvd);
    s.kv_v.reserve((size_t)nl * n * kvd);
    for (int64_t l = 0; l < nl; ++l)
        for (int64_t p = 0; p < n; ++p) {
            const float* k = kv.k(l, p);
            const float* v = kv.v(l, p);
            s.kv_k.insert(s.kv_k.end(), k, k + kvd);
            s.kv_v.insert(s.kv_v.end(), v, v + kvd);
        }
}

// Old path: forward() once per token; keep the last token's logits + final KV.
static Snapshot run_per_token(const std::string& path, LayerLoader::Options opt,
                              const std::vector<int64_t>& tokens) {
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(4);
    Transformer tf(&loader, &kv, &pool);
    const float* logits = nullptr;
    for (int64_t pos = 0; pos < (int64_t)tokens.size(); ++pos)
        logits = tf.forward(tokens[pos], pos);
    Snapshot s;
    s.logits.assign(logits, logits + cfg.vocab_size);
    snapshot_kv(kv, (int64_t)tokens.size(), s);
    return s;
}

// New path: prefill() the whole prompt in one batched sweep.
static Snapshot run_prefill(const std::string& path, LayerLoader::Options opt,
                            const std::vector<int64_t>& tokens) {
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(4);
    Transformer tf(&loader, &kv, &pool);
    const float* logits = tf.prefill(tokens.data(), (int64_t)tokens.size(), 0);
    Snapshot s;
    s.logits.assign(logits, logits + cfg.vocab_size);
    snapshot_kv(kv, (int64_t)tokens.size(), s);
    return s;
}

static void expect_bit_identical(const Snapshot& a, const Snapshot& b,
                                 const std::string& what) {
    CHECK_MSG(a.logits.size() == b.logits.size(), what + ": logits size");
    for (size_t i = 0; i < a.logits.size(); ++i)
        CHECK_MSG(a.logits[i] == b.logits[i], what + ": logits differ at " + std::to_string(i));
    CHECK_MSG(a.kv_k.size() == b.kv_k.size(), what + ": kv_k size");
    for (size_t i = 0; i < a.kv_k.size(); ++i)
        CHECK_MSG(a.kv_k[i] == b.kv_k[i], what + ": kv_k differ at " + std::to_string(i));
    CHECK_MSG(a.kv_v.size() == b.kv_v.size(), what + ": kv_v size");
    for (size_t i = 0; i < a.kv_v.size(); ++i)
        CHECK_MSG(a.kv_v[i] == b.kv_v[i], what + ": kv_v differ at " + std::to_string(i));
    // The sampled first token (greedy) must therefore match too.
    CHECK_MSG(Sampler::argmax(a.logits.data(), (int64_t)a.logits.size()) ==
              Sampler::argmax(b.logits.data(), (int64_t)b.logits.size()),
              what + ": argmax token differs");
}

// Multi-head attention (n_kv == n_heads).
TEST(prefill_matches_per_token_mha) {
    ToyConfig tc; tc.n_layers = 4; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 4;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 123;
    std::string path = scratch("toy_prefill_mha.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {3, 1, 4, 1, 5, 9, 2, 6, 8, 7};

    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = true; opt.n_buffers = 2;
    expect_bit_identical(run_per_token(path, opt, tokens),
                         run_prefill(path, opt, tokens), "mha");
}

// Grouped-query attention (n_kv < n_heads) — the case a shared K/V head must
// serve several query heads; batched prefill must still write/read KV correctly.
TEST(prefill_matches_per_token_gqa) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.seed = 7;
    std::string path = scratch("toy_prefill_gqa.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {7, 3, 11, 42, 5, 5, 1, 0, 23, 6, 6, 9};

    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = true; opt.n_buffers = 2;
    expect_bit_identical(run_per_token(path, opt, tokens),
                         run_prefill(path, opt, tokens), "gqa");
}

// Multi-query attention (single KV head) + quantized residency: exercises the
// quantized matmul kernels and the extreme GQA group size.
TEST(prefill_matches_per_token_mqa_quant) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 8; tc.n_kv_heads = 1;
    tc.ffn_dim = 64; tc.vocab_size = 40; tc.seed = 55;
    std::string path = scratch("toy_prefill_mqa.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {1, 2, 3, 4, 5, 6, 7};

    LayerLoader::Options opt; opt.residency = Residency::Quantized; opt.async = false; opt.n_buffers = 1;
    expect_bit_identical(run_per_token(path, opt, tokens),
                         run_prefill(path, opt, tokens), "mqa_quant");
}

// Tied embeddings (no separate output.weight): the output projection reuses the
// embedding table; prefill's final projection must handle it identically.
TEST(prefill_matches_per_token_tied) {
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 2; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 32; tc.tied = true; tc.seed = 5;
    std::string path = scratch("toy_prefill_tied.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {7, 3, 11, 2, 9, 1};

    LayerLoader::Options opt; opt.residency = Residency::FP32;
    expect_bit_identical(run_per_token(path, opt, tokens),
                         run_prefill(path, opt, tokens), "tied");
}

// A single-token prompt must degrade to exactly the one-token forward result.
TEST(prefill_single_token) {
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 40; tc.seed = 9;
    std::string path = scratch("toy_prefill_one.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {13};

    LayerLoader::Options opt; opt.residency = Residency::FP32;
    expect_bit_identical(run_per_token(path, opt, tokens),
                         run_prefill(path, opt, tokens), "single");
}

// ---- spans longer than one attention chunk (512 positions) ----------------
// Attention splits such a span into fixed chunks merged by log-sum-exp. The
// split must depend on the span alone, so a long prompt still prefills to the
// exact bytes of per-token decoding (spec verify relies on the same property)
// and no result depends on the thread count.

static std::vector<int64_t> long_prompt(int64_t n, int64_t vocab) {
    std::vector<int64_t> t((size_t)n);
    uint32_t s = 2463534242u;
    for (auto& v : t) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; v = s % vocab; }
    return t;
}

static Snapshot run_threads(const std::string& path, const std::vector<int64_t>& prompt,
                            const std::vector<int64_t>& decode, int threads) {
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
    ModelFile f(path, false);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(threads);
    Transformer tf(&loader, &kv, threads > 1 ? &pool : nullptr);
    const float* logits = tf.prefill(prompt.data(), (int64_t)prompt.size(), 0);
    int64_t pos = (int64_t)prompt.size();
    for (int64_t t : decode) logits = tf.forward(t, pos++);
    Snapshot s;
    s.logits.assign(logits, logits + cfg.vocab_size);
    snapshot_kv(kv, pos, s);
    return s;
}

static std::string write_long_mqa(const char* name) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 1;
    tc.ffn_dim = 64; tc.vocab_size = 40; tc.ctx_len = 1400; tc.seed = 77;
    std::string path = scratch(name);
    write_toy_model(path, tc);
    return path;
}

TEST(prefill_matches_per_token_long_span) {
    const std::string path = write_long_mqa("toy_prefill_long.llmw");
    const auto tokens = long_prompt(1200, 40);
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
    expect_bit_identical(run_per_token(path, opt, tokens),
                         run_prefill(path, opt, tokens), "long span");
}

TEST(attention_bytes_independent_of_thread_count) {
    const std::string path = write_long_mqa("toy_prefill_threads.llmw");
    const auto prompt = long_prompt(1100, 40);
    const std::vector<int64_t> decode = {3, 17, 29, 4};
    const Snapshot ref = run_threads(path, prompt, decode, 1);
    for (int threads : {2, 3, 8, 12})
        expect_bit_identical(ref, run_threads(path, prompt, decode, threads),
                             "threads=" + std::to_string(threads));
}

// Attention runs on the thread pool (one work item per token x head). Every
// item does exactly the arithmetic the serial loop did, so logits must be
// bit-identical for any thread count, for prefill and decode, FP32 and Q8 KV.
#include "llm/runtime.h"
#include <cstring>

static std::vector<float> run_threads(const std::string& path, int threads, KVPrecision kvp,
                                      std::string* text) {
    LayerLoader::Options opt; opt.residency = Residency::Quantized; opt.async = false;
    Runtime rt(open_model(path), opt, 256, threads, 0, false, kvp);
    SamplerConfig scfg; scfg.temperature = 0.0f;
    // A long prompt makes prefill attention span many positions and heads.
    std::string prompt;
    for (int i = 0; i < 5; ++i) prompt += "the quick brown fox jumps over the lazy dog ";
    *text = rt.generate(prompt, 24, scfg, nullptr, nullptr);
    return rt.first_logits();
}

TEST(attention_threads_bit_identical) {
    // head_dim 32: the Q8_0 KV path quantizes per head in blocks of 32.
    ToyConfig tc; tc.n_layers = 3; tc.dim = 256; tc.n_heads = 8; tc.n_kv_heads = 2;
    tc.ffn_dim = 256; tc.vocab_size = 256; tc.ctx_len = 256; tc.seed = 2026;
    std::string path = scratch("toy_attn_threads.llmw");
    write_toy_model(path, tc);
    for (KVPrecision kvp : {KVPrecision::FP32, KVPrecision::Q8_0}) {
        std::string t1, t4;
        const std::vector<float> l1 = run_threads(path, 1, kvp, &t1);
        const std::vector<float> l4 = run_threads(path, 4, kvp, &t4);
        CHECK_MSG(l1.size() == l4.size() && !l1.empty(), "first logits present");
        CHECK_MSG(std::memcmp(l1.data(), l4.data(), l1.size() * sizeof(float)) == 0,
                  "prefill logits bit-identical across thread counts");
        CHECK_MSG(t1 == t4, "greedy decode identical across thread counts");
    }
}

int main() {
    printf("== test_prefill ==\n");
    return llmtest::run_all();
}
