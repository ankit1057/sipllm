// test_e2e.cpp — Phase 4: prove the architecture end to end.
//
// Build a tiny random Llama (.llmw), then check that a forward pass driven
// entirely through the streaming LayerLoader matches the independent
// all-resident reference. Also confirm the async double-buffer pipeline and the
// synchronous single-buffer loader agree, and that the prefetcher scores hits.
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/ref_forward.h"
#include "tests/test_util.h"

#include <string>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

// Run the streaming Transformer over a token sequence; return last logits.
static std::vector<float> stream_forward(const std::string& path,
                                         LayerLoader::Options opt,
                                         const std::vector<int64_t>& tokens,
                                         LayerLoader::Stats* out = nullptr) {
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(4);
    Transformer tf(&loader, &kv, &pool);
    const float* logits = nullptr;
    for (int64_t pos = 0; pos < (int64_t)tokens.size(); ++pos)
        logits = tf.forward(tokens[pos], pos);
    std::vector<float> res(logits, logits + cfg.vocab_size);
    if (out) {
        out->prefetch_hits = loader.stats().prefetch_hits.load();
        out->prefetch_misses = loader.stats().prefetch_misses.load();
        out->bytes_read = loader.stats().bytes_read.load();
    }
    return res;
}

TEST(e2e_streaming_matches_reference) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.seed = 7;
    std::string path = scratch("toy_e2e.llmw");
    write_toy_model(path, tc);

    ModelFile f(path);
    ModelConfig cfg = ModelConfig::from_source(f);
    std::vector<int64_t> tokens = {3, 1, 4, 1, 5, 9, 2, 6};
    auto ref = llmtest::ref_forward(f, cfg, tokens);

    LayerLoader::Options opt;
    opt.residency = Residency::FP32; opt.async = true; opt.n_buffers = 2;
    auto got = stream_forward(path, opt, tokens);

    CHECK(ref.size() == got.size());
    for (size_t i = 0; i < ref.size(); ++i) APPROX(got[i], ref[i], 1e-3);
}

TEST(e2e_async_equals_sync) {
    ToyConfig tc; tc.n_layers = 4; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 4;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 99;
    std::string path = scratch("toy_sync.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {1, 2, 3, 4, 5};

    LayerLoader::Options a; a.async = true;  a.n_buffers = 2; a.residency = Residency::Quantized;
    LayerLoader::Options b; b.async = false; b.n_buffers = 1; b.residency = Residency::Quantized;
    LayerLoader::Stats sa;
    auto ga = stream_forward(path, a, tokens, &sa);
    auto gb = stream_forward(path, b, tokens);
    for (size_t i = 0; i < ga.size(); ++i) APPROX(ga[i], gb[i], 1e-4);
    CHECK_MSG(sa.prefetch_hits > 0, "expected prefetch hits > 0");
}

TEST(e2e_tied_embeddings) {
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 2; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 32; tc.tied = true; tc.seed = 5;
    std::string path = scratch("toy_tied.llmw");
    write_toy_model(path, tc);

    ModelFile f(path);
    ModelConfig cfg = ModelConfig::from_source(f);
    CHECK(cfg.tie_embeddings);
    std::vector<int64_t> tokens = {7, 3, 11};
    auto ref = llmtest::ref_forward(f, cfg, tokens);
    LayerLoader::Options opt; opt.residency = Residency::FP32;
    auto got = stream_forward(path, opt, tokens);
    for (size_t i = 0; i < ref.size(); ++i) APPROX(got[i], ref[i], 1e-3);
}

int main() {
    printf("== test_e2e ==\n");
    return llmtest::run_all();
}
