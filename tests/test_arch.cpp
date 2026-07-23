// test_arch.cpp — multi-architecture dispatch (#7 and friends).
//
// These are *implementation* checks: architecture resolution, per-arch metadata
// discovery, and that each new arch's block behaves as designed on a toy model.
// Cross-engine numeric parity (golden vs llama.cpp) needs a real model download
// + a llama.cpp build and is not run here.
#include "llm/gguf.h"
#include "llm/gguf_writer.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <string>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

// Streaming forward over a token sequence; returns the final logits.
static std::vector<float> forward_gguf(const std::string& path,
                                       const std::vector<int64_t>& toks) {
    GgufFile g(path);
    ModelConfig cfg = ModelConfig::from_source(g);
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
    LayerLoader loader(&g, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(2);
    Transformer tf(&loader, &kv, &pool);
    const float* lg = nullptr;
    for (size_t i = 0; i < toks.size(); ++i) lg = tf.forward(toks[i], (int64_t)i);
    return std::vector<float>(lg, lg + cfg.vocab_size);
}

TEST(arch_name_resolution) {
    CHECK(arch_from_name("llama") == Arch::Llama);
    CHECK(arch_from_name("mistral") == Arch::Mistral);
    CHECK(arch_from_name("totally-made-up") == Arch::Unknown);
    CHECK(std::string(arch_name(Arch::Mistral)) == "mistral");
}

TEST(mistral_config_discovery) {
    // hparams namespaced under "mistral.*" must be discovered generically.
    ToyGgufConfig c; c.arch = "mistral";
    c.n_layers = 3; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = 13;
    std::string p = scratch("mistral_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.arch == "mistral" && cfg.arch_kind == Arch::Mistral);
    CHECK(cfg.n_layers == 3 && cfg.n_heads == 4 && cfg.n_kv_heads == 2);
    CHECK(cfg.dim == 32 && cfg.ffn_dim == 64 && cfg.vocab_size == 48);
}

TEST(mistral_matches_llama_block) {
    // Identical weights (same seed), one labeled llama and one mistral. Mistral
    // is RMSNorm+RoPE+GQA+SwiGLU like Llama and routes to the same block, so the
    // logits must be byte-identical.
    ToyGgufConfig cl; cl.arch = "llama";
    cl.n_layers = 3; cl.dim = 32; cl.n_heads = 4; cl.n_kv_heads = 2;
    cl.ffn_dim = 64; cl.vocab_size = 48; cl.seed = 13;
    ToyGgufConfig cm = cl; cm.arch = "mistral";
    std::string pl = scratch("as_llama.gguf"), pm = scratch("as_mistral.gguf");
    write_toy_gguf(pl, cl); write_toy_gguf(pm, cm);

    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(pl, toks), b = forward_gguf(pm, toks);
    CHECK(a.size() == b.size() && !a.empty());
    for (size_t i = 0; i < a.size(); ++i) APPROX(a[i], b[i], 0);
}

int main() {
    printf("== test_arch ==\n");
    return llmtest::run_all();
}
