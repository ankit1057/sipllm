#include "llm/runtime.h"
#include "llm/semantic_cache.h"
#include "llm/toy_model.h"
#include "tests/test_util.h"

#include <algorithm>
#include <cmath>

using namespace llm;

static std::string make_toy_sc() {
    ToyConfig tc;
    tc.n_layers = 2;
    tc.dim = 32;
    tc.n_heads = 4;
    tc.n_kv_heads = 2;
    tc.ffn_dim = 64;
    tc.vocab_size = 256;
    tc.ctx_len = 256;
    tc.seed = 42;
    std::string path = llmtest::scratch_path("toy_sem_cache.llmw");
    write_toy_model(path, tc);
    return path;
}

TEST(semantic_cache_basic_prefix_match) {
    const std::string path = make_toy_sc();
    auto src = open_model(path);
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    Runtime rt(std::move(src), opt, 256, 1);
    rt.enable_semantic_cache(10 * 1024 * 1024); // 10MB budget

    SamplerConfig scfg;
    scfg.temperature = 0.0f; // greedy

    // Turn 1: cold cache
    GenStats s1;
    rt.generate("The capital of France is", 4, scfg, nullptr, &s1);
    CHECK(s1.semantic_cache_hits == 0);

    // Reset runtime state (simulating new prompt/session)
    rt.reset();

    // Turn 2: same prefix + continuation
    GenStats s2;
    rt.generate("The capital of France is Paris", 4, scfg, nullptr, &s2);
    // Should have matched the prefix from turn 1!
    CHECK(s2.semantic_cache_hits > 0);
}

TEST(semantic_cache_full_prompt_hit_matches_cold_output) {
    // A repeated prompt is fully cached, so the hit is trimmed by one token (the
    // last token is re-run for logits). The injected rows must still line up per
    // layer: the output must equal a cold run, token for token.
    const std::string path = make_toy_sc();
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    SamplerConfig scfg;
    scfg.temperature = 0.0f;
    const std::string prompt = "The capital of France is";

    Runtime cold(open_model(path), opt, 256, 1);
    const std::string want = cold.generate(prompt, 8, scfg, nullptr, nullptr);
    const std::vector<float> want_logits = cold.first_logits();

    Runtime rt(open_model(path), opt, 256, 1);
    rt.enable_semantic_cache(10 * 1024 * 1024);
    rt.generate(prompt, 8, scfg, nullptr, nullptr);
    rt.reset();
    GenStats st;
    const std::string got = rt.generate(prompt, 8, scfg, nullptr, &st);
    CHECK(st.semantic_cache_hits > 0);
    CHECK(got == want);
    // Text can coincide on a toy model; the first-step logits cannot.
    const std::vector<float>& got_logits = rt.first_logits();
    CHECK(got_logits.size() == want_logits.size());
    float max_diff = 0.f;
    for (size_t i = 0; i < want_logits.size() && i < got_logits.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(got_logits[i] - want_logits[i]));
    CHECK(max_diff < 1e-4f);
}

TEST(semantic_cache_eviction_bounded) {
    // 50KB small budget
    SemanticCache cache(50 * 1024, 2, 32);
    std::vector<float> k_data(2 * 100 * 32, 1.0f);
    std::vector<float> v_data(2 * 100 * 32, 2.0f);

    auto k_fn = [&](int64_t l, int64_t p) -> const float* {
        return k_data.data() + (l * 100 + p) * 32;
    };
    auto v_fn = [&](int64_t l, int64_t p) -> const float* {
        return v_data.data() + (l * 100 + p) * 32;
    };

    // Insert 10 distinct long sequences
    for (int64_t s = 0; s < 10; ++s) {
        std::vector<int64_t> tokens(50);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = s * 1000 + i;
        cache.commit(tokens, k_fn, v_fn);
    }

    // Must never exceed memory budget
    CHECK(cache.current_bytes() <= cache.max_bytes());
}

int main() {
    printf("== test_semantic_cache ==\n");
    return llmtest::run_all();
}
