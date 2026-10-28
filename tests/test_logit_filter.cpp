// test_logit_filter.cpp — Runtime::set_logit_filter: the filter decides what
// can be sampled (prefill's first token and every decode step), and leaving it
// unset is byte-identical to the unfiltered engine. Byte-tokenizer toy model.
#include "llm/runtime.h"
#include "llm/toy_model.h"
#include "tests/test_util.h"

#include <cmath>
#include <memory>
#include <string>

using namespace llm;

static std::string make_toy() {
    ToyConfig tc;
    tc.n_layers = 2;
    tc.dim = 32;
    tc.n_heads = 4;
    tc.n_kv_heads = 2;
    tc.ffn_dim = 64;
    tc.vocab_size = 256;   // byte tokenizer
    tc.ctx_len = 128;
    tc.seed = 5;
    std::string path = llmtest::scratch_path("toy_filter.llmw");
    write_toy_model(path, tc);
    return path;
}

static std::unique_ptr<Runtime> make_rt(const std::string& path) {
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    opt.n_buffers = 1;
    return std::make_unique<Runtime>(open_model(path), opt, /*max_ctx=*/0, /*threads=*/1);
}

static SamplerConfig greedy() {
    SamplerConfig s;
    s.temperature = 0.0f;
    return s;
}

TEST(logit_filter_unset_is_byte_identical) {
    const std::string path = make_toy();
    GenStats st;
    std::string a = make_rt(path)->generate("hello", 12, greedy(), nullptr, &st);
    std::string b = make_rt(path)->generate("hello", 12, greedy(), nullptr, &st);
    CHECK(a == b);
    auto rt = make_rt(path);
    rt->set_logit_filter(nullptr);
    CHECK(rt->generate("hello", 12, greedy(), nullptr, &st) == a);
}

TEST(logit_filter_controls_every_sampled_token) {
    const std::string path = make_toy();
    auto rt = make_rt(path);
    const int64_t x = rt->tokenizer().encode("x", /*add_bos=*/false).at(0);
    // Only 'x' may ever be sampled, including the first token after prefill.
    rt->set_logit_filter([x](float* logits, int64_t vocab) {
        for (int64_t i = 0; i < vocab; ++i)
            if (i != x) logits[i] = -INFINITY;
    });
    GenStats st;
    const std::string out = rt->generate("hello", 8, greedy(), nullptr, &st);
    CHECK_MSG(out == "xxxxxxxx", out);
}

TEST(logit_filter_can_depend_on_generation_state) {
    // Agent-style rule: a token is allowed only as the first generated token.
    const std::string path = make_toy();
    auto base = make_rt(path);
    GenStats st;
    const std::string free_out = base->generate("hello", 10, greedy(), nullptr, &st);
    const int64_t first = base->tokenizer().encode(free_out.substr(0, 1), false).at(0);
    auto rt = make_rt(path);
    int generated = 0;
    rt->set_logit_filter([&generated, first](float* logits, int64_t) {
        if (generated > 0) logits[first] = -INFINITY;
    });
    const std::string out = rt->generate("hello", 10, greedy(),
        [&generated](const std::string&, int64_t) { ++generated; return true; }, &st);
    CHECK(!out.empty() && out[0] == free_out[0]);                    // allowed first
    CHECK_MSG(out.find(free_out[0], 1) == std::string::npos, out);   // never again
}

TEST(logit_filter_applies_to_speculative_steps) {
    // --spec verifies drafted tokens against the model's logits; the filter
    // must see those logits too, so spec output equals filtered plain output.
    // Only 'x' is allowed and the prompt ends in x's, so prompt lookup drafts x's
    // that the unfiltered model would reject.
    const std::string path = make_toy();
    auto spec = make_rt(path);
    const int64_t x = spec->tokenizer().encode("x", /*add_bos=*/false).at(0);
    spec->set_logit_filter([x](float* logits, int64_t vocab) {
        for (int64_t i = 0; i < vocab; ++i)
            if (i != x) logits[i] = -INFINITY;
    });
    spec->enable_speculative_decoding(3);
    GenStats st;
    const std::string out = spec->generate("hello xxxxxxxx", 12, greedy(), nullptr, &st);
    CHECK_MSG(st.spec_drafted > 0, "speculative path was not exercised");
    CHECK_MSG(out == "xxxxxxxxxxxx", out);
}

int main() {
    printf("== test_logit_filter ==\n");
    return llmtest::run_all();
}
