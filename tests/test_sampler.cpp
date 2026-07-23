// test_sampler.cpp — sampler determinism and the repetition penalty (#1).
#include "llm/sampler.h"
#include "tests/test_util.h"

#include <vector>

using namespace llm;

// Peaked logits: a clear winner (idx 2 = 10) and runner-up (idx 4 = 6). Used
// for the greedy repetition-penalty cases where the ordering must be exact.
static std::vector<float> peaked() { return {0.f, 0.f, 10.f, 0.f, 6.f}; }
// Flat-ish logits so the stochastic path actually explores multiple tokens.
static std::vector<float> flat() { return {2.f, 1.5f, 1.0f, 0.5f, 0.2f}; }

TEST(sampler_seed_is_deterministic) {
    SamplerConfig cfg;
    cfg.temperature = 1.0f; cfg.top_k = 0; cfg.top_p = 1.0f; cfg.seed = 123456789ULL;
    Sampler a(cfg), b(cfg);
    auto L = flat();
    std::vector<int64_t> sa, sb;
    for (int i = 0; i < 32; ++i) { sa.push_back(a.sample(L.data(), (int64_t)L.size()));
                                   sb.push_back(b.sample(L.data(), (int64_t)L.size())); }
    CHECK(sa == sb);                         // same seed => identical stream
    // and the stream is not constant (temperature actually sampled)
    bool varied = false;
    for (size_t i = 1; i < sa.size(); ++i) if (sa[i] != sa[0]) { varied = true; break; }
    CHECK(varied);
}

TEST(sampler_different_seed_differs) {
    SamplerConfig c1; c1.temperature = 1.0f; c1.top_k = 0; c1.top_p = 1.0f; c1.seed = 1;
    SamplerConfig c2; c2.temperature = 1.0f; c2.top_k = 0; c2.top_p = 1.0f; c2.seed = 2;
    Sampler a(c1), b(c2);
    auto L = flat();
    std::vector<int64_t> sa, sb;
    for (int i = 0; i < 64; ++i) { sa.push_back(a.sample(L.data(), (int64_t)L.size()));
                                   sb.push_back(b.sample(L.data(), (int64_t)L.size())); }
    CHECK(sa != sb);                         // overwhelmingly likely to diverge
}

TEST(repeat_penalty_greedy_demotes_repeat) {
    // Greedy: token 2 (logit 10) wins first. With penalty 2.0 its logit becomes
    // 5, below token 4 (logit 6), so the second draw switches to token 4.
    SamplerConfig cfg;
    cfg.temperature = 0.f; cfg.repeat_penalty = 2.0f; cfg.repeat_last_n = 64;
    Sampler s(cfg);
    auto L = peaked();
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 2);
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 4);
}

TEST(repeat_penalty_disabled_keeps_repeat) {
    // penalty == 1.0 (default): greedy keeps picking the same winner.
    SamplerConfig cfg; cfg.temperature = 0.f; cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);
    auto L = peaked();
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 2);
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 2);
}

TEST(repeat_penalty_window_forgets_old_tokens) {
    // repeat_last_n == 1: only the immediately previous token is penalized.
    SamplerConfig cfg;
    cfg.temperature = 0.f; cfg.repeat_penalty = 2.0f; cfg.repeat_last_n = 1;
    Sampler s(cfg);
    auto L = peaked();
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 2);  // history {}   -> 2
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 4);  // window {2}   -> 4
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 2);  // window {4}   -> 2 again
}

TEST(repeat_penalty_negative_logit_pushed_down) {
    // A negative logit is multiplied by the penalty (pushed further down), never
    // flipped up. Seed token 2 into history directly: without penalty its -0.5
    // would beat token 4's -0.9; penalized it becomes -1.0 and loses.
    SamplerConfig cfg; cfg.temperature = 0.f; cfg.repeat_penalty = 2.0f;
    Sampler s(cfg);
    s.accept(2);
    std::vector<float> L = {-2.f, -2.f, -0.5f, -2.f, -0.9f};
    CHECK(s.sample(L.data(), (int64_t)L.size()) == 4);
}

int main() { return llmtest::run_all(); }
