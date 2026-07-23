// sampler.h — turn logits into a token id.
//
// Greedy when temperature <= 0; otherwise temperature + top-k + top-p (nucleus)
// with a small xorshift RNG so runs are reproducible from a seed. A repetition
// penalty (Keskar et al. / llama.cpp) discourages tokens already seen in a
// recent window; it applies to both the greedy and stochastic paths.
#pragma once

#include "llm/common.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace llm {

struct SamplerConfig {
    float    temperature = 0.8f;
    int      top_k = 40;             // <=0 disables
    float    top_p = 0.95f;          // 1.0 disables
    float    repeat_penalty = 1.0f;  // 1.0 disables; >1 penalizes repeats
    int      repeat_last_n = 64;     // history window; <=0 = whole history
    uint64_t seed = 0x2545F4914F6CDD1DULL;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg) : cfg_(cfg), state_(cfg.seed ? cfg.seed : 1) {}

    // Record a token (prompt or generated) into the recent-history window so it
    // participates in the repetition penalty. sample() calls this for the token
    // it returns; callers seed prompt tokens with it before decoding.
    void accept(int64_t token) { history_.push_back(token); }

    int64_t sample(const float* logits, int64_t vocab) {
        const float* L = logits;
        if (cfg_.repeat_penalty != 1.0f && !history_.empty()) {
            work_.assign(logits, logits + vocab);
            apply_repeat_penalty(work_.data(), vocab);
            L = work_.data();
        }
        int64_t tok = (cfg_.temperature <= 0.f) ? argmax(L, vocab)
                                                : sample_temperature(L, vocab);
        accept(tok);
        return tok;
    }

    static int64_t argmax(const float* v, int64_t n) {
        int64_t bi = 0; float bv = v[0];
        for (int64_t i = 1; i < n; ++i) if (v[i] > bv) { bv = v[i]; bi = i; }
        return bi;
    }

private:
    // Divide the logit of each token seen in the recent window by the penalty
    // (or multiply, when the logit is negative — pushing it further down), once
    // per unique token id. Mirrors llama.cpp's `--repeat-penalty`.
    void apply_repeat_penalty(float* logits, int64_t vocab) {
        size_t start = 0;
        if (cfg_.repeat_last_n > 0 && history_.size() > (size_t)cfg_.repeat_last_n)
            start = history_.size() - (size_t)cfg_.repeat_last_n;
        for (size_t j = start; j < history_.size(); ++j) {
            const int64_t id = history_[j];
            if (id < 0 || id >= vocab) continue;
            bool seen = false;                                  // apply once per id
            for (size_t k = start; k < j; ++k) if (history_[k] == id) { seen = true; break; }
            if (seen) continue;
            float& v = logits[id];
            v = (v > 0.f) ? v / cfg_.repeat_penalty : v * cfg_.repeat_penalty;
        }
    }

    int64_t sample_temperature(const float* logits, int64_t vocab) {
        // softmax with temperature over a working copy
        probs_.resize(vocab);
        float maxl = logits[0];
        for (int64_t i = 1; i < vocab; ++i) maxl = std::max(maxl, logits[i]);
        double sum = 0;
        const float inv_t = 1.0f / cfg_.temperature;
        for (int64_t i = 0; i < vocab; ++i) {
            float p = std::exp((logits[i] - maxl) * inv_t);
            probs_[i] = p; sum += p;
        }
        for (int64_t i = 0; i < vocab; ++i) probs_[i] /= (float)sum;

        // build candidate index list, sort by prob desc
        idx_.resize(vocab);
        for (int64_t i = 0; i < vocab; ++i) idx_[i] = i;
        int64_t k = cfg_.top_k > 0 ? std::min<int64_t>(cfg_.top_k, vocab) : vocab;
        std::partial_sort(idx_.begin(), idx_.begin() + k, idx_.end(),
                          [&](int64_t a, int64_t b) { return probs_[a] > probs_[b]; });

        // nucleus: keep the smallest prefix whose mass >= top_p
        double cum = 0; int64_t keep = k;
        for (int64_t i = 0; i < k; ++i) {
            cum += probs_[idx_[i]];
            if (cum >= cfg_.top_p) { keep = i + 1; break; }
        }

        // renormalize and sample
        double norm = 0;
        for (int64_t i = 0; i < keep; ++i) norm += probs_[idx_[i]];
        double r = next_uniform() * norm, acc = 0;
        for (int64_t i = 0; i < keep; ++i) {
            acc += probs_[idx_[i]];
            if (r <= acc) return idx_[i];
        }
        return idx_[keep - 1];
    }

    double next_uniform() {
        // xorshift64*
        state_ ^= state_ >> 12; state_ ^= state_ << 25; state_ ^= state_ >> 27;
        uint64_t x = state_ * 0x2545F4914F6CDD1DULL;
        return (x >> 11) * (1.0 / 9007199254740992.0);
    }
    SamplerConfig cfg_;
    uint64_t state_;
    std::vector<float> probs_;
    std::vector<float> work_;        // penalized logit scratch
    std::vector<int64_t> idx_;
    std::vector<int64_t> history_;   // accepted tokens (prompt + generated)
};

} // namespace llm
