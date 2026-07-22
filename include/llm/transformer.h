// transformer.h — Task 5: the decoder forward pass.
//
// A Llama-style block: RMSNorm -> RoPE'd causal GQA attention -> residual ->
// RMSNorm -> SwiGLU MLP -> residual. Every weight is fetched through
// LayerLoader, so the entire forward pass runs with only one (or two, when
// double-buffered) transformer block resident in RAM at any instant — the
// whole point of the project.
#pragma once

#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/model.h"
#include "llm/threadpool.h"

#include <functional>
#include <vector>

namespace llm {

class Transformer {
public:
    Transformer(LayerLoader* loader, KVCache* kv, ThreadPool* pool = nullptr);

    // Run one token at absolute position `pos`; returns a pointer to the vocab
    // logits (owned by the Transformer, valid until the next forward call).
    // Writes K/V for `pos` into the cache and attends over [0, pos].
    const float* forward(int64_t token, int64_t pos);

    // RoPE is exposed for testing.
    static void apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                           int64_t pos, float theta_base);

    const ModelConfig& config() const { return cfg_; }

    // ---- profiling (Task 13) --------------------------------------------
    // Per-layer breakdown of the most recent forward() call. io/dequant come
    // from the loader (may be ~0 when a layer was prefetched during the prior
    // layer's compute — that overlap is the whole point of the pipeline).
    struct LayerTiming {
        double load_ms = 0;     // time spent inside loadLayer (wait/sync read)
        double io_ms = 0;       // pread time attributed to this layer
        double dequant_ms = 0;  // dequant time attributed to this layer
        double compute_ms = 0;  // attention + MLP math
        size_t rss_bytes = 0;   // resident memory sampled after this layer
    };
    void enable_profiling(bool on) { profiling_ = on; }
    const std::vector<LayerTiming>& last_timings() const { return timings_; }
    size_t peak_rss() const { return peak_rss_; }

    // Golden-test hook: invoked with the residual stream after each block, so a
    // harness can checksum/compare hidden states layer-by-layer with llama.cpp.
    using HiddenHook = std::function<void(int layer, const float* x, int64_t dim)>;
    void set_hidden_hook(HiddenHook h) { hidden_hook_ = std::move(h); }

private:
    void block(int64_t layer, int64_t pos);   // one transformer block on x_

    LayerLoader* loader_;
    KVCache*     kv_;
    ThreadPool*  pool_;
    ModelConfig  cfg_;

    // scratch, allocated once and reused every token
    std::vector<float> x_;       // dim   — residual stream
    std::vector<float> xb_;      // dim   — normed input
    std::vector<float> q_;       // q_dim
    std::vector<float> k_;       // kv_dim
    std::vector<float> v_;       // kv_dim
    std::vector<float> att_;     // max_ctx (attention scores per head)
    std::vector<float> attn_out_;// q_dim
    std::vector<float> proj_;    // dim
    std::vector<float> hb_;      // ffn_dim
    std::vector<float> hb2_;     // ffn_dim
    std::vector<float> logits_;  // vocab

    bool profiling_ = false;
    std::vector<LayerTiming> timings_;
    size_t peak_rss_ = 0;
    HiddenHook hidden_hook_;
};

} // namespace llm
