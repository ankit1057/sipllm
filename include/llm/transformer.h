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

    // Single-pass batched prefill (RFC-007). Runs the whole prompt through the
    // model streaming every layer EXACTLY ONCE (vs `forward` which streams all
    // layers once PER token, i.e. P full-model streams for a P-token prompt).
    // For each layer: project/attend/FFN all `n` positions [start_pos,
    // start_pos+n) while the block is resident, then unload. Positions are swept
    // in ascending order so position t's K/V is written before any later
    // position attends to it -> causality (and the resulting KV cache + logits)
    // is IDENTICAL to calling forward() for each token in turn. Returns the
    // logits of the LAST position (the only ones decode needs), owned by the
    // Transformer and valid until the next forward/prefill call.
    //
    // Every architecture on the shared block() path is batched safely: no block
    // carries cross-position state beyond the residual stream (member `x_`) and
    // the KV cache, so there is no per-arch fallback.
    const float* prefill(const int64_t* tokens, int64_t n, int64_t start_pos);

    // RoPE is exposed for testing. Optional llama3 frequency scaling: when
    // `rs.llama3` is set the per-wavelength stretch (issue #9) is applied;
    // default-constructed => plain RoPE, identical to pre-#9 behavior.
    struct RopeScaling {
        bool  llama3 = false;
        float factor = 8.f;
        float low_freq_factor = 1.f;
        float high_freq_factor = 4.f;
        float orig_ctx_len = 8192.f;
    };
    static void apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                           int64_t pos, float theta_base);
    static void apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                           int64_t pos, float theta_base, const RopeScaling& rs);

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
    void block(int64_t layer, int64_t pos);        // arch dispatch on cfg_.arch_kind
    void block_llama(int64_t layer, int64_t pos);  // reference: RMSNorm+RoPE+GQA+SwiGLU
    void block_gemma2(int64_t layer, int64_t pos); // GeGLU + pre/post (1+w) norms + softcap
    void block_phi3(int64_t layer, int64_t pos);   // fused QKV + fused gate/up + partial RoPE
    void block_moe(int64_t layer, int64_t pos);    // Mixtral: router + top-k expert FFNs
    void block_gpt2(int64_t layer, int64_t pos);   // LayerNorm + learned pos + GELU MLP
    void block_phi2(int64_t layer, int64_t pos);   // parallel LayerNorm block + partial RoPE
    void attention_llama(int64_t layer, int64_t pos); // shared attention sublayer

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
    std::vector<float> fused_;   // q_dim+2*kv_dim or 2*ffn_dim (Phi-3 fused proj)
    std::vector<float> router_;  // n_experts (MoE gating scores)
    std::vector<float> moe_;     // dim (MoE weighted expert accumulator)
    std::vector<float> logits_;  // vocab
    // Prefill-only: P residual streams (n x dim), the sole per-position state
    // that must survive the layer sweep. Sized on demand in prefill(); the only
    // buffer that grows with prompt length (tiny vs weights/KV — see RFC-007).
    std::vector<float> resid_;   // n*dim (batched prefill)

    bool profiling_ = false;
    std::vector<LayerTiming> timings_;
    size_t peak_rss_ = 0;
    HiddenHook hidden_hook_;
};

} // namespace llm
