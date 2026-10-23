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

    // Speculative forward: batched like prefill, but returns the logits for ALL
    // `n` positions instead of just the last one. Required for speculative
    // decoding verification where we must check the argmax/distribution of
    // every drafted token against the target model.
    // Returns a pointer to an internal buffer of size (n * vocab_size).
    const float* spec_forward(const int64_t* tokens, int64_t n, int64_t start_pos);

    // Batched multi-sequence decode: one token for each of `n` independent
    // sequences. Row i is token tokens[i] at position pos[i] of its own cache
    // kvs[i] (all caches must share this model's layout and be distinct). Every
    // layer's weights are read once for all n rows, instead of once per
    // sequence. Returns n x vocab logits (row-major), owned by the Transformer
    // and valid until the next forward/forward_multi/prefill call. Row i equals
    // forward(tokens[i], pos[i]) run against kvs[i]: bit-identical for CPU
    // weights; weights offloaded to the GPU use the batched matmul kernel,
    // which can differ from the single-token kernel in rounding.
    const float* forward_multi(const int64_t* tokens, const int64_t* pos,
                               KVCache* const* kvs, int64_t n);

    // Point subsequent forward()/prefill() calls at another cache with the same
    // layout (used to prefill several independent sequences in one process).
    void set_kv(KVCache* kv) { kv_ = kv; }
    KVCache* kv() const { return kv_; }

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

    // ---- prefill tuning ---------------------------------------------------
    // Positions per batched block() call in prefill() (default 32, env
    // SIPLLM_PREFILL_BATCH), and the query-tiled prefill attention (default on,
    // env SIPLLM_TILED_ATTN=0 disables it). Neither changes results: the tiled
    // path is bit-identical to per-token attention.
    void set_prefill_batch(int64_t n) { prefill_batch_ = n < 1 ? 1 : (n > 1024 ? 1024 : n); }
    int64_t prefill_batch() const { return prefill_batch_; }
    void set_tiled_attention(bool on) { prefill_tiled_attn_ = on; }
    bool tiled_attention() const { return prefill_tiled_attn_; }

private:
    // arch dispatch on cfg_.block_spec. Rows are positions pos..pos+bs-1 of kv_,
    // unless row_pos/row_kv are given: then row i is position row_pos[i] of
    // cache row_kv[i] (independent sequences, see forward_multi).
    void block(int64_t layer, int64_t pos, int64_t bs = 1, float* x_in = nullptr,
               const int64_t* row_pos = nullptr, KVCache* const* row_kv = nullptr);

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
    std::vector<std::vector<float>> attn_scratch_;  // per-thread [group][span] scores + dequant row
    std::vector<float> attn_part_;                  // flash-decoding partials [item][group][m,l,out[hd]]
    std::vector<float> attn_out_;// q_dim
    std::vector<float> proj_;    // dim
    std::vector<float> hb_;      // ffn_dim
    std::vector<float> hb2_;     // ffn_dim
    std::vector<float> fused_;   // q_dim+2*kv_dim or 2*ffn_dim (Phi-3 fused proj)
    std::vector<float> router_;  // n_experts (MoE gating scores)
    std::vector<float> moe_;     // dim (MoE weighted expert accumulator)
    std::vector<float> logits_;  // vocab
    std::vector<float> spec_logits_; // n * vocab (for spec_forward)
    // Prefill-only: P residual streams (n x dim), the sole per-position state
    // that must survive the layer sweep. Sized on demand in prefill(); the only
    // buffer that grows with prompt length (tiny vs weights/KV — see RFC-007).
    std::vector<float> resid_;   // n*dim (batched prefill)
    // forward_multi: n residual streams, their final-normed copies, n x vocab logits.
    std::vector<float> multi_x_, multi_xb_, multi_logits_;

    int64_t prefill_batch_ = 32;
    bool prefill_tiled_attn_ = true;

    bool profiling_ = false;
    std::vector<LayerTiming> timings_;
    size_t peak_rss_ = 0;
    HiddenHook hidden_hook_;
};

} // namespace llm
