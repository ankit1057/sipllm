// runtime.h — end-to-end generation: prompt prefill + token-by-token decode,
// with the throughput/latency stats the web GUI displays.
//
// Owns the whole stack: WeightSource -> LayerLoader -> KVCache -> Transformer,
// plus Tokenizer and Sampler. open_model() sniffs GGUF vs .llmw by magic.
#pragma once

#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/model.h"
#include "llm/sampler.h"
#include "llm/tokenizer.h"
#include "llm/semantic_cache.h"
#include "llm/spec_decoder.h"
#include "llm/transformer.h"
#include "llm/weight_source.h"

#include <functional>
#include <memory>
#include <string>

namespace llm {

// Optional plugin seam: Kosh (context intelligence) + RTK (runtime/KV
// intelligence). Forward-declared so runtime.h stays light; the host is owned
// by the caller and installed via Runtime::set_plugins().
class PluginHost;

// Open a model file by content sniffing (GGUF or .llmw). Returns a WeightSource.
std::unique_ptr<WeightSource> open_model(const std::string& path, bool use_mmap = false);

struct GenStats {
    int    prompt_tokens = 0;
    int    gen_tokens = 0;
    double load_s = 0;          // model open + config
    double ttft_s = 0;          // time to first generated token (incl. prefill)
    double prefill_s = 0;       // prompt processing time
    double decode_s = 0;        // time generating the rest
    double prefill_tok_s = 0;   // prompt tokens / prefill_s
    double decode_tok_s = 0;    // gen tokens / decode_s
    size_t weights_resident_bytes = 0;
    int    pinned_layers = 0;   // #37: layers pinned resident under --ram-budget
    int    gpu_layers = 0;      // --gpu-layers: layers whose projections ran on CUDA
    bool   gpu_output = false;  // LM head on CUDA
    size_t gpu_bytes = 0;       // device bytes of placed weights
    size_t kv_bytes = 0;
    uint64_t bytes_read = 0;    // total streamed from disk
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_misses = 0;
    int    ctx_used = 0;
    int    ctx_max = 0;
    // Plugin seam accounting. Populated only when a PluginHost with the
    // respective plugin is installed; all zero/false on the default path.
    bool     kosh_active = false;
    int64_t  kosh_tokens_in = 0;
    int64_t  kosh_tokens_out = 0;
    // Speculative decoding (prompt-lookup, opt-in via --spec N). All zero/false
    // on the default path.
    bool     spec_active = false;
    int64_t  spec_batches = 0;   // verify passes that carried >= 1 draft token
    int64_t  spec_drafted = 0;   // draft tokens proposed
    int64_t  spec_accepted = 0;  // draft tokens accepted (= model-free tokens)
    bool     rtk_active = false;
    uint64_t rtk_steps = 0;
    int64_t  rtk_max_seq = 0;
    size_t   rtk_peak_kv_bytes = 0;
    // Context-reuse accounting (opt-in; zero on the default path).
    bool reuse_active = false;
    int  reused_prefix_tokens = 0;
    int  processed_tokens = 0;
    uint64_t semantic_cache_hits = 0;
};

class Runtime {
public:
    // Takes ownership of the source. opt controls residency/prefetch/mmap.
    // ram_budget_total (bytes, 0 = unlimited) is a TOTAL peak-RSS target; the
    // runtime calculates how many layers can be resident alongside the KV cache,
    // setting loader capacity. If force_budget is true, throws if the minimum
    // working set exceeds the budget.
    Runtime(std::unique_ptr<WeightSource> src, LayerLoader::Options opt,
            int max_ctx = 0, int threads = 0, size_t ram_budget_total = 0, bool force_budget = false, KVPrecision kv_precision = KVPrecision::FP32);

    const ModelConfig& config() const { return cfg_; }
    const Tokenizer&   tokenizer() const { return tok_; }
    size_t weights_resident_bytes() const { return loader_->resident_bytes(); }
    int interleaved_matrices() const { return loader_->interleaved_matrices(); }   // wave 15
    int interleaved_q6k_matrices() const { return loader_->interleaved_q6k_matrices(); }   // wave 16
    size_t kv_bytes() const { return kv_->bytes(); }
    ThreadPool* thread_pool() { return pool_.get(); }

    // Streaming callback: called with each newly produced piece of text and the
    // token id. Return false to stop early. `on_token` may be null.
    using TokenCallback = std::function<bool(const std::string& piece, int64_t id)>;

    // Generate up to max_new tokens continuing `prompt`. Fills `stats`.
    std::string generate(const std::string& prompt, int max_new,
                         SamplerConfig scfg, const TokenCallback& on_token,
                         GenStats* stats);

    // Per-token profiling sink: when set, each decode step reports its
    // per-layer timing breakdown and peak RSS (for the live visualization).
    using ProfileSink = std::function<void(int token_index,
                                           const std::vector<Transformer::LayerTiming>&,
                                           size_t peak_rss)>;
    void set_profile_sink(ProfileSink sink) { profile_sink_ = std::move(sink); }
    const std::vector<Transformer::LayerTiming>& last_timings() const {
        return tf_->last_timings();
    }
    size_t peak_rss() const { return tf_->peak_rss(); }

    // Golden-test hooks.
    void set_hidden_hook(Transformer::HiddenHook h) { tf_->set_hidden_hook(std::move(h)); }
    const std::vector<float>& first_logits() const { return first_logits_; }

    // Reset conversation state (KV cache).
    void reset() { kv_->clear(); pos_ = 0; committed_.clear(); }

    // Install an optional plugin host (non-owning). Null (default) => generate()
    // is byte-identical to the plugin-free engine. The host is init()'d lazily
    // on the first generate() with the live KV view.
    void set_plugins(PluginHost* host) { host_ = host; }

    // Enable cross-turn context reuse: reuse the KV of the longest common
    // prefix with the previously committed tokens, reprocessing only the tail.
    // Opt-in (default off => byte-identical). Set before the first generate().
    void set_context_reuse(bool on) { reuse_ = on; }

    // Persist / restore the committed context (tokens + KV) to a session file
    // so cross-turn reuse survives a process restart (Phase 5). Opt-in.
    // load_session resets the runtime and returns false on any mismatch
    // (magic/version/model-id/dims) or IO error — a session is only valid for
    // the model that wrote it.
    // Enable Semantic Cache (Radix Tree + memcpy injection)
    void enable_semantic_cache(size_t max_bytes);
    
    // Enable Speculative Decoding (Prompt Lookup)
    void enable_speculative_decoding(int max_draft_len = 3);
    bool speculative_enabled() const { return spec_decoder_ != nullptr; }

    bool save_session(const std::string& path) const;
    bool load_session(const std::string& path);

    // ---- Issue #70: LAN pooling ---------------------------------------------
    // Layer range this runtime owns (LayerLoader::Options::layer_begin/end;
    // default all layers). Plan, pinning and KV cover only these layers.
    int layer_begin() const { return layer_begin_; }
    int layer_end() const { return layer_end_; }
    const WeightSource& source() const { return *src_; }
    int64_t max_ctx() const { return kv_->max_ctx(); }
    KVPrecision kv_precision() const { return kv_->precision(); }
    Residency residency() const { return opt_.residency; }
    // Coordinator: layers [layer_end(), n_layers) run on other machines.
    // Refuses speculative decoding, the semantic cache, plugins and sessions.
    void set_remote_segments(std::vector<RemoteSegment> segs);
    bool has_remote_layers() const { return tf_->has_remote_segments(); }
    // Worker: run exactly this runtime's layers over n residual rows.
    void run_layers(int64_t begin, int64_t end, float* resid, int64_t n, int64_t start_pos);

private:
    std::unique_ptr<WeightSource> src_;
    ModelConfig cfg_;
    LayerLoader::Options opt_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<LayerLoader> loader_;
    std::unique_ptr<KVCache> kv_;
    std::unique_ptr<Transformer> tf_;
    Tokenizer tok_;
    int64_t pos_ = 0;   // absolute position in the KV cache
    ProfileSink profile_sink_;
    std::vector<float> first_logits_;   // logits at first prediction (golden)
    PluginHost* host_ = nullptr;   // optional, non-owning plugin seam
    bool reuse_ = false;                 // cross-turn context reuse (opt-in)
    std::vector<int64_t> committed_;     // mirrors tokens in KV [0,pos_) (for reuse + session save)
    std::unique_ptr<class SemanticCache> semantic_cache_;
    std::unique_ptr<class SpecDecoder> spec_decoder_;
    int max_draft_len_ = 0;
    int layer_begin_ = 0, layer_end_ = 0;   // #70: this node's layers
    bool budgeted_ = false;        // #81: a --ram-budget plan governs this runtime
    bool budget_forced_ = false;   // #81: ... and the operator forced it (--ram-budget-force)
};

} // namespace llm
