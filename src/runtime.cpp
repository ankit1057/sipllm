// runtime.cpp — see runtime.h.
#include "llm/runtime.h"
#include "llm/format.h"
#include "llm/gguf.h"

#include <algorithm>

namespace llm {

std::unique_ptr<WeightSource> open_model(const std::string& path, bool use_mmap) {
    // Sniff the first 4 bytes.
    FileBacking probe(path, false);
    uint32_t magic = 0;
    probe.pread_exact(0, &magic, 4);
    if (magic == kGGUFMagic) return std::make_unique<GgufFile>(path, use_mmap);
    if (magic == kLLMWMagic) return std::make_unique<ModelFile>(path, use_mmap);
    throw Error("open_model: unrecognized file magic in " + path);
}

Runtime::Runtime(std::unique_ptr<WeightSource> src, LayerLoader::Options opt,
                 int max_ctx, int threads)
    : src_(std::move(src)), opt_(opt) {
    cfg_ = ModelConfig::from_source(*src_);
    LLM_CHECK(cfg_.n_layers > 0 && cfg_.dim > 0, "runtime: invalid model config");

    // Cap the DEFAULT context window. Modern models advertise huge windows
    // (Llama 3.2 = 131072), and the KV cache is allocated for the full window up
    // front — at 131072 that's multiple GB, an instant OOM on an edge device.
    // So unless the caller asks for a specific --ctx, use the model's window
    // clamped to a modest edge-friendly default. Power users raise it explicitly.
    constexpr int64_t kDefaultMaxCtx = 4096;
    int ctx;
    if (max_ctx > 0)              ctx = max_ctx;                       // explicit --ctx
    else if (cfg_.ctx_len > 0)    ctx = (int)std::min<int64_t>(cfg_.ctx_len, kDefaultMaxCtx);
    else                          ctx = 2048;

    pool_ = std::make_unique<ThreadPool>(threads);
    opt_.dequant_pool = pool_.get();
    loader_ = std::make_unique<LayerLoader>(src_.get(), cfg_, opt_);
    kv_ = std::make_unique<KVCache>(cfg_.n_layers, cfg_.kv_dim(), ctx);
    tf_ = std::make_unique<Transformer>(loader_.get(), kv_.get(), pool_.get());
    tok_ = Tokenizer::from_source(*src_);
}

std::string Runtime::generate(const std::string& prompt, int max_new,
                              SamplerConfig scfg, const TokenCallback& on_token,
                              GenStats* stats) {
    Sampler sampler(scfg);
    GenStats st;
    st.ctx_max = (int)kv_->max_ctx();

    // Tokenize (add BOS only at the very start of a fresh context).
    bool fresh = (pos_ == 0);
    std::vector<int64_t> prompt_ids = tok_.encode(prompt, /*add_bos=*/fresh);
    st.prompt_tokens = (int)prompt_ids.size();

    std::string output;
    const int64_t vocab = cfg_.vocab_size;

    // ---- prefill (RFC-007: single-pass batched) ----
    // One batched sweep streams the whole model ONCE for the entire prompt,
    // instead of the old loop that called tf_->forward() per token — which
    // re-streamed every layer for every prompt token (P full-model streams).
    double t_start = now_sec();
    int64_t next = -1;
    if (!prompt_ids.empty()) {
        LLM_CHECK(pos_ + (int64_t)prompt_ids.size() - 1 < kv_->max_ctx(),
                  "context window exceeded during prefill");
        const float* logits = tf_->prefill(prompt_ids.data(),
                                            (int64_t)prompt_ids.size(), pos_);
        pos_ += (int64_t)prompt_ids.size();
        for (int64_t id : prompt_ids) sampler.accept(id);  // seed repetition history
        first_logits_.assign(logits, logits + vocab);
        next = sampler.sample(logits, vocab);
    }
    double t_prefill_done = now_sec();
    st.prefill_s = t_prefill_done - t_start;
    st.ttft_s = st.prefill_s;   // first token emerges right after prefill
    st.prefill_tok_s = st.prefill_s > 0 ? st.prompt_tokens / st.prefill_s : 0;

    // ---- decode ----
    if (profile_sink_) tf_->enable_profiling(true);
    double t_decode_start = now_sec();
    for (int n = 0; n < max_new; ++n) {
        if (next < 0) break;
        if (tok_.is_eog(next)) break;
        if (pos_ >= kv_->max_ctx()) break;

        std::string piece = tok_.decode_token(next);
        output += piece;
        ++st.gen_tokens;
        if (on_token && !on_token(piece, next)) break;

        const float* logits = tf_->forward(next, pos_);
        ++pos_;
        if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());
        next = sampler.sample(logits, vocab);
    }
    if (profile_sink_) tf_->enable_profiling(false);
    double t_end = now_sec();
    st.decode_s = t_end - t_decode_start;
    st.decode_tok_s = st.decode_s > 0 ? st.gen_tokens / st.decode_s : 0;

    st.weights_resident_bytes = loader_->resident_bytes();
    st.kv_bytes = kv_->bytes();
    st.bytes_read = loader_->stats().bytes_read.load();
    st.prefetch_hits = loader_->stats().prefetch_hits.load();
    st.prefetch_misses = loader_->stats().prefetch_misses.load();
    st.ctx_used = (int)pos_;
    if (stats) *stats = st;
    return output;
}

} // namespace llm
