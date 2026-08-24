#include "llm/runtime.h"
#include "llm/format.h"
#include "llm/gguf.h"
#include "llm/sip_ir_reader.h"
#include "llm/common.h"
#include "llm/neon.h"
#include "llm/mem_plan.h"
#include "llm/plugin.h"
#include "llm/session.h"

#include <cstdio>

#include <algorithm>

namespace llm {

std::unique_ptr<WeightSource> open_model(const std::string& path, bool use_mmap) {
    // Sniff the first 4 bytes.
    FileBacking probe(path, false);
    uint32_t magic = 0;
    probe.pread_exact(0, &magic, 4);
    if (magic == kGGUFMagic) return std::make_unique<GgufFile>(path, use_mmap);
    if (magic == kLLMWMagic) return std::make_unique<ModelFile>(path, use_mmap);
    if (magic == kSipIRMagic) return std::make_unique<SipIRReader>(path, use_mmap);
    throw Error("open_model: unrecognized file magic in " + path);
}

// #37: conservative RAM allowance for everything that is NOT streamed weights or
// KV — transformer scratch (residual/qkv/ffn/logits), attention scores (~ctx),
// plus code/allocator slop. Keeps the total peak-RSS ceiling honest.
static size_t runtime_reserve_bytes(const ModelConfig& c, int ctx) {
    const size_t scratch = (size_t)(c.dim * 8 + c.ffn_dim * 2 + c.q_dim()
                         + 2 * c.kv_dim() + c.vocab_size + (int64_t)ctx) * sizeof(float);
    return (size_t)24 * 1024 * 1024 + scratch;   // 24 MB base slop + scratch
}

Runtime::Runtime(std::unique_ptr<WeightSource> src, LayerLoader::Options opt,
                 int max_ctx, int threads, size_t ram_budget_total, bool force_budget, KVPrecision kv_precision)
    : src_(std::move(src)), opt_(opt) {
    cfg_ = ModelConfig::from_source(*src_);
    set_fast_quant(opt_.fast_quant);   // #demo: opt-in int8 SDOT for Q8_0 (--fast)
    LLM_CHECK(cfg_.n_layers > 0 && cfg_.dim > 0, "runtime: invalid model config");

    BudgetRequest req;
    req.budget_bytes = ram_budget_total;
    req.ctx_req = max_ctx;
    req.ctx_explicit = (max_ctx > 0);
    req.n_buffers_req = opt_.n_buffers;
    req.async_req = opt_.async;
    req.residency = opt_.residency;
    req.stream_head_req = opt_.stream_lm_head;
    req.force = force_budget;
    req.default_ctx_cap = 4096;
    req.kv_precision = kv_precision;

    MemoryPlan plan = plan_memory(*src_, cfg_, req);

    if (ram_budget_total > 0) {
        fprintf(stderr, "\n%s\n", plan.report.c_str());
        if (!plan.feasible && !plan.overridden) {
            throw Error("RAM budget impossible. Use --ram-budget-force to run anyway.");
        }
    }

    int ctx = plan.ctx;
    opt_.stream_lm_head = plan.stream_lm_head;
    opt_.n_buffers = plan.n_buffers;
    opt_.ram_budget_bytes = plan.weight_ceiling;

    pool_ = std::make_unique<ThreadPool>(threads);
    opt_.dequant_pool = pool_.get();
    loader_ = std::make_unique<LayerLoader>(src_.get(), cfg_, opt_);
    kv_ = std::make_unique<KVCache>(cfg_.n_layers, cfg_.kv_dim(), ctx, kv_precision);
    tf_ = std::make_unique<Transformer>(loader_.get(), kv_.get(), pool_.get());
    tok_ = Tokenizer::from_source(*src_);
}

std::string Runtime::generate(const std::string& prompt, int max_new,
                              SamplerConfig scfg, const TokenCallback& on_token,
                              GenStats* stats) {
    Sampler sampler(scfg);
    GenStats st;
    st.ctx_max = (int)kv_->max_ctx();


    // ---- plugin seam setup (Kosh #52 context, RTK #53 runtime/KV) ----------
    // A read-only KV snapshot for RTK; expressed as RtkKvView so the contract
    // survives a future distributed/persisted KV backing.
    auto kv_view = [&]() -> RtkKvView {
        RtkKvView v;
        v.n_layers = kv_->n_layers();
        v.kv_dim   = kv_->kv_dim();
        v.seq_len  = pos_;
        v.capacity = kv_->capacity();
        v.max_ctx  = kv_->max_ctx();
        v.bytes    = kv_->bytes();
        return v;
    };
    if (host_ && !host_->inited()) host_->init(cfg_, kv_view());
    RtkPlugin* rtk = host_ ? host_->rtk() : nullptr;
    if (rtk) st.rtk_active = true;
    // Tokenize (add BOS only at the very start of a fresh context).
    bool fresh = reuse_ ? true : (pos_ == 0);  // reuse expects full-context resends
    std::vector<int64_t> prompt_ids = tok_.encode(prompt, /*add_bos=*/fresh);
    st.prompt_tokens = (int)prompt_ids.size();

    std::string output;
    const int64_t vocab = cfg_.vocab_size;

    // ---- Kosh seam (context intelligence, #52) ----------------------------
    // Optimize the token stream before prefill. Guarded: any failure or an
    // invalid result falls back to the untouched stream, so the default
    // (no-Kosh) path is byte-identical.
    if (host_ && host_->kosh()) {
        KoshRequest kr;
        kr.cfg       = &cfg_;
        kr.prompt    = prompt;
        kr.tokens    = prompt_ids;
        kr.start_pos = pos_;
        kr.add_bos   = fresh;
        KoshResult kres;
        bool ok = false;
        try { kres = host_->kosh()->optimize(kr); ok = true; }
        catch (const std::exception& e) {
            fprintf(stderr, "[plugin] kosh optimize threw: %s — fallback\n", e.what());
        } catch (...) {
            fprintf(stderr, "[plugin] kosh optimize threw (unknown) — fallback\n");
        }
        // Validate: non-empty, fits the remaining context, ids in vocab range.
        bool valid = ok && !kres.tokens.empty()
                     && (int64_t)kres.tokens.size() <= kv_->max_ctx() - pos_;
        if (valid)
            for (int64_t id : kres.tokens)
                if (id < 0 || id >= vocab) { valid = false; break; }
        if (valid) {
            st.kosh_active     = true;
            st.kosh_tokens_in  = kres.tokens_in;
            st.kosh_tokens_out = kres.tokens_out;
            prompt_ids         = std::move(kres.tokens);
            st.prompt_tokens   = (int)prompt_ids.size();
        }
    }

    // ---- context reuse (Kosh #52 x RTK #53) --------------------------------
    // Reuse the KV of the longest common prefix already committed, reprocessing
    // only the changed tail. Opt-in; when off, reuse_r == 0 and the prefill below
    // is byte-identical to the plugin-free engine.
    st.reuse_active = reuse_;
    int64_t reuse_r = 0;
    if (reuse_) {
        const int64_t maxr = std::min((int64_t)prompt_ids.size(),
                                      (int64_t)committed_.size());
        while (reuse_r < maxr && prompt_ids[reuse_r] == committed_[reuse_r]) ++reuse_r;
        // Always reprocess >= 1 token so we obtain last-position logits.
        if (reuse_r > (int64_t)prompt_ids.size() - 1)
            reuse_r = (int64_t)prompt_ids.size() - 1;
        if (reuse_r < 0) reuse_r = 0;
        pos_ = reuse_r;
        committed_.resize((size_t)reuse_r);
        st.reused_prefix_tokens = (int)reuse_r;
    }
    st.processed_tokens = (int)((int64_t)prompt_ids.size() - reuse_r);

    // ---- prefill (RFC-007: single-pass batched) ----
    // One batched sweep streams the whole model ONCE for the entire prompt,
    // instead of the old loop that called tf_->forward() per token — which
    // re-streamed every layer for every prompt token (P full-model streams).
    double t_start = now_sec();
    int64_t next = -1;
    if ((int64_t)prompt_ids.size() > reuse_r) {
        const int64_t n_delta = (int64_t)prompt_ids.size() - reuse_r;
        LLM_CHECK(pos_ + n_delta - 1 < kv_->max_ctx(),
                  "context window exceeded during prefill");
        const float* logits = tf_->prefill(prompt_ids.data() + reuse_r, n_delta, pos_);
        pos_ += n_delta;
        for (int64_t id : prompt_ids) sampler.accept(id);  // full history for repetition
        committed_.insert(committed_.end(),
                          prompt_ids.begin() + reuse_r, prompt_ids.end());
        first_logits_.assign(logits, logits + vocab);
        next = sampler.sample(logits, vocab);
    }
    if (rtk) rtk->on_prefill(kv_view());
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
        committed_.push_back(next);
        ++pos_;
        if (rtk) rtk->on_step(kv_view());
        if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());
        next = sampler.sample(logits, vocab);
    }
    if (profile_sink_) tf_->enable_profiling(false);
    double t_end = now_sec();
    st.decode_s = t_end - t_decode_start;
    st.decode_tok_s = st.decode_s > 0 ? st.gen_tokens / st.decode_s : 0;

    st.weights_resident_bytes = loader_->resident_bytes();
    st.pinned_layers = loader_->pinned_layers();
    st.kv_bytes = kv_->bytes();
    st.bytes_read = loader_->stats().bytes_read.load();
    st.prefetch_hits = loader_->stats().prefetch_hits.load();
    st.prefetch_misses = loader_->stats().prefetch_misses.load();
    st.ctx_used = (int)pos_;
    if (rtk) {
        RtkMetrics rm = rtk->metrics();
        st.rtk_steps         = rm.steps;
        st.rtk_max_seq       = rm.max_seq;
        st.rtk_peak_kv_bytes = rm.peak_kv_bytes;
    }
    if (stats) *stats = st;
    return output;
}

// FNV-1a over the model's identity (arch + shape). Guards session load against
// a mismatched model; it does NOT detect same-config-but-different-weights
// (that is the caller's responsibility — a session pairs with its model).
static uint64_t session_model_id(const ModelConfig& c) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t x) { h ^= x; h *= 1099511628211ULL; };
    for (unsigned char ch : c.arch) mix(ch);
    mix((uint64_t)c.n_layers);
    mix((uint64_t)c.n_heads);
    mix((uint64_t)c.n_kv_heads);
    mix((uint64_t)c.dim);
    mix((uint64_t)c.head_dim);
    mix((uint64_t)c.ffn_dim);
    mix((uint64_t)c.vocab_size);
    return h;
}

bool Runtime::save_session(const std::string& path) const {
    return session_write(path, committed_, *kv_, (int64_t)committed_.size(),
                         session_model_id(cfg_));
}

bool Runtime::load_session(const std::string& path) {
    std::vector<int64_t> toks;
    int64_t sl = 0;
    if (!session_read(path, toks, *kv_, &sl, session_model_id(cfg_))) {
        reset();   // leave the runtime clean and usable
        return false;
    }
    committed_ = std::move(toks);
    pos_ = sl;
    return true;
}

} // namespace llm
