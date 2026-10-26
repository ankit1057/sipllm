#include "llm/runtime.h"
#include "llm/format.h"
#include "llm/gguf.h"
#include "llm/sip_ir_reader.h"
#include "llm/common.h"
#include "llm/neon.h"
#include "llm/mem_plan.h"
#include "llm/plugin.h"
#include "llm/semantic_cache.h"
#include "llm/spec_decoder.h"
#include "llm/session.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include <algorithm>
#include <thread>

namespace llm {

// RSS growth since `before`, clamped at 0 (another thread may free memory).
static size_t rss_growth(size_t before) {
    const size_t now = current_rss_bytes();
    return now > before ? now - before : 0;
}

std::unique_ptr<WeightSource> open_model(const std::string& path, bool use_mmap) {
    // Sniff the first 4 bytes.
    FileBacking probe(path, false);
    uint32_t magic = 0;
    probe.pread_exact(0, &magic, 4);
    const size_t rss0 = current_rss_bytes();
    std::unique_ptr<WeightSource> src;
    if (magic == kGGUFMagic) src = std::make_unique<GgufFile>(path, use_mmap);
    else if (magic == kLLMWMagic) src = std::make_unique<ModelFile>(path, use_mmap);
    else if (magic == kSipIRMagic) src = std::make_unique<SipIRReader>(path, use_mmap);
    else throw Error("open_model: unrecognized file magic in " + path);
    src->open_rss_bytes = rss_growth(rss0);
    return src;
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

    // Issue #70: the contiguous layer range this node runs (default: all). The
    // memory plan and the KV cache cover only those layers; a worker (range not
    // starting at 0) also keeps the LM head on disk, since it never uses it.
    layer_begin_ = opt_.layer_begin;
    layer_end_   = opt_.layer_end < 0 ? (int)cfg_.n_layers : opt_.layer_end;
    LLM_CHECK(layer_begin_ >= 0 && layer_begin_ < layer_end_ && layer_end_ <= (int)cfg_.n_layers,
              "runtime: invalid layer range");
    ModelConfig plan_cfg = cfg_;
    plan_cfg.n_layers = layer_end_ - layer_begin_;
    if (layer_begin_ > 0) opt_.stream_lm_head = true;

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
    // Attention keeps one max_ctx score buffer per pool worker (see Transformer).
    req.n_threads = threads > 0 ? threads : std::max(1, (int)std::thread::hardware_concurrency());

    // The tokenizer is built before planning so the RSS it costs (large for
    // 150k-vocab models) is measured into the ledger's base, not guessed. Only
    // what SipLLM adds counts: an embedding app's own memory is not the budget's.
    const size_t rss_before_tok = current_rss_bytes();
    tok_ = Tokenizer::from_source(*src_);
    req.measured_base = src_->open_rss_bytes + rss_growth(rss_before_tok);

    MemoryPlan plan = plan_memory(*src_, plan_cfg, req);

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

    // Waves 15-16 experiment: SIPLLM_FAST_LAYOUT=il4 interleaves resident Q4_K
    // and Q6_K matrices for the lane-per-row SDOT kernels (il4-q4: Q4_K only,
    // the wave-15 behaviour). Only with --fast on an SDOT core; otherwise
    // ignored, and every path is unchanged.
    if (const char* lay = std::getenv("SIPLLM_FAST_LAYOUT")) {
        const std::string l(lay);
        const bool ok = opt_.fast_quant && q4_K_interleave_supported();
        opt_.interleave_q4k = ok && (l == "il4" || l == "il4-q4");
        opt_.interleave_q6k = ok && l == "il4";
    }

    pool_ = std::make_unique<ThreadPool>(threads);
    opt_.dequant_pool = pool_.get();
    loader_ = std::make_unique<LayerLoader>(src_.get(), cfg_, opt_);
    if (!cfg_.layer_swa.empty() || !cfg_.layer_head_dim.empty() || !cfg_.layer_n_kv_heads.empty()) {
        // Per-layer geometry (Gemma 4): each layer's rows are exactly its kv
        // width, and sliding-window layers keep only a window-sized ring, so a
        // long session's KV grows with the few global layers alone.
        const int64_t n_local = layer_end_ - layer_begin_;
        std::vector<int64_t> dims((size_t)n_local), wins((size_t)n_local, 0);
        for (int64_t i = 0; i < n_local; ++i) {
            const int64_t l = layer_begin_ + i;
            dims[(size_t)i] = cfg_.kv_dim_at(l);
            if (!cfg_.layer_swa.empty() && cfg_.layer_swa[(size_t)l] && cfg_.sliding_window > 0)
                wins[(size_t)i] = cfg_.sliding_window;
        }
        kv_ = std::make_unique<KVCache>(dims, wins, ctx, kv_precision, layer_begin_);
    } else {
        kv_ = std::make_unique<KVCache>(layer_end_ - layer_begin_, cfg_.kv_dim(), ctx, kv_precision,
                                        layer_begin_);
    }
    // #81: the plan priced the KV at the full chosen ctx; allocate it once so
    // doubling never holds old+new buffers above that price.
    budgeted_ = ram_budget_total > 0;
    budget_forced_ = force_budget;
    if (budgeted_) kv_->reserve_full();
    tf_ = std::make_unique<Transformer>(loader_.get(), kv_.get(), pool_.get());
}

void Runtime::enable_semantic_cache(size_t max_bytes) {
    // The cache stores and re-injects FP32 rows for every layer; a Q8_0 KV row is
    // smaller than kv_dim floats, so reading it as float would run off the row.
    LLM_CHECK(kv_->precision() == KVPrecision::FP32,
              "--semantic-cache-budget needs the FP32 KV cache (drop --kv-q8)");
    LLM_CHECK(layer_begin_ == 0 && layer_end_ == (int)cfg_.n_layers,
              "--semantic-cache-budget is not supported on a LAN worker");
    // #81: the memory planner does not price the semantic cache, so it would sit
    // outside the --ram-budget ceiling.
    LLM_CHECK(!budgeted_ || budget_forced_,
              "--semantic-cache-budget is not counted by the --ram-budget planner; "
              "drop one of them, or pass --ram-budget-force");
    if (!semantic_cache_) {
        semantic_cache_ = std::make_unique<SemanticCache>(max_bytes, cfg_.n_layers, cfg_.kv_dim());
    }
}

void Runtime::enable_speculative_decoding(int max_draft_len) {
    if (!spec_decoder_) {
        spec_decoder_ = std::make_unique<PromptLookupDecoder>();
        max_draft_len_ = max_draft_len;
    }
}

std::string Runtime::generate(const std::string& prompt, int max_new,
                              SamplerConfig scfg, const TokenCallback& on_token,
                              GenStats* stats) {
    Sampler sampler(scfg);
    GenStats st;
    st.ctx_max = (int)kv_->max_ctx();
    if (tf_->has_remote_segments())
        LLM_CHECK(!spec_decoder_ && !semantic_cache_ && host_ == nullptr,
                  "--nodes: --spec, the semantic cache and plugins are not supported with remote layers");


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
        // A sliding-window ring may already have overwritten the history the
        // rewound tail would attend to; then recompute from scratch.
        if (!kv_->can_rewind(reuse_r, (int64_t)committed_.size())) reuse_r = 0;
        pos_ = reuse_r;
        committed_.resize((size_t)reuse_r);
        st.reused_prefix_tokens = (int)reuse_r;
    }

    // ---- semantic cache (Radix Tree prefix match) ----
    if (semantic_cache_) {
        std::vector<float> cached_k, cached_v;
        const int64_t matched = semantic_cache_->find_longest_prefix(prompt_ids, cached_k, cached_v);
        // Keep the last prompt token for prefill (it produces the logits). The
        // staged buffers are laid out [layer][matched][kv_dim], so the per-layer
        // stride is `matched` even when fewer rows are injected.
        int64_t hit_len = std::min(matched, (int64_t)prompt_ids.size() - 1);
        if (hit_len > reuse_r) {
            const int64_t stride = matched * cfg_.kv_dim();
            for (int64_t l = 0; l < cfg_.n_layers; ++l) {
                kv_->inject(l, cached_k.data() + l * stride, cached_v.data() + l * stride, hit_len);
            }
            pos_ = hit_len;
            committed_.assign(prompt_ids.begin(), prompt_ids.begin() + hit_len);
            reuse_r = hit_len;
            st.semantic_cache_hits = hit_len;
        }
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
        if (logit_filter_) {
            filtered_logits_.assign(logits, logits + vocab);
            logit_filter_(filtered_logits_.data(), vocab);
            logits = filtered_logits_.data();
        }
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
    for (int n = 0; n < max_new; ) {
        if (next < 0) break;
        if (tok_.is_eog(next)) break;
        if (pos_ >= kv_->max_ctx()) break;

        std::string piece = tok_.decode_token(next);
        output += piece;
        ++st.gen_tokens;
        if (on_token && !on_token(piece, next)) break;

        committed_.push_back(next);
        ++n;

        // ---- speculative decoding (prompt lookup; opt-in, default OFF) ------
        // When no draft is available this iteration is byte-identical to the
        // plain path below: one forward, one sample.
        std::vector<int64_t> draft;
        if (spec_decoder_ && n < max_new && pos_ + max_draft_len_ < kv_->max_ctx()) {
            st.spec_active = true;
            draft = spec_decoder_->draft(committed_, std::min(max_draft_len_, max_new - n));
        }

        if (draft.empty()) {
            const float* logits = tf_->forward(next, pos_);
            ++pos_;
            if (rtk) rtk->on_step(kv_view());
            if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());
            if (logit_filter_) {
                filtered_logits_.assign(logits, logits + vocab);
                logit_filter_(filtered_logits_.data(), vocab);
                logits = filtered_logits_.data();
            }
            next = sampler.sample(logits, vocab);
        } else {
            // One batched verify pass over [next, draft...] streams every layer
            // ONCE for K+1 positions. Step i's logits are the model's prediction
            // after consuming batch[i]; draft[i] is accepted iff the sampler
            // would have produced it there, so every emitted token is still
            // sampled from the model's own distribution given its true prefix.
            const int K = (int)draft.size();
            std::vector<int64_t> batch;
            batch.reserve(K + 1);
            batch.push_back(next);
            batch.insert(batch.end(), draft.begin(), draft.end());
            ++st.spec_batches;
            st.spec_drafted += K;

            const float* spec_logits = tf_->spec_forward(batch.data(), K + 1, pos_);
            if (rtk) rtk->on_step(kv_view());
            if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());

            for (int i = 0; i <= K; ++i) {
                const float* step_logits = spec_logits + (size_t)i * vocab;
                if (logit_filter_) {   // same filter as the plain path, per step
                    filtered_logits_.assign(step_logits, step_logits + vocab);
                    logit_filter_(filtered_logits_.data(), vocab);
                    step_logits = filtered_logits_.data();
                }
                int64_t sampled = sampler.sample(step_logits, vocab);
                ++pos_;   // batch[i] is now consumed at position pos_-1

                if (i == K) {
                    // Every draft token verified; `sampled` is the prediction
                    // for the next step, exactly as the plain path would have.
                    next = sampled;
                    break;
                }
                if (sampled != draft[i]) {
                    // Reject: KV written for draft[i..] is stale beyond pos_.
                    kv_->set_seq_len(pos_);
                    next = sampled;
                    break;
                }
                // Accept draft[i]. Mirror the top-of-loop emission exactly,
                // including the end-of-generation and context checks.
                ++st.spec_accepted;
                if (tok_.is_eog(draft[i]) || n >= max_new) { next = -1; break; }
                piece = tok_.decode_token(draft[i]);
                output += piece;
                ++st.gen_tokens;
                if (on_token && !on_token(piece, draft[i])) { next = -1; break; }
                committed_.push_back(draft[i]);
                ++n;
            }
        }
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
    if (semantic_cache_ && pos_ > 0) {
        auto k_fn = [&](int64_t l, int64_t p) -> const float* { return kv_->k(l, p); };
        auto v_fn = [&](int64_t l, int64_t p) -> const float* { return kv_->v(l, p); };
        semantic_cache_->commit(committed_, k_fn, v_fn);
    }
    if (stats) *stats = st;
    return output;
}

// FNV-1a over the model's identity (arch + shape). Guards session load against
// a mismatched model; it does NOT detect same-config-but-different-weights
// (that is the caller's responsibility — a session pairs with its model).
static uint64_t session_model_id(const ModelConfig& c, KVPrecision kv_precision) {
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
    // Rows differ in size per precision. FP32 mixes nothing, so FP32 sessions
    // saved by earlier builds still load.
    if (kv_precision == KVPrecision::Q8_0) mix(0x51385f30ULL);   // "Q8_0"
    return h;
}

void Runtime::set_remote_segments(std::vector<RemoteSegment> segs) {
    // Features whose state spans every layer cannot work with a split model.
    LLM_CHECK(!spec_decoder_, "--nodes: speculative decoding (--spec) is not supported with remote layers");
    LLM_CHECK(!semantic_cache_, "--nodes: the semantic cache is not supported with remote layers");
    LLM_CHECK(host_ == nullptr, "--nodes: plugins (--kosh/--rtk) are not supported with remote layers");
    LLM_CHECK(layer_begin_ == 0, "--nodes: the coordinator must own layers [0, a)");
    int64_t expect = layer_end_;
    for (const RemoteSegment& r : segs) {
        LLM_CHECK(r.begin == expect, "--nodes: worker ranges must start where the previous one ends "
                                     "(coordinator owns [0, a), workers cover [a, n_layers) in order)");
        expect = r.end;
    }
    LLM_CHECK(expect == cfg_.n_layers, "--nodes: worker ranges must end at the model's last layer ("
                                       + std::to_string(cfg_.n_layers) + ")");
    tf_->set_remote_segments(std::move(segs));
}

void Runtime::run_layers(int64_t begin, int64_t end, float* resid, int64_t n, int64_t start_pos) {
    LLM_CHECK(begin == layer_begin_ && end == layer_end_, "run_layers: range is not this node's");
    tf_->run_layers(begin, end, resid, n, start_pos);
}

bool Runtime::save_session(const std::string& path) const {
    if (tf_->has_remote_segments()) {   // the KV is split across machines
        fprintf(stderr, "sessions are not supported with remote layers (--nodes)\n");
        return false;
    }
    return session_write(path, committed_, *kv_, (int64_t)committed_.size(),
                         session_model_id(cfg_, kv_->precision()));
}

bool Runtime::load_session(const std::string& path) {
    if (tf_->has_remote_segments()) {
        fprintf(stderr, "sessions are not supported with remote layers (--nodes)\n");
        return false;
    }
    std::vector<int64_t> toks;
    int64_t sl = 0;
    if (!session_read(path, toks, *kv_, &sl, session_model_id(cfg_, kv_->precision()))) {
        reset();   // leave the runtime clean and usable
        return false;
    }
    committed_ = std::move(toks);
    pos_ = sl;
    return true;
}

} // namespace llm
