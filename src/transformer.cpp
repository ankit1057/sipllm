// transformer.cpp — decoder forward pass (see transformer.h).
#include "llm/transformer.h"
#include "llm/linear.h"
#include "llm/ops.h"
#include "llm/simd.h"

#include <cmath>

namespace llm {

Transformer::Transformer(LayerLoader* loader, KVCache* kv, ThreadPool* pool)
    : loader_(loader), kv_(kv), pool_(pool), cfg_(loader->config()) {
    x_.assign(cfg_.dim, 0.f);
    xb_.assign(cfg_.dim, 0.f);
    q_.assign(cfg_.q_dim(), 0.f);
    k_.assign(cfg_.kv_dim(), 0.f);
    v_.assign(cfg_.kv_dim(), 0.f);
    att_.assign(kv_->max_ctx(), 0.f);
    attn_out_.assign(cfg_.q_dim(), 0.f);
    proj_.assign(cfg_.dim, 0.f);
    hb_.assign(cfg_.ffn_dim, 0.f);
    hb2_.assign(cfg_.ffn_dim, 0.f);
    logits_.assign(cfg_.vocab_size, 0.f);
}

// Rotary embedding on adjacent element pairs (ggml "rope_norm"). GGUF Llama
// weights are permuted at conversion time so that this adjacent-pair rotation
// reproduces HF's rotate_half — so we apply exactly this to both q and k.
void Transformer::apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                             int64_t pos, float theta_base) {
    const int64_t half = head_dim / 2;
    for (int64_t h = 0; h < n_heads; ++h) {
        float* p = vec + h * head_dim;
        for (int64_t i = 0; i < half; ++i) {
            float freq = std::pow(theta_base, -2.0f * (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = p[2 * i], x1 = p[2 * i + 1];
            p[2 * i]     = x0 * c - x1 * s;
            p[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

// Dispatch one block to the per-architecture implementation. Every arch except
// the Llama reference is added as a new case here (issues #10-#16); Unknown
// architectures fall through to the Llama path unchanged.
void Transformer::block(int64_t layer, int64_t pos) {
    switch (cfg_.arch_kind) {
        case Arch::Llama:
        case Arch::Unknown:
        default:
            block_llama(layer, pos);
            return;
    }
}

void Transformer::block_llama(int64_t layer, int64_t pos) {
    const int64_t dim = cfg_.dim;
    const int64_t hd = cfg_.head_dim;
    const int64_t n_heads = cfg_.n_heads;
    const int64_t n_kv = cfg_.n_kv_heads;
    const int64_t kv_dim = cfg_.kv_dim();
    const int64_t group = cfg_.gqa_group();
    const float scale = 1.0f / std::sqrt((float)hd);

    // --- attention ---
    WeightRef an = loader_->getWeight(Role::AttnNorm);
    rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(an.data), dim, cfg_.rms_eps);

    linear(q_.data(), loader_->getWeight(Role::AttnQ), xb_.data(), pool_);
    linear(k_.data(), loader_->getWeight(Role::AttnK), xb_.data(), pool_);
    linear(v_.data(), loader_->getWeight(Role::AttnV), xb_.data(), pool_);

    apply_rope(q_.data(), n_heads, hd, pos, cfg_.rope_theta);
    apply_rope(k_.data(), n_kv,    hd, pos, cfg_.rope_theta);

    // store K/V for this position
    std::memcpy(kv_->k(layer, pos), k_.data(), kv_dim * sizeof(float));
    std::memcpy(kv_->v(layer, pos), v_.data(), kv_dim * sizeof(float));

    // per-head causal attention over [0, pos]
    for (int64_t h = 0; h < n_heads; ++h) {
        const float* qh = q_.data() + h * hd;
        const int64_t kvh = h / group;                 // GQA: shared kv head
        // scores
        for (int64_t t = 0; t <= pos; ++t) {
            const float* kt = kv_->k(layer, t) + kvh * hd;
            att_[t] = dot_f32(qh, kt, hd) * scale;
        }
        softmax(att_.data(), pos + 1);
        // weighted sum of V
        float* out = attn_out_.data() + h * hd;
        for (int64_t d = 0; d < hd; ++d) out[d] = 0.f;
        for (int64_t t = 0; t <= pos; ++t) {
            const float* vt = kv_->v(layer, t) + kvh * hd;
            axpy_f32(out, vt, att_[t], hd);
        }
    }

    // output projection + residual
    linear(proj_.data(), loader_->getWeight(Role::AttnOut), attn_out_.data(), pool_);
    vec_add_inplace(x_.data(), proj_.data(), dim);

    // --- feed-forward (SwiGLU) ---
    WeightRef fn = loader_->getWeight(Role::FfnNorm);
    rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(fn.data), dim, cfg_.rms_eps);

    linear(hb_.data(),  loader_->getWeight(Role::FfnGate), xb_.data(), pool_);
    linear(hb2_.data(), loader_->getWeight(Role::FfnUp),   xb_.data(), pool_);
    silu_inplace(hb_.data(), cfg_.ffn_dim);
    mul_f32(hb_.data(), hb_.data(), hb2_.data(), cfg_.ffn_dim);   // gate*up
    linear(proj_.data(), loader_->getWeight(Role::FfnDown), hb_.data(), pool_);
    vec_add_inplace(x_.data(), proj_.data(), dim);

    if (hidden_hook_) hidden_hook_((int)layer, x_.data(), dim);
}

const float* Transformer::forward(int64_t token, int64_t pos) {
    LLM_CHECK(pos < kv_->max_ctx(), "forward: position exceeds context window");

    // token -> embedding (streamed one row from disk, or resident if tied)
    loader_->embed_token(token, x_.data());

    if (profiling_) { timings_.assign(cfg_.n_layers, LayerTiming{}); }

    for (int64_t l = 0; l < cfg_.n_layers; ++l) {
        if (profiling_) {
            double t0 = now_sec();
            loader_->loadLayer((int)l);        // may block awaiting prefetch
            double t1 = now_sec();
            block(l, pos);
            double t2 = now_sec();
            loader_->unloadLayer();
            auto ls = loader_->layer_stat((int)l);
            LayerTiming& lt = timings_[l];
            lt.load_ms = (t1 - t0) * 1e3;
            lt.compute_ms = (t2 - t1) * 1e3;
            lt.io_ms = ls.io_us / 1e3;
            lt.dequant_ms = ls.dequant_us / 1e3;
            lt.rss_bytes = current_rss_bytes();
            if (lt.rss_bytes > peak_rss_) peak_rss_ = lt.rss_bytes;
        } else {
            loader_->loadLayer((int)l);        // stream/await this block's weights
            block(l, pos);
            loader_->unloadLayer();
        }
    }
    kv_->set_seq_len(pos + 1);

    // final norm + output projection to logits
    WeightRef on = loader_->output_norm_weight();
    rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(on.data), cfg_.dim, cfg_.rms_eps);
    linear(logits_.data(), loader_->output_weight(), xb_.data(), pool_);
    return logits_.data();
}

} // namespace llm
