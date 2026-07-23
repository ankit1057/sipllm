// transformer.cpp — decoder forward pass (see transformer.h).
#include "llm/transformer.h"
#include "llm/linear.h"
#include "llm/ops.h"
#include "llm/simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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
    // Phi-3 fused projections need q_dim+2*kv_dim (qkv) or 2*ffn_dim (gate+up).
    fused_.assign(std::max(cfg_.q_dim() + 2 * cfg_.kv_dim(), 2 * cfg_.ffn_dim), 0.f);
    logits_.assign(cfg_.vocab_size, 0.f);
}

// llama3 RoPE scaling: stretch a base angular frequency per its wavelength so
// the trained short-context RoPE generalizes to the long context. Matches HF
// transformers `_compute_llama3_parameters` / llama.cpp: high-frequency
// components pass through, low-frequency ones are divided by `factor`, and the
// band between is a smooth interpolation of the two.
static inline float llama3_scale_freq(float freq, const Transformer::RopeScaling& rs) {
    constexpr float kPi = 3.14159265358979323846f;
    const float wavelen = 2.0f * kPi / freq;
    const float low_wl  = rs.orig_ctx_len / rs.low_freq_factor;   // long wavelength bound
    const float high_wl = rs.orig_ctx_len / rs.high_freq_factor;  // short wavelength bound
    if (wavelen < high_wl) return freq;                           // high freq: unchanged
    if (wavelen > low_wl)  return freq / rs.factor;               // low freq: /factor
    const float smooth = (rs.orig_ctx_len / wavelen - rs.low_freq_factor) /
                         (rs.high_freq_factor - rs.low_freq_factor);
    return (1.0f - smooth) * (freq / rs.factor) + smooth * freq;  // medium: interpolate
}

// Rotary embedding on adjacent element pairs (ggml "rope_norm"). GGUF Llama
// weights are permuted at conversion time so that this adjacent-pair rotation
// reproduces HF's rotate_half — so we apply exactly this to both q and k. This
// is the pristine plain-RoPE hot loop (pre-#9); the scaled variant is separate
// so the common path carries no per-element scaling branch.
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

// RoPE with optional llama3 frequency scaling. When scaling is off this defers
// to the plain loop above (identical codegen on the common path); only Llama-3.x
// models with rope.scaling.type=llama3 take the per-wavelength stretch branch.
void Transformer::apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                             int64_t pos, float theta_base, const RopeScaling& rs) {
    if (!rs.llama3) { apply_rope(vec, n_heads, head_dim, pos, theta_base); return; }
    const int64_t half = head_dim / 2;
    for (int64_t h = 0; h < n_heads; ++h) {
        float* p = vec + h * head_dim;
        for (int64_t i = 0; i < half; ++i) {
            float freq = std::pow(theta_base, -2.0f * (float)i / (float)head_dim);
            freq = llama3_scale_freq(freq, rs);
            float angle = (float)pos * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = p[2 * i], x1 = p[2 * i + 1];
            p[2 * i]     = x0 * c - x1 * s;
            p[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

// Add an optional bias vector to a projection output, in place. A no-op when
// the weight is absent (invalid ref) — the common case for bias-free archs.
static inline void add_bias(float* y, const WeightRef& b, int64_t n) {
    if (!b.valid()) return;
    const float* bv = static_cast<const float*>(b.data);
    for (int64_t i = 0; i < n; ++i) y[i] += bv[i];
}

// Dispatch one block to the per-architecture implementation. Every arch except
// the Llama reference is added as a new case here (issues #10-#16); Unknown
// architectures fall through to the Llama path unchanged.
void Transformer::block(int64_t layer, int64_t pos) {
    switch (cfg_.arch_kind) {
        case Arch::Llama:
        case Arch::Mistral:   // RMSNorm + RoPE + GQA + SwiGLU, same as Llama
        case Arch::Qwen2:     // Llama block + optional q/k/v biases (applied below)
            block_llama(layer, pos);
            return;
        case Arch::Gemma2:
        case Arch::Gemma3:    // Gemma2 shape + QK-norm + per-layer RoPE (handled in-block)
            block_gemma2(layer, pos);
            return;
        case Arch::Phi3:
            block_phi3(layer, pos);
            return;
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

    // Optional attention biases (Qwen2). No-op when the tensors are absent
    // (Llama / Mistral), so this path stays byte-identical for them.
    add_bias(q_.data(), loader_->getWeight(Role::AttnQBias), cfg_.q_dim());
    add_bias(k_.data(), loader_->getWeight(Role::AttnKBias), kv_dim);
    add_bias(v_.data(), loader_->getWeight(Role::AttnVBias), kv_dim);

    RopeScaling rs;
    if (cfg_.use_llama3_rope()) {
        rs.llama3 = true;
        rs.factor = cfg_.rope_scale_factor;
        rs.low_freq_factor = cfg_.rope_low_freq_factor;
        rs.high_freq_factor = cfg_.rope_high_freq_factor;
        rs.orig_ctx_len = (float)cfg_.rope_orig_ctx_len;
    }
    apply_rope(q_.data(), n_heads, hd, pos, cfg_.rope_theta, rs);
    apply_rope(k_.data(), n_kv,    hd, pos, cfg_.rope_theta, rs);

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

// Gemma 2 block: (1+w) RMSNorm, pre + post norms around each sublayer, GeGLU
// FFN, optional attention-logit soft-capping, and a head_dim that is
// independent of the model dim. No q/k/v biases. See ops.h for the primitives.
void Transformer::block_gemma2(int64_t layer, int64_t pos) {
    const int64_t dim = cfg_.dim;
    const int64_t hd = cfg_.head_dim;
    const int64_t n_heads = cfg_.n_heads;
    const int64_t n_kv = cfg_.n_kv_heads;
    const int64_t kv_dim = cfg_.kv_dim();
    const int64_t group = cfg_.gqa_group();
    // Attention scale: 1/sqrt(query_pre_attn_scalar) if given, else 1/sqrt(head_dim).
    const float qs = cfg_.query_pre_attn_scalar > 0.f ? cfg_.query_pre_attn_scalar : (float)hd;
    const float scale = 1.0f / std::sqrt(qs);
    const float eps = cfg_.rms_eps;

    // --- attention (pre-norm) ---
    WeightRef an = loader_->getWeight(Role::AttnNorm);
    rmsnorm_gemma(xb_.data(), x_.data(), static_cast<const float*>(an.data), dim, eps);

    linear(q_.data(), loader_->getWeight(Role::AttnQ), xb_.data(), pool_);
    linear(k_.data(), loader_->getWeight(Role::AttnK), xb_.data(), pool_);
    linear(v_.data(), loader_->getWeight(Role::AttnV), xb_.data(), pool_);

    // Gemma 3 QK-norm: (1+w) RMSNorm per head over head_dim, applied to q/k
    // before RoPE. Absent in Gemma 2 (invalid refs) => skipped.
    WeightRef qn = loader_->getWeight(Role::AttnQNorm);
    WeightRef kn = loader_->getWeight(Role::AttnKNorm);
    if (qn.valid())
        for (int64_t h = 0; h < n_heads; ++h)
            rmsnorm_gemma(q_.data() + h * hd, q_.data() + h * hd,
                          static_cast<const float*>(qn.data), hd, eps);
    if (kn.valid())
        for (int64_t h = 0; h < n_kv; ++h)
            rmsnorm_gemma(k_.data() + h * hd, k_.data() + h * hd,
                          static_cast<const float*>(kn.data), hd, eps);

    // Gemma 3 uses a smaller RoPE base on sliding-window (local) layers; the
    // pattern marks every Nth layer global. 0 pattern / 0 local base => the
    // single rope_theta (Gemma 2 and everything else).
    float theta = cfg_.rope_theta;
    if (cfg_.rope_theta_local > 0.f && cfg_.sliding_window_pattern > 0 &&
        ((layer + 1) % cfg_.sliding_window_pattern) != 0)
        theta = cfg_.rope_theta_local;   // local (sliding-window) layer
    apply_rope(q_.data(), n_heads, hd, pos, theta);
    apply_rope(k_.data(), n_kv,    hd, pos, theta);

    std::memcpy(kv_->k(layer, pos), k_.data(), kv_dim * sizeof(float));
    std::memcpy(kv_->v(layer, pos), v_.data(), kv_dim * sizeof(float));

    for (int64_t h = 0; h < n_heads; ++h) {
        const float* qh = q_.data() + h * hd;
        const int64_t kvh = h / group;
        for (int64_t t = 0; t <= pos; ++t) {
            const float* kt = kv_->k(layer, t) + kvh * hd;
            att_[t] = dot_f32(qh, kt, hd) * scale;
        }
        // Gemma 2 caps attention logits before the softmax.
        softcap_inplace(att_.data(), pos + 1, cfg_.attn_logit_softcap);
        softmax(att_.data(), pos + 1);
        float* out = attn_out_.data() + h * hd;
        for (int64_t d = 0; d < hd; ++d) out[d] = 0.f;
        for (int64_t t = 0; t <= pos; ++t) {
            const float* vt = kv_->v(layer, t) + kvh * hd;
            axpy_f32(out, vt, att_[t], hd);
        }
    }

    // output projection -> post-attention norm -> residual
    linear(proj_.data(), loader_->getWeight(Role::AttnOut), attn_out_.data(), pool_);
    WeightRef apn = loader_->getWeight(Role::AttnPostNorm);
    if (apn.valid())
        rmsnorm_gemma(proj_.data(), proj_.data(), static_cast<const float*>(apn.data), dim, eps);
    vec_add_inplace(x_.data(), proj_.data(), dim);

    // --- feed-forward (GeGLU, pre-norm) ---
    WeightRef fn = loader_->getWeight(Role::FfnNorm);
    rmsnorm_gemma(xb_.data(), x_.data(), static_cast<const float*>(fn.data), dim, eps);

    linear(hb_.data(),  loader_->getWeight(Role::FfnGate), xb_.data(), pool_);
    linear(hb2_.data(), loader_->getWeight(Role::FfnUp),   xb_.data(), pool_);
    gelu_inplace(hb_.data(), cfg_.ffn_dim);                       // GELU, not SiLU
    mul_f32(hb_.data(), hb_.data(), hb2_.data(), cfg_.ffn_dim);   // gate*up
    linear(proj_.data(), loader_->getWeight(Role::FfnDown), hb_.data(), pool_);

    // post-FFN norm -> residual
    WeightRef fpn = loader_->getWeight(Role::FfnPostNorm);
    if (fpn.valid())
        rmsnorm_gemma(proj_.data(), proj_.data(), static_cast<const float*>(fpn.data), dim, eps);
    vec_add_inplace(x_.data(), proj_.data(), dim);

    if (hidden_hook_) hidden_hook_((int)layer, x_.data(), dim);
}

// Partial-rotary RoPE: rotate only the first `rot_dim` dims of each head (the
// remaining head_dim - rot_dim pass through). rot_dim == head_dim is full RoPE.
static void rope_rot(float* vec, int64_t n_heads, int64_t head_dim,
                     int64_t rot_dim, int64_t pos, float theta) {
    const int64_t half = rot_dim / 2;
    for (int64_t h = 0; h < n_heads; ++h) {
        float* p = vec + h * head_dim;
        for (int64_t i = 0; i < half; ++i) {
            float freq = std::pow(theta, -2.0f * (float)i / (float)rot_dim);
            float angle = (float)pos * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = p[2 * i], x1 = p[2 * i + 1];
            p[2 * i]     = x0 * c - x1 * s;
            p[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

// Phi-3 block: RMSNorm + SwiGLU like Llama, but q/k/v come from one fused
// attn_qkv projection, gate+up come from one fused ffn_up ([gate;up], 2*ffn
// rows), and RoPE is partial (rope_dim rotary dims per head).
void Transformer::block_phi3(int64_t layer, int64_t pos) {
    const int64_t dim = cfg_.dim;
    const int64_t hd = cfg_.head_dim;
    const int64_t n_heads = cfg_.n_heads;
    const int64_t n_kv = cfg_.n_kv_heads;
    const int64_t kv_dim = cfg_.kv_dim();
    const int64_t q_dim = cfg_.q_dim();
    const int64_t group = cfg_.gqa_group();
    const int64_t ff = cfg_.ffn_dim;
    const int64_t rot = cfg_.rope_dim > 0 ? cfg_.rope_dim : hd;
    const float scale = 1.0f / std::sqrt((float)hd);

    // --- attention: one fused QKV projection, then split ---
    WeightRef an = loader_->getWeight(Role::AttnNorm);
    rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(an.data), dim, cfg_.rms_eps);
    linear(fused_.data(), loader_->getWeight(Role::AttnQKV), xb_.data(), pool_);
    std::memcpy(q_.data(), fused_.data(),                 q_dim * sizeof(float));
    std::memcpy(k_.data(), fused_.data() + q_dim,         kv_dim * sizeof(float));
    std::memcpy(v_.data(), fused_.data() + q_dim + kv_dim, kv_dim * sizeof(float));

    rope_rot(q_.data(), n_heads, hd, rot, pos, cfg_.rope_theta);
    rope_rot(k_.data(), n_kv,    hd, rot, pos, cfg_.rope_theta);

    std::memcpy(kv_->k(layer, pos), k_.data(), kv_dim * sizeof(float));
    std::memcpy(kv_->v(layer, pos), v_.data(), kv_dim * sizeof(float));

    for (int64_t h = 0; h < n_heads; ++h) {
        const float* qh = q_.data() + h * hd;
        const int64_t kvh = h / group;
        for (int64_t t = 0; t <= pos; ++t)
            att_[t] = dot_f32(qh, kv_->k(layer, t) + kvh * hd, hd) * scale;
        softmax(att_.data(), pos + 1);
        float* out = attn_out_.data() + h * hd;
        for (int64_t d = 0; d < hd; ++d) out[d] = 0.f;
        for (int64_t t = 0; t <= pos; ++t)
            axpy_f32(out, kv_->v(layer, t) + kvh * hd, att_[t], hd);
    }
    linear(proj_.data(), loader_->getWeight(Role::AttnOut), attn_out_.data(), pool_);
    vec_add_inplace(x_.data(), proj_.data(), dim);

    // --- feed-forward: one fused gate+up projection ([gate;up]) ---
    WeightRef fn = loader_->getWeight(Role::FfnNorm);
    rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(fn.data), dim, cfg_.rms_eps);
    linear(fused_.data(), loader_->getWeight(Role::FfnUp), xb_.data(), pool_);   // [gate; up]
    float* gate = fused_.data();
    float* up   = fused_.data() + ff;
    silu_inplace(gate, ff);
    mul_f32(hb_.data(), gate, up, ff);                       // silu(gate) * up
    linear(proj_.data(), loader_->getWeight(Role::FfnDown), hb_.data(), pool_);
    vec_add_inplace(x_.data(), proj_.data(), dim);

    if (hidden_hook_) hidden_hook_((int)layer, x_.data(), dim);
}

const float* Transformer::forward(int64_t token, int64_t pos) {
    LLM_CHECK(pos < kv_->max_ctx(), "forward: position exceeds context window");

    // token -> embedding (streamed one row from disk, or resident if tied)
    loader_->embed_token(token, x_.data());
    // Gemma scales token embeddings by sqrt(dim); 1.0 elsewhere (no-op).
    if (cfg_.embedding_scale != 1.0f) scale_f32(x_.data(), cfg_.embedding_scale, cfg_.dim);

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
    if (cfg_.gemma_rmsnorm)
        rmsnorm_gemma(xb_.data(), x_.data(), static_cast<const float*>(on.data), cfg_.dim, cfg_.rms_eps);
    else
        rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(on.data), cfg_.dim, cfg_.rms_eps);
    linear(logits_.data(), loader_->output_weight(), xb_.data(), pool_);
    // Gemma 2 caps the final logits (no-op when the cap is 0).
    softcap_inplace(logits_.data(), cfg_.vocab_size, cfg_.final_logit_softcap);
    return logits_.data();
}

} // namespace llm
