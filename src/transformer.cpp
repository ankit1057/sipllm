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
    fused_.assign(std::max(cfg_.q_dim() + 2 * cfg_.kv_dim(), 2 * cfg_.ffn_dim), 0.f);
    router_.assign(cfg_.block_spec.n_experts > 0 ? cfg_.block_spec.n_experts : 0, 0.f);
    moe_.assign(cfg_.dim, 0.f);
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

// Single data-driven transformer block pipeline (Issue #44).
// Conditionally applies all per-architecture features based on cfg_.block_spec.
void Transformer::block(int64_t layer, int64_t pos, int64_t bs, float* x_in) {
    if (bs == 0) return;
    if (x_in == nullptr) { x_in = x_.data(); bs = 1; }

    const int64_t dim = cfg_.dim;
    const int64_t hd = cfg_.head_dim;
    const int64_t n_heads = cfg_.n_heads;
    const int64_t n_kv = cfg_.n_kv_heads;
    const int64_t kv_dim = cfg_.kv_dim();
    const int64_t q_dim = cfg_.q_dim();
    const int64_t group = cfg_.gqa_group();
    const BlockSpec& b = cfg_.block_spec;

    // Ensure buffers are large enough for bs
    if (xb_.size() < (size_t)bs * dim) {
        xb_.resize(bs * dim);
        q_.resize(bs * q_dim);
        k_.resize(bs * kv_dim);
        v_.resize(bs * kv_dim);
        attn_out_.resize(bs * q_dim);
        proj_.resize(bs * dim);
        hb_.resize(bs * cfg_.ffn_dim);
        hb2_.resize(bs * cfg_.ffn_dim);
        fused_.resize(bs * std::max(q_dim + 2 * kv_dim, 2 * cfg_.ffn_dim));
        if (b.n_experts > 0) {
            router_.resize(bs * b.n_experts);
            moe_.resize(bs * dim);
        }
    }

    // --- 1. Pre-Attention Norm ---
    WeightRef an = loader_->getWeight(Role::AttnNorm);
    for (int64_t i = 0; i < bs; ++i) {
        float* xb_i = xb_.data() + i * dim;
        float* x_i = x_in + i * dim;
        switch (b.norm) {
            case NormKind::LayerNorm:
                layernorm(xb_i, x_i, static_cast<const float*>(an.data),
                          static_cast<const float*>(loader_->getWeight(Role::AttnNormBias).data), dim, cfg_.layernorm_eps);
                break;
            case NormKind::RMSNormGemma:
                rmsnorm_gemma(xb_i, x_i, static_cast<const float*>(an.data), dim, cfg_.rms_eps);
                break;
            case NormKind::RMSNorm:
                rmsnorm(xb_i, x_i, static_cast<const float*>(an.data), dim, cfg_.rms_eps);
                break;
        }
    }

    // --- 2. QKV Projection ---
    if (b.qkv_fused) {
        linear_batch(fused_.data(), loader_->getWeight(Role::AttnQKV), xb_.data(), bs, pool_);
        for (int64_t i = 0; i < bs; ++i) {
            float* fused_i = fused_.data() + i * (q_dim + 2 * kv_dim);
            add_bias(fused_i, loader_->getWeight(Role::AttnQKVBias), q_dim + 2 * kv_dim);
            std::memcpy(q_.data() + i * q_dim, fused_i, q_dim * sizeof(float));
            std::memcpy(k_.data() + i * kv_dim, fused_i + q_dim, kv_dim * sizeof(float));
            std::memcpy(v_.data() + i * kv_dim, fused_i + q_dim + kv_dim, kv_dim * sizeof(float));
        }
    } else {
        linear_batch(q_.data(), loader_->getWeight(Role::AttnQ), xb_.data(), bs, pool_);
        linear_batch(k_.data(), loader_->getWeight(Role::AttnK), xb_.data(), bs, pool_);
        linear_batch(v_.data(), loader_->getWeight(Role::AttnV), xb_.data(), bs, pool_);
        for (int64_t i = 0; i < bs; ++i) {
            add_bias(q_.data() + i * q_dim, loader_->getWeight(Role::AttnQBias), q_dim);
            add_bias(k_.data() + i * kv_dim, loader_->getWeight(Role::AttnKBias), kv_dim);
            add_bias(v_.data() + i * kv_dim, loader_->getWeight(Role::AttnVBias), kv_dim);
        }
    }

    // --- 3. Q/K Norm ---
    if (b.qk_norm) {
        WeightRef qn = loader_->getWeight(Role::AttnQNorm);
        WeightRef kn = loader_->getWeight(Role::AttnKNorm);
        for (int64_t i = 0; i < bs; ++i) {
            if (qn.valid())
                for (int64_t h = 0; h < n_heads; ++h) {
                    float* qh = q_.data() + i * q_dim + h * hd;
                    rmsnorm_gemma(qh, qh, static_cast<const float*>(qn.data), hd, cfg_.rms_eps);
                }
            if (kn.valid())
                for (int64_t h = 0; h < n_kv; ++h) {
                    float* kh = k_.data() + i * kv_dim + h * hd;
                    rmsnorm_gemma(kh, kh, static_cast<const float*>(kn.data), hd, cfg_.rms_eps);
                }
        }
    }

    // --- 4. RoPE ---
    if (b.rope != RopeKind::None) {
        float theta_base = cfg_.rope_theta;
        if (b.rope_dual_base && ((layer + 1) % cfg_.sliding_window_pattern) != 0)
            theta_base = cfg_.rope_theta_local;
            
        for (int64_t i = 0; i < bs; ++i) {
            float* qi = q_.data() + i * q_dim;
            float* ki = k_.data() + i * kv_dim;
            int64_t pos_i = pos + i;
            switch (b.rope) {
                case RopeKind::None: break;
                case RopeKind::Partial:
                    rope_rot(qi, n_heads, hd, b.rope_dim > 0 ? b.rope_dim : hd, pos_i, theta_base);
                    rope_rot(ki, n_kv,    hd, b.rope_dim > 0 ? b.rope_dim : hd, pos_i, theta_base);
                    break;
                case RopeKind::Llama3Scaled: {
                    RopeScaling rs;
                    rs.llama3 = true;
                    rs.factor = cfg_.rope_scale_factor;
                    rs.low_freq_factor = cfg_.rope_low_freq_factor;
                    rs.high_freq_factor = cfg_.rope_high_freq_factor;
                    rs.orig_ctx_len = (float)cfg_.rope_orig_ctx_len;
                    apply_rope(qi, n_heads, hd, pos_i, theta_base, rs);
                    apply_rope(ki, n_kv,    hd, pos_i, theta_base, rs);
                    break;
                }
                case RopeKind::Full:
                    apply_rope(qi, n_heads, hd, pos_i, theta_base);
                    apply_rope(ki, n_kv,    hd, pos_i, theta_base);
                    break;
            }
        }
    }

    // --- 5. Cache Write & 6. Attention Math ---
    const float scale = cfg_.query_pre_attn_scalar > 0.f ? (1.0f / std::sqrt(cfg_.query_pre_attn_scalar)) 
                                                         : (1.0f / std::sqrt((float)hd));
    std::vector<float> head_scratch;
    if (kv_->precision() == KVPrecision::Q8_0) head_scratch.resize(hd);

    for (int64_t i = 0; i < bs; ++i) {
        int64_t pos_i = pos + i;
        float* ki = k_.data() + i * kv_dim;
        float* vi = v_.data() + i * kv_dim;

        if (kv_->precision() == KVPrecision::Q8_0) {
            const int64_t q8_bytes = type_nbytes(DType::Q8_0, hd);
            const int64_t n_kv_heads = cfg_.n_kv_heads;
            for (int64_t h = 0; h < n_kv_heads; ++h) {
                uint8_t* k8 = (uint8_t*)kv_->k_ptr(layer, pos_i) + h * q8_bytes;
                uint8_t* v8 = (uint8_t*)kv_->v_ptr(layer, pos_i) + h * q8_bytes;
                quantize_q8_0(ki + h * hd, k8, hd);
                quantize_q8_0(vi + h * hd, v8, hd);
            }
        } else {
            std::memcpy(kv_->k_ptr(layer, pos_i), ki, kv_dim * sizeof(float));
            std::memcpy(kv_->v_ptr(layer, pos_i), vi, kv_dim * sizeof(float));
        }

        float* qi = q_.data() + i * q_dim;
        float* attn_out_i = attn_out_.data() + i * q_dim;
        
        const int64_t start_t = cfg_.sliding_window > 0 ? std::max<int64_t>(0, pos_i - cfg_.sliding_window + 1) : 0;
        const int64_t window_len = pos_i - start_t + 1;
        
        for (int64_t h = 0; h < n_heads; ++h) {
            const float* qh = qi + h * hd;
            const int64_t kvh = h / group;
            for (int64_t t = start_t; t <= pos_i; ++t) {
                if (kv_->precision() == KVPrecision::Q8_0) {
                    const uint8_t* k8 = (const uint8_t*)kv_->k_ptr(layer, t) + kvh * type_nbytes(DType::Q8_0, hd);
                    dequantize_row(DType::Q8_0, k8, head_scratch.data(), hd);
                    att_[t] = dot_f32(qh, head_scratch.data(), hd) * scale;
                } else {
                    att_[t] = dot_f32(qh, (const float*)kv_->k_ptr(layer, t) + kvh * hd, hd) * scale;
                }
            }
            if (b.attn_softcap) softcap_inplace(att_.data() + start_t, window_len, cfg_.attn_logit_softcap);
            softmax(att_.data() + start_t, window_len);
            float* out = attn_out_i + h * hd;
            for (int64_t d = 0; d < hd; ++d) out[d] = 0.f;
            for (int64_t t = start_t; t <= pos_i; ++t) {
                if (kv_->precision() == KVPrecision::Q8_0) {
                    const uint8_t* v8 = (const uint8_t*)kv_->v_ptr(layer, t) + kvh * type_nbytes(DType::Q8_0, hd);
                    dequantize_row(DType::Q8_0, v8, head_scratch.data(), hd);
                    axpy_f32(out, head_scratch.data(), att_[t], hd);
                } else {
                    axpy_f32(out, (const float*)kv_->v_ptr(layer, t) + kvh * hd, att_[t], hd);
                }
            }
        }
    }

    // --- 7. Attention Out Projection ---
    linear_batch(proj_.data(), loader_->getWeight(Role::AttnOut), attn_out_.data(), bs, pool_);
    for (int64_t i = 0; i < bs; ++i) {
        float* proj_i = proj_.data() + i * dim;
        float* x_i = x_in + i * dim;
        add_bias(proj_i, loader_->getWeight(Role::AttnOutBias), dim);
        if (b.post_attn_norm) {
            WeightRef apn = loader_->getWeight(Role::AttnPostNorm);
            if (apn.valid()) rmsnorm_gemma(proj_i, proj_i, static_cast<const float*>(apn.data), dim, cfg_.rms_eps);
        }
        vec_add_inplace(x_i, proj_i, dim);
    }

    // --- 8. FFN Pre-Norm ---
    if (!b.parallel_residual) {
        WeightRef fn = loader_->getWeight(Role::FfnNorm);
        for (int64_t i = 0; i < bs; ++i) {
            float* xb_i = xb_.data() + i * dim;
            float* x_i = x_in + i * dim;
            switch (b.norm) {
                case NormKind::LayerNorm:
                    layernorm(xb_i, x_i, static_cast<const float*>(fn.data),
                              static_cast<const float*>(loader_->getWeight(Role::FfnNormBias).data), dim, cfg_.layernorm_eps);
                    break;
                case NormKind::RMSNormGemma:
                    rmsnorm_gemma(xb_i, x_i, static_cast<const float*>(fn.data), dim, cfg_.rms_eps);
                    break;
                case NormKind::RMSNorm:
                    rmsnorm(xb_i, x_i, static_cast<const float*>(fn.data), dim, cfg_.rms_eps);
                    break;
            }
        }
    }

    // --- 9. FFN ---
    float* ffn_out = proj_.data();
    if (b.parallel_residual) ffn_out = moe_.data();

    if (b.moe) {
        const int64_t ff  = cfg_.ffn_dim;
        const int64_t ne  = b.n_experts;
        const int64_t k   = std::min<int64_t>(b.n_experts_used, ne);

        linear_batch(router_.data(), loader_->getWeight(Role::FfnGateInp), xb_.data(), bs, pool_);
        WeightRef ge = loader_->getWeight(Role::FfnGateExps);
        WeightRef ue = loader_->getWeight(Role::FfnUpExps);
        WeightRef de = loader_->getWeight(Role::FfnDownExps);
        const int64_t gate_rb = type_nbytes(ge.dtype, dim);
        const int64_t down_rb = type_nbytes(de.dtype, ff);
        const uint8_t* gbase = static_cast<const uint8_t*>(ge.data);
        const uint8_t* ubase = static_cast<const uint8_t*>(ue.data);
        const uint8_t* dbase = static_cast<const uint8_t*>(de.data);

        for (int64_t i = 0; i < bs; ++i) {
            float* router_i = router_.data() + i * ne;
            float* xb_i = xb_.data() + i * dim;
            float* moe_i = moe_.data() + i * dim;
            float* hb_i = hb_.data() + i * ff;
            float* hb2_i = hb2_.data() + i * ff;
            float* proj_i = proj_.data() + i * dim;
            
            softmax(router_i, ne);
            std::vector<int> ord(ne);
            for (int64_t e = 0; e < ne; ++e) ord[e] = (int)e;
            std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
                              [&](int lhs, int rhs) { return router_i[lhs] > router_i[rhs]; });
            float wsum = 0.f;
            for (int64_t j = 0; j < k; ++j) wsum += router_i[ord[j]];
            const float winv = wsum > 0.f ? 1.0f / wsum : 0.f;

            for (int64_t d = 0; d < dim; ++d) moe_i[d] = 0.f;
            for (int64_t j = 0; j < k; ++j) {
                const int64_t e = ord[j];
                const float w = router_i[e] * winv;
                WeightRef eg{gbase + e * ff * gate_rb, ge.dtype, ff, dim};
                WeightRef eu{ubase + e * ff * gate_rb, ue.dtype, ff, dim};
                WeightRef ed{dbase + e * dim * down_rb, de.dtype, dim, ff};
                // Can't batch easily across experts unless they match, so call linear
                linear(hb_i,  eg, xb_i, pool_);
                linear(hb2_i, eu, xb_i, pool_);
                silu_inplace(hb_i, ff);
                mul_f32(hb_i, hb_i, hb2_i, ff);
                linear(proj_i, ed, hb_i, pool_);
                axpy_f32(moe_i, proj_i, w, dim);
            }
        }
        if (ffn_out != moe_.data()) {
            std::memcpy(ffn_out, moe_.data(), bs * dim * sizeof(float));
        }
    } else {
        const int64_t ff = cfg_.ffn_dim;
        switch (b.ffn) {
            case FfnKind::GeluMLP:
                linear_batch(hb_.data(), loader_->getWeight(Role::FfnUp), xb_.data(), bs, pool_);
                for (int64_t i = 0; i < bs; ++i) {
                    float* hb_i = hb_.data() + i * ff;
                    add_bias(hb_i, loader_->getWeight(Role::FfnUpBias), ff);
                    gelu_inplace(hb_i, ff);
                }
                linear_batch(ffn_out, loader_->getWeight(Role::FfnDown), hb_.data(), bs, pool_);
                break;
            case FfnKind::GeGLU:
            case FfnKind::SwiGLU:
                if (b.ffn_fused_gate_up) {
                    linear_batch(fused_.data(), loader_->getWeight(Role::FfnUp), xb_.data(), bs, pool_);
                    for (int64_t i = 0; i < bs; ++i) {
                        float* gate = fused_.data() + i * 2 * ff;
                        float* up   = fused_.data() + i * 2 * ff + ff;
                        float* hb_i = hb_.data() + i * ff;
                        if (b.ffn == FfnKind::GeGLU) gelu_inplace(gate, ff);
                        else silu_inplace(gate, ff);
                        mul_f32(hb_i, gate, up, ff);
                    }
                } else {
                    linear_batch(hb_.data(),  loader_->getWeight(Role::FfnGate), xb_.data(), bs, pool_);
                    linear_batch(hb2_.data(), loader_->getWeight(Role::FfnUp),   xb_.data(), bs, pool_);
                    for (int64_t i = 0; i < bs; ++i) {
                        float* hb_i = hb_.data() + i * ff;
                        float* hb2_i = hb2_.data() + i * ff;
                        if (b.ffn == FfnKind::GeGLU) gelu_inplace(hb_i, ff);
                        else silu_inplace(hb_i, ff);
                        mul_f32(hb_i, hb_i, hb2_i, ff);
                    }
                }
                linear_batch(ffn_out, loader_->getWeight(Role::FfnDown), hb_.data(), bs, pool_);
                break;
        }
        if (b.proj_bias) {
            for (int64_t i = 0; i < bs; ++i) {
                add_bias(ffn_out + i * dim, loader_->getWeight(Role::FfnDownBias), dim);
            }
        }
    }

    // --- 10. Post-FFN Norm ---
    if (b.post_ffn_norm) {
        WeightRef fpn = loader_->getWeight(Role::FfnPostNorm);
        for (int64_t i = 0; i < bs; ++i) {
            float* ffn_out_i = ffn_out + i * dim;
            if (fpn.valid()) rmsnorm_gemma(ffn_out_i, ffn_out_i, static_cast<const float*>(fpn.data), dim, cfg_.rms_eps);
        }
    }

    // --- 11. Residual Add ---
    for (int64_t i = 0; i < bs; ++i) {
        float* x_i = x_in + i * dim;
        float* ffn_out_i = ffn_out + i * dim;
        vec_add_inplace(x_i, ffn_out_i, dim);
        if (hidden_hook_) hidden_hook_((int)layer, x_i, dim);
    }
}

const float* Transformer::forward(int64_t token, int64_t pos) {
    LLM_CHECK(pos < kv_->max_ctx(), "forward: position exceeds context window");

    // token -> embedding (streamed one row from disk, or resident if tied)
    loader_->embed_token(token, x_.data());
    // Gemma scales token embeddings by sqrt(dim); 1.0 elsewhere (no-op).
    if (cfg_.embedding_scale != 1.0f) scale_f32(x_.data(), cfg_.embedding_scale, cfg_.dim);
    // GPT-2 adds learned absolute position embeddings at the input.
    if (cfg_.learned_pos_emb) loader_->add_pos_embd(pos, x_.data());

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
    switch (cfg_.block_spec.norm) {
        case NormKind::LayerNorm:
            layernorm(xb_.data(), x_.data(), static_cast<const float*>(on.data),
                      static_cast<const float*>(loader_->output_norm_bias_weight().data),
                      cfg_.dim, cfg_.layernorm_eps);
            break;
        case NormKind::RMSNormGemma:
            rmsnorm_gemma(xb_.data(), x_.data(), static_cast<const float*>(on.data), cfg_.dim, cfg_.rms_eps);
            break;
        case NormKind::RMSNorm:
        default:
            rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(on.data), cfg_.dim, cfg_.rms_eps);
            break;
    }
    loader_->project_output(xb_.data(), logits_.data(), pool_);
    // Gemma 2 caps the final logits (no-op when the cap is 0).
    softcap_inplace(logits_.data(), cfg_.vocab_size, cfg_.final_logit_softcap);
    return logits_.data();
}

// Single-pass batched prefill (see transformer.h / RFC-007).
//
// `forward` streams the whole model once PER token, so a P-token prompt reads
// the entire model P times before decode starts. Here we stream each layer
// ONCE and push all P prompt positions through it while resident. The only
// per-position state carried between layers is the residual stream, so we keep
// P residual vectors (`resid_`) and shuttle each into/out of the existing
// single-position scratch `x_` around a normal `block()` call. Because every
// block() invocation is byte-for-byte the same computation as in `forward`
// (same kernels, same scratch, same reduction order), and positions are visited
// in ascending order so each position's K/V is committed before any later
// position attends to it, the resulting KV cache and final logits are IDENTICAL
// to running forward() token-by-token — only the redundant re-streaming is gone.
const float* Transformer::prefill(const int64_t* tokens, int64_t n, int64_t start_pos) {
    if (n <= 0) return logits_.data();
    LLM_CHECK(start_pos + n - 1 < kv_->max_ctx(),
              "prefill: position exceeds context window");

    const int64_t dim = cfg_.dim;

    // Materialize P residual streams: token embedding (+ Gemma scale, + GPT-2
    // learned position embeddings), exactly as forward() seeds x_.
    resid_.assign((size_t)n * dim, 0.f);
    for (int64_t i = 0; i < n; ++i) {
        float* xi = resid_.data() + (size_t)i * dim;
        loader_->embed_token(tokens[i], xi);
        if (cfg_.embedding_scale != 1.0f) scale_f32(xi, cfg_.embedding_scale, dim);
        if (cfg_.learned_pos_emb) loader_->add_pos_embd(start_pos + i, xi);
    }

    // Stream every layer once; run all positions through it (ascending => the
    // causal history each position needs is already in the KV cache).
    const int64_t C = 32; // batched GEMM size
    for (int64_t l = 0; l < cfg_.n_layers; ++l) {
        loader_->loadLayer((int)l);
        for (int64_t i = 0; i < n; i += C) {
            int64_t bs = std::min(C, n - i);
            float* xi = resid_.data() + (size_t)i * dim;
            block(l, start_pos + i, bs, xi);
        }
        loader_->unloadLayer();
    }
    kv_->set_seq_len(start_pos + n);

    // Only the LAST position's logits are needed to begin decode (intermediate
    // logits are discarded by the caller). Run the final norm + output
    // projection once, on the last position's residual.
    std::memcpy(x_.data(), resid_.data() + (size_t)(n - 1) * dim, dim * sizeof(float));
    WeightRef on = loader_->output_norm_weight();
    switch (cfg_.block_spec.norm) {
        case NormKind::LayerNorm:
            layernorm(xb_.data(), x_.data(), static_cast<const float*>(on.data),
                      static_cast<const float*>(loader_->output_norm_bias_weight().data),
                      dim, cfg_.layernorm_eps);
            break;
        case NormKind::RMSNormGemma:
            rmsnorm_gemma(xb_.data(), x_.data(), static_cast<const float*>(on.data), dim, cfg_.rms_eps);
            break;
        case NormKind::RMSNorm:
        default:
            rmsnorm(xb_.data(), x_.data(), static_cast<const float*>(on.data), dim, cfg_.rms_eps);
            break;
    }
    loader_->project_output(xb_.data(), logits_.data(), pool_);
    softcap_inplace(logits_.data(), cfg_.vocab_size, cfg_.final_logit_softcap);
    return logits_.data();
}

} // namespace llm
