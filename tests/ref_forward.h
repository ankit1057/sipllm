// ref_forward.h — independent, all-weights-resident reference forward pass.
//
// Shared oracle for the end-to-end tests. It dequantizes every weight up front
// (so it works for fp32 .llmw and quantized GGUF alike) and runs the plain
// textbook math, giving a ground truth the streaming engine must match.
#pragma once

#include "llm/model.h"
#include "llm/ops.h"
#include "llm/quant.h"
#include "llm/simd.h"
#include "llm/transformer.h"
#include "llm/weight_source.h"

#include <cstring>
#include <vector>

namespace llmtest {

using namespace llm;

inline std::vector<float> get_dequant(WeightSource& s, const std::string& name) {
    const TensorInfo* ti = s.find(name);
    if (!ti) return {};
    std::vector<float> out(ti->numel());
    std::vector<uint8_t> raw(ti->nbytes);
    s.read_raw(*ti, raw.data());
    dequantize_row(ti->dtype, raw.data(), out.data(), ti->numel());
    return out;
}

// Full forward over `tokens`; returns logits at the final position.
inline std::vector<float> ref_forward(WeightSource& s, const ModelConfig& cfg,
                                      const std::vector<int64_t>& tokens) {
    const int64_t dim = cfg.dim, hd = cfg.head_dim, nh = cfg.n_heads,
                  nkv = cfg.n_kv_heads, kv_dim = cfg.kv_dim(), q_dim = cfg.q_dim(),
                  group = cfg.gqa_group(), ffn = cfg.ffn_dim;

    auto embd = get_dequant(s, names::token_embd);
    auto out_norm = get_dequant(s, names::output_norm);
    std::vector<float> out_w = cfg.tie_embeddings ? embd : get_dequant(s, names::output);

    std::vector<std::vector<float>> an(cfg.n_layers), wq(cfg.n_layers), wk(cfg.n_layers),
        wv(cfg.n_layers), wo(cfg.n_layers), fn(cfg.n_layers), wg(cfg.n_layers),
        wu(cfg.n_layers), wd(cfg.n_layers);
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
        an[l] = get_dequant(s, names::attn_norm(l)); wq[l] = get_dequant(s, names::attn_q(l));
        wk[l] = get_dequant(s, names::attn_k(l));    wv[l] = get_dequant(s, names::attn_v(l));
        wo[l] = get_dequant(s, names::attn_out(l));  fn[l] = get_dequant(s, names::ffn_norm(l));
        wg[l] = get_dequant(s, names::ffn_gate(l));  wu[l] = get_dequant(s, names::ffn_up(l));
        wd[l] = get_dequant(s, names::ffn_down(l));
    }

    const int64_t T = (int64_t)tokens.size();
    std::vector<float> kc((size_t)cfg.n_layers * T * kv_dim), vc((size_t)cfg.n_layers * T * kv_dim);
    auto kptr = [&](int64_t l, int64_t t) { return kc.data() + ((l * T + t) * kv_dim); };
    auto vptr = [&](int64_t l, int64_t t) { return vc.data() + ((l * T + t) * kv_dim); };

    std::vector<float> logits(cfg.vocab_size), x(dim), xb(dim), q(q_dim), k(kv_dim),
        v(kv_dim), att(T), ao(q_dim), proj(dim), hb(ffn), hb2(ffn);

    for (int64_t pos = 0; pos < T; ++pos) {
        std::memcpy(x.data(), embd.data() + tokens[pos] * dim, dim * sizeof(float));
        for (int64_t l = 0; l < cfg.n_layers; ++l) {
            rmsnorm(xb.data(), x.data(), an[l].data(), dim, cfg.rms_eps);
            matmul(q.data(), wq[l].data(), xb.data(), q_dim, dim);
            matmul(k.data(), wk[l].data(), xb.data(), kv_dim, dim);
            matmul(v.data(), wv[l].data(), xb.data(), kv_dim, dim);
            Transformer::apply_rope(q.data(), nh, hd, pos, cfg.rope_theta);
            Transformer::apply_rope(k.data(), nkv, hd, pos, cfg.rope_theta);
            std::memcpy(kptr(l, pos), k.data(), kv_dim * sizeof(float));
            std::memcpy(vptr(l, pos), v.data(), kv_dim * sizeof(float));
            float scale = 1.0f / std::sqrt((float)hd);
            for (int64_t h = 0; h < nh; ++h) {
                const float* qh = q.data() + h * hd;
                int64_t kvh = h / group;
                for (int64_t t = 0; t <= pos; ++t)
                    att[t] = dot_f32(qh, kptr(l, t) + kvh * hd, hd) * scale;
                softmax(att.data(), pos + 1);
                float* o = ao.data() + h * hd;
                for (int64_t d = 0; d < hd; ++d) o[d] = 0.f;
                for (int64_t t = 0; t <= pos; ++t)
                    axpy_f32(o, vptr(l, t) + kvh * hd, att[t], hd);
            }
            matmul(proj.data(), wo[l].data(), ao.data(), dim, q_dim);
            vec_add_inplace(x.data(), proj.data(), dim);
            rmsnorm(xb.data(), x.data(), fn[l].data(), dim, cfg.rms_eps);
            matmul(hb.data(), wg[l].data(), xb.data(), ffn, dim);
            matmul(hb2.data(), wu[l].data(), xb.data(), ffn, dim);
            silu_inplace(hb.data(), ffn);
            mul_f32(hb.data(), hb.data(), hb2.data(), ffn);
            matmul(proj.data(), wd[l].data(), hb.data(), dim, ffn);
            vec_add_inplace(x.data(), proj.data(), dim);
        }
    }
    rmsnorm(xb.data(), x.data(), out_norm.data(), dim, cfg.rms_eps);
    matmul(logits.data(), out_w.data(), xb.data(), cfg.vocab_size, dim);
    return logits;
}

} // namespace llmtest
