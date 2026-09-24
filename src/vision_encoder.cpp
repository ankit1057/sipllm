// vision_encoder.cpp — ViT forward pass in pure C++17.
#include "llm/vision.h"
#include "llm/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace llm {

VisionEncoder::VisionEncoder(VisionConfig cfg)
    : cfg_(cfg), blocks_(cfg.num_layers) {
    int patch_dim = cfg_.in_channels * cfg_.patch_size * cfg_.patch_size;
    patch_embed_w_.resize(cfg_.hidden_dim * patch_dim, 0.0f);
    patch_embed_b_.resize(cfg_.hidden_dim, 0.0f);

    if (cfg_.use_cls_token) {
        cls_token_.resize(cfg_.hidden_dim, 0.0f);
    }
    pos_embed_.resize(cfg_.total_tokens() * cfg_.hidden_dim, 0.0f);

    ln_pre_w_.resize(cfg_.hidden_dim, 1.0f);
    ln_pre_b_.resize(cfg_.hidden_dim, 0.0f);
    ln_post_w_.resize(cfg_.hidden_dim, 1.0f);
    ln_post_b_.resize(cfg_.hidden_dim, 0.0f);

    for (auto& blk : blocks_) {
        blk.ln1_w.resize(cfg_.hidden_dim, 1.0f);
        blk.ln1_b.resize(cfg_.hidden_dim, 0.0f);
        blk.q_w.resize(cfg_.hidden_dim * cfg_.hidden_dim, 0.0f);
        blk.q_b.resize(cfg_.hidden_dim, 0.0f);
        blk.k_w.resize(cfg_.hidden_dim * cfg_.hidden_dim, 0.0f);
        blk.k_b.resize(cfg_.hidden_dim, 0.0f);
        blk.v_w.resize(cfg_.hidden_dim * cfg_.hidden_dim, 0.0f);
        blk.v_b.resize(cfg_.hidden_dim, 0.0f);
        blk.out_w.resize(cfg_.hidden_dim * cfg_.hidden_dim, 0.0f);
        blk.out_b.resize(cfg_.hidden_dim, 0.0f);
        blk.ln2_w.resize(cfg_.hidden_dim, 1.0f);
        blk.ln2_b.resize(cfg_.hidden_dim, 0.0f);
        blk.mlp_fc1_w.resize(cfg_.intermediate_dim * cfg_.hidden_dim, 0.0f);
        blk.mlp_fc1_b.resize(cfg_.intermediate_dim, 0.0f);
        blk.mlp_fc2_w.resize(cfg_.hidden_dim * cfg_.intermediate_dim, 0.0f);
        blk.mlp_fc2_b.resize(cfg_.hidden_dim, 0.0f);
    }
}

static bool copy_buf(std::vector<float>& dst, const float* src, size_t count) {
    if (dst.empty() || dst.size() != count) {
        dst.assign(src, src + count);
    } else {
        std::copy(src, src + count, dst.begin());
    }
    return true;
}

bool VisionEncoder::load_weight(const std::string& name, const float* data, size_t count) {
    if (name == "v.patch_embed.weight" || name == "v.patch_embed") {
        return copy_buf(patch_embed_w_, data, count);
    } else if (name == "v.patch_embed.bias") {
        return copy_buf(patch_embed_b_, data, count);
    } else if (name == "v.cls_token") {
        return copy_buf(cls_token_, data, count);
    } else if (name == "v.position_embed" || name == "v.pos_embed") {
        return copy_buf(pos_embed_, data, count);
    } else if (name == "v.ln_pre.weight") {
        return copy_buf(ln_pre_w_, data, count);
    } else if (name == "v.ln_pre.bias") {
        return copy_buf(ln_pre_b_, data, count);
    } else if (name == "v.ln_post.weight") {
        return copy_buf(ln_post_w_, data, count);
    } else if (name == "v.ln_post.bias") {
        return copy_buf(ln_post_b_, data, count);
    }

    // Format: v.blk.<idx>.<param>
    if (name.rfind("v.blk.", 0) == 0) {
        const char* p = name.c_str() + 6;
        char* endp = nullptr;
        long idx = strtol(p, &endp, 10);
        if (idx < 0 || idx >= (long)blocks_.size() || *endp != '.') {
            return false;
        }
        std::string sub = endp + 1;
        auto& blk = blocks_[idx];

        if (sub == "ln1.weight") return copy_buf(blk.ln1_w, data, count);
        if (sub == "ln1.bias")   return copy_buf(blk.ln1_b, data, count);
        if (sub == "attn_q.weight" || sub == "q_proj.weight") return copy_buf(blk.q_w, data, count);
        if (sub == "attn_q.bias" || sub == "q_proj.bias")     return copy_buf(blk.q_b, data, count);
        if (sub == "attn_k.weight" || sub == "k_proj.weight") return copy_buf(blk.k_w, data, count);
        if (sub == "attn_k.bias" || sub == "k_proj.bias")     return copy_buf(blk.k_b, data, count);
        if (sub == "attn_v.weight" || sub == "v_proj.weight") return copy_buf(blk.v_w, data, count);
        if (sub == "attn_v.bias" || sub == "v_proj.bias")     return copy_buf(blk.v_b, data, count);
        if (sub == "attn_out.weight" || sub == "out_proj.weight") return copy_buf(blk.out_w, data, count);
        if (sub == "attn_out.bias" || sub == "out_proj.bias")     return copy_buf(blk.out_b, data, count);
        if (sub == "ln2.weight") return copy_buf(blk.ln2_w, data, count);
        if (sub == "ln2.bias")   return copy_buf(blk.ln2_b, data, count);
        if (sub == "mlp_fc1.weight" || sub == "mlp.0.weight" || sub == "ffn_up.weight")
            return copy_buf(blk.mlp_fc1_w, data, count);
        if (sub == "mlp_fc1.bias" || sub == "mlp.0.bias" || sub == "ffn_up.bias")
            return copy_buf(blk.mlp_fc1_b, data, count);
        if (sub == "mlp_fc2.weight" || sub == "mlp.2.weight" || sub == "ffn_down.weight")
            return copy_buf(blk.mlp_fc2_w, data, count);
        if (sub == "mlp_fc2.bias" || sub == "mlp.2.bias" || sub == "ffn_down.bias")
            return copy_buf(blk.mlp_fc2_b, data, count);
    }

    return false;
}

void VisionEncoder::init_mock_weights() {
    auto fill_mat = [](std::vector<float>& w, float scale) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = std::sin(static_cast<float>(i + 1) * 0.137f) * scale;
        }
    };
    fill_mat(patch_embed_w_, 0.02f);
    fill_mat(pos_embed_, 0.01f);
    if (cfg_.use_cls_token) fill_mat(cls_token_, 0.01f);

    for (auto& blk : blocks_) {
        fill_mat(blk.q_w, 0.02f);
        fill_mat(blk.k_w, 0.02f);
        fill_mat(blk.v_w, 0.02f);
        fill_mat(blk.out_w, 0.02f);
        fill_mat(blk.mlp_fc1_w, 0.02f);
        fill_mat(blk.mlp_fc2_w, 0.02f);
    }
}

void VisionEncoder::patchify(const ImageTensor& img, float* out_patches) const {
    int P = cfg_.patch_size;
    int C = cfg_.in_channels;
    int W = img.width;
    int grid = cfg_.image_size / P;
    int patch_dim = C * P * P;

    for (int gy = 0; gy < grid; ++gy) {
        for (int gx = 0; gx < grid; ++gx) {
            int patch_idx = gy * grid + gx;
            float* p_out = out_patches + patch_idx * patch_dim;

            for (int c = 0; c < C; ++c) {
                for (int py = 0; py < P; ++py) {
                    for (int px = 0; px < P; ++px) {
                        int iy = gy * P + py;
                        int ix = gx * P + px;
                        int in_idx = (c * cfg_.image_size + iy) * W + ix;
                        *p_out++ = img.data[in_idx];
                    }
                }
            }
        }
    }
}

std::vector<float> VisionEncoder::forward(const ImageTensor& img, bool drop_cls, ThreadPool* pool) {
    if (!img.valid()) {
        throw std::invalid_argument("VisionEncoder: invalid ImageTensor input");
    }

    int N = cfg_.num_patches();
    int T = cfg_.total_tokens();
    int D = cfg_.hidden_dim;
    int patch_dim = cfg_.in_channels * cfg_.patch_size * cfg_.patch_size;

    // 1. Extract patches
    std::vector<float> patches(N * patch_dim);
    patchify(img, patches.data());

    // 2. Linear patch embedding: [N, D]
    std::vector<float> tokens(T * D, 0.0f);
    int token_offset = cfg_.use_cls_token ? 1 : 0;

    matmul_batch(tokens.data() + token_offset * D, patches.data(), patch_embed_w_.data(),
                 N, D, patch_dim, pool);

    // Add patch embedding bias
    for (int i = 0; i < N; ++i) {
        vec_add_inplace(tokens.data() + (i + token_offset) * D, patch_embed_b_.data(), D);
    }

    // Insert CLS token
    if (cfg_.use_cls_token) {
        std::copy(cls_token_.begin(), cls_token_.end(), tokens.begin());
    }

    // Add position embeddings
    vec_add_inplace(tokens.data(), pos_embed_.data(), T * D);

    // Pre-norm
    if (!ln_pre_w_.empty()) {
        for (int i = 0; i < T; ++i) {
            layernorm(tokens.data() + i * D, tokens.data() + i * D,
                      ln_pre_w_.data(), ln_pre_b_.data(), D, cfg_.eps);
        }
    }

    // Scratch buffers for Transformer blocks
    std::vector<float> norm1(T * D);
    std::vector<float> Q(T * D), K(T * D), V(T * D);
    std::vector<float> attn_scores(T * T);
    std::vector<float> attn_out(T * D);
    std::vector<float> proj_out(T * D);
    std::vector<float> norm2(T * D);
    std::vector<float> mlp1(T * cfg_.intermediate_dim);
    std::vector<float> mlp2(T * D);

    int heads = cfg_.num_heads;
    int head_dim = D / heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // 3. Encoder blocks
    for (size_t l = 0; l < blocks_.size(); ++l) {
        const auto& blk = blocks_[l];

        // --- LayerNorm 1 ---
        for (int i = 0; i < T; ++i) {
            layernorm(norm1.data() + i * D, tokens.data() + i * D,
                      blk.ln1_w.data(), blk.ln1_b.data(), D, cfg_.eps);
        }

        // --- Q, K, V projections ---
        matmul_batch(Q.data(), norm1.data(), blk.q_w.data(), T, D, D, pool);
        matmul_batch(K.data(), norm1.data(), blk.k_w.data(), T, D, D, pool);
        matmul_batch(V.data(), norm1.data(), blk.v_w.data(), T, D, D, pool);

        for (int i = 0; i < T; ++i) {
            vec_add_inplace(Q.data() + i * D, blk.q_b.data(), D);
            vec_add_inplace(K.data() + i * D, blk.k_b.data(), D);
            vec_add_inplace(V.data() + i * D, blk.v_b.data(), D);
        }

        // --- Multi-Head Self Attention ---
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);

        for (int h = 0; h < heads; ++h) {
            int h_off = h * head_dim;

            // QK^T / sqrt(head_dim)
            for (int i = 0; i < T; ++i) {
                const float* q_row = Q.data() + i * D + h_off;
                float* score_row = attn_scores.data() + i * T;

                for (int j = 0; j < T; ++j) {
                    const float* k_row = K.data() + j * D + h_off;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_row[d] * k_row[d];
                    }
                    score_row[j] = dot * scale;
                }
                softmax(score_row, T);

                // Weighted sum over V
                float* out_row = attn_out.data() + i * D + h_off;
                for (int j = 0; j < T; ++j) {
                    float s = score_row[j];
                    const float* v_row = V.data() + j * D + h_off;
                    for (int d = 0; d < head_dim; ++d) {
                        out_row[d] += s * v_row[d];
                    }
                }
            }
        }

        // Project out + residual
        matmul_batch(proj_out.data(), attn_out.data(), blk.out_w.data(), T, D, D, pool);
        for (int i = 0; i < T; ++i) {
            vec_add_inplace(proj_out.data() + i * D, blk.out_b.data(), D);
        }
        vec_add_inplace(tokens.data(), proj_out.data(), T * D);

        // --- LayerNorm 2 ---
        for (int i = 0; i < T; ++i) {
            layernorm(norm2.data() + i * D, tokens.data() + i * D,
                      blk.ln2_w.data(), blk.ln2_b.data(), D, cfg_.eps);
        }

        // --- MLP / FFN ---
        matmul_batch(mlp1.data(), norm2.data(), blk.mlp_fc1_w.data(),
                     T, cfg_.intermediate_dim, D, pool);
        for (int i = 0; i < T; ++i) {
            vec_add_inplace(mlp1.data() + i * cfg_.intermediate_dim, blk.mlp_fc1_b.data(), cfg_.intermediate_dim);
        }
        gelu_inplace(mlp1.data(), T * cfg_.intermediate_dim);

        matmul_batch(mlp2.data(), mlp1.data(), blk.mlp_fc2_w.data(),
                     T, D, cfg_.intermediate_dim, pool);
        for (int i = 0; i < T; ++i) {
            vec_add_inplace(mlp2.data() + i * D, blk.mlp_fc2_b.data(), D);
        }
        vec_add_inplace(tokens.data(), mlp2.data(), T * D);
    }

    // 4. Post-norm
    if (!ln_post_w_.empty()) {
        for (int i = 0; i < T; ++i) {
            layernorm(tokens.data() + i * D, tokens.data() + i * D,
                      ln_post_w_.data(), ln_post_b_.data(), D, cfg_.eps);
        }
    }

    // 5. If drop_cls, return only patch tokens [1..T-1]
    if (drop_cls && cfg_.use_cls_token) {
        std::vector<float> res(N * D);
        std::copy(tokens.begin() + D, tokens.end(), res.begin());
        return res;
    }

    return tokens;
}

} // namespace llm
