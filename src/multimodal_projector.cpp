// multimodal_projector.cpp — 2-layer GELU MLP Projector for vision-to-text token alignment.
#include "llm/vision.h"
#include "llm/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace llm {

MultimodalProjector::MultimodalProjector(int vision_dim, int text_dim)
    : vision_dim_(vision_dim), text_dim_(text_dim) {
    fc1_w_.resize(text_dim_ * vision_dim_, 0.0f);
    fc1_b_.resize(text_dim_, 0.0f);
    fc2_w_.resize(text_dim_ * text_dim_, 0.0f);
    fc2_b_.resize(text_dim_, 0.0f);
}

static bool copy_buf(std::vector<float>& dst, const float* src, size_t count) {
    if (dst.empty() || dst.size() != count) {
        dst.assign(src, src + count);
    } else {
        std::copy(src, src + count, dst.begin());
    }
    return true;
}

bool MultimodalProjector::load_weight(const std::string& name, const float* data, size_t count) {
    if (name == "mm.0.weight" || name == "model.mm_projector.0.weight") {
        return copy_buf(fc1_w_, data, count);
    } else if (name == "mm.0.bias" || name == "model.mm_projector.0.bias") {
        return copy_buf(fc1_b_, data, count);
    } else if (name == "mm.2.weight" || name == "model.mm_projector.2.weight") {
        return copy_buf(fc2_w_, data, count);
    } else if (name == "mm.2.bias" || name == "model.mm_projector.2.bias") {
        return copy_buf(fc2_b_, data, count);
    }
    return false;
}

void MultimodalProjector::init_mock_weights() {
    auto fill_mat = [](std::vector<float>& w, float scale) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = std::sin(static_cast<float>(i + 1) * 0.093f) * scale;
        }
    };
    fill_mat(fc1_w_, 0.02f);
    fill_mat(fc2_w_, 0.02f);
    std::fill(fc1_b_.begin(), fc1_b_.end(), 0.01f);
    std::fill(fc2_b_.begin(), fc2_b_.end(), 0.01f);
}

std::vector<float> MultimodalProjector::forward(const float* vision_feats, int num_tokens, ThreadPool* pool) {
    if (!vision_feats || num_tokens <= 0) {
        throw std::invalid_argument("MultimodalProjector: invalid input features");
    }

    std::vector<float> h1(num_tokens * text_dim_);
    std::vector<float> h2(num_tokens * text_dim_);

    // Layer 1: [num_tokens, text_dim] = [num_tokens, vision_dim] @ fc1_w^T + fc1_b
    matmul_batch(h1.data(), vision_feats, fc1_w_.data(),
                 num_tokens, text_dim_, vision_dim_, pool);

    for (int i = 0; i < num_tokens; ++i) {
        vec_add_inplace(h1.data() + i * text_dim_, fc1_b_.data(), text_dim_);
    }

    // GELU activation
    gelu_inplace(h1.data(), num_tokens * text_dim_);

    // Layer 2: [num_tokens, text_dim] = [num_tokens, text_dim] @ fc2_w^T + fc2_b
    matmul_batch(h2.data(), h1.data(), fc2_w_.data(),
                 num_tokens, text_dim_, text_dim_, pool);

    for (int i = 0; i < num_tokens; ++i) {
        vec_add_inplace(h2.data() + i * text_dim_, fc2_b_.data(), text_dim_);
    }

    return h2;
}

} // namespace llm
