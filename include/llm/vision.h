// vision.h — Vision Transformer (ViT) & Multimodal Projector for multimodal inference.
//
// Pure C++17 implementation of ViT patch extraction, multi-head self-attention,
// and 2-layer GELU MLP projector compatible with LLaVA/CLIP tensor conventions.
#pragma once

#include "llm/ops.h"
#include "llm/threadpool.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// Raw float32 RGB image tensor. Caller/plugin handles PNG/JPEG decoding and normalization.
struct ImageTensor {
    int width = 0;
    int height = 0;
    int channels = 3;
    // Planar CHW or interleaved HWC data (size = width * height * channels)
    std::vector<float> data;

    bool valid() const {
        return width > 0 && height > 0 && channels > 0 &&
               data.size() == static_cast<size_t>(width * height * channels);
    }
};

struct VisionConfig {
    int image_size = 336;        // Standard LLaVA/CLIP input resolution
    int patch_size = 14;         // 14x14 patches -> (336/14)^2 = 576 patches
    int in_channels = 3;         // RGB
    int hidden_dim = 1024;       // ViT-L: 1024, ViT-B: 768
    int num_layers = 24;         // ViT-L: 24, ViT-B: 12
    int num_heads = 16;          // 1024 / 16 = 64 head_dim
    int intermediate_dim = 4096; // FFN hidden dimension
    int text_hidden_dim = 4096;  // Text model embedding dimension
    float eps = 1e-5f;
    bool use_cls_token = true;

    int num_patches() const {
        int grid = image_size / patch_size;
        return grid * grid;
    }
    int total_tokens() const {
        return num_patches() + (use_cls_token ? 1 : 0);
    }
};

// ViT Transformer block weights (v.blk.N.*)
struct VisionBlockWeights {
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> q_w, q_b;
    std::vector<float> k_w, k_b;
    std::vector<float> v_w, v_b;
    std::vector<float> out_w, out_b;
    std::vector<float> ln2_w, ln2_b;
    std::vector<float> mlp_fc1_w, mlp_fc1_b;
    std::vector<float> mlp_fc2_w, mlp_fc2_b;
};

// Pure C++17 Vision Transformer Encoder
class VisionEncoder {
public:
    explicit VisionEncoder(VisionConfig cfg);

    const VisionConfig& config() const { return cfg_; }

    // Load tensor by LLaVA/CLIP name (e.g. "v.patch_embed.weight", "v.blk.0.attn_q.weight", etc.)
    bool load_weight(const std::string& name, const float* data, size_t count);

    // Initialize weights with deterministic test values (for unit testing and verification)
    void init_mock_weights();

    // Extract patches from image: out_patches has size num_patches * (in_channels * patch_size * patch_size)
    void patchify(const ImageTensor& img, float* out_patches) const;

    // Forward pass: encodes ImageTensor into patch embeddings.
    // If drop_cls is true (standard in LLaVA), removes CLS token and returns [num_patches, hidden_dim].
    std::vector<float> forward(const ImageTensor& img, bool drop_cls = true, ThreadPool* pool = nullptr);

private:
    VisionConfig cfg_;

    std::vector<float> patch_embed_w_; // [hidden_dim, in_channels * patch_size * patch_size]
    std::vector<float> patch_embed_b_; // [hidden_dim]
    std::vector<float> cls_token_;     // [hidden_dim]
    std::vector<float> pos_embed_;     // [total_tokens, hidden_dim]
    std::vector<float> ln_pre_w_, ln_pre_b_;
    std::vector<float> ln_post_w_, ln_post_b_;

    std::vector<VisionBlockWeights> blocks_;
};

// Multimodal Projector (2-layer GELU MLP: mm.0.weight, mm.0.bias, mm.2.weight, mm.2.bias)
class MultimodalProjector {
public:
    MultimodalProjector(int vision_dim, int text_dim);

    int vision_dim() const { return vision_dim_; }
    int text_dim() const { return text_dim_; }

    // Load tensor by projector name: "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"
    bool load_weight(const std::string& name, const float* data, size_t count);

    // Initialize weights with deterministic test values
    void init_mock_weights();

    // Forward pass: projects [num_tokens, vision_dim] -> [num_tokens, text_dim]
    std::vector<float> forward(const float* vision_feats, int num_tokens, ThreadPool* pool = nullptr);

private:
    int vision_dim_;
    int text_dim_;
    std::vector<float> fc1_w_; // [text_dim, vision_dim]
    std::vector<float> fc1_b_; // [text_dim]
    std::vector<float> fc2_w_; // [text_dim, text_dim]
    std::vector<float> fc2_b_; // [text_dim]
};

} // namespace llm
