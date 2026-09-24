#include "llm/vision.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace llm;

static void test_image_tensor_and_patchify() {
    VisionConfig cfg;
    cfg.image_size = 28;
    cfg.patch_size = 14;
    cfg.in_channels = 3;
    cfg.hidden_dim = 64;
    cfg.num_layers = 1;
    cfg.num_heads = 4;
    cfg.intermediate_dim = 128;
    cfg.text_hidden_dim = 64;

    assert(cfg.num_patches() == 4); // (28/14)^2 = 4

    ImageTensor img;
    img.width = 28;
    img.height = 28;
    img.channels = 3;
    img.data.resize(28 * 28 * 3, 0.5f);
    assert(img.valid());

    VisionEncoder enc(cfg);
    int patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    std::vector<float> patches(cfg.num_patches() * patch_dim);
    enc.patchify(img, patches.data());

    for (float v : patches) {
        assert(std::fabs(v - 0.5f) < 1e-6f);
    }
    printf("  PASS  test_image_tensor_and_patchify\n");
}

static void test_vision_encoder_forward() {
    VisionConfig cfg;
    cfg.image_size = 28;
    cfg.patch_size = 14;
    cfg.in_channels = 3;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.intermediate_dim = 128;
    cfg.text_hidden_dim = 128;
    cfg.use_cls_token = true;

    VisionEncoder enc(cfg);
    enc.init_mock_weights();

    ImageTensor img;
    img.width = 28;
    img.height = 28;
    img.channels = 3;
    img.data.resize(28 * 28 * 3);
    for (size_t i = 0; i < img.data.size(); ++i) {
        img.data[i] = std::sin(static_cast<float>(i) * 0.05f);
    }

    // Forward with drop_cls = true
    auto out_dropped = enc.forward(img, true);
    assert(out_dropped.size() == static_cast<size_t>(cfg.num_patches() * cfg.hidden_dim));

    // Forward with drop_cls = false
    auto out_full = enc.forward(img, false);
    assert(out_full.size() == static_cast<size_t>(cfg.total_tokens() * cfg.hidden_dim));

    // Verify non-trivial finite output values
    for (float v : out_dropped) {
        assert(std::isfinite(v));
    }

    printf("  PASS  test_vision_encoder_forward\n");
}

static void test_multimodal_projector() {
    int vision_dim = 64;
    int text_dim = 128;
    int num_tokens = 4;

    MultimodalProjector proj(vision_dim, text_dim);
    proj.init_mock_weights();

    std::vector<float> vision_feats(num_tokens * vision_dim, 0.25f);
    auto text_embeddings = proj.forward(vision_feats.data(), num_tokens);

    assert(text_embeddings.size() == static_cast<size_t>(num_tokens * text_dim));
    for (float v : text_embeddings) {
        assert(std::isfinite(v));
    }

    printf("  PASS  test_multimodal_projector\n");
}

static void test_end_to_end_multimodal_pipeline() {
    VisionConfig cfg;
    cfg.image_size = 28;
    cfg.patch_size = 14;
    cfg.in_channels = 3;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.intermediate_dim = 128;
    cfg.text_hidden_dim = 256; // e.g. target LLM hidden dim
    cfg.use_cls_token = true;

    VisionEncoder enc(cfg);
    enc.init_mock_weights();

    MultimodalProjector proj(cfg.hidden_dim, cfg.text_hidden_dim);
    proj.init_mock_weights();

    ImageTensor img;
    img.width = 28;
    img.height = 28;
    img.channels = 3;
    img.data.resize(28 * 28 * 3, 0.7f);

    // 1. Vision encode
    auto patch_tokens = enc.forward(img, true); // 4 patches x 64 dims
    assert(patch_tokens.size() == 4 * 64);

    // 2. Project to text token embedding space
    auto projected = proj.forward(patch_tokens.data(), 4);
    assert(projected.size() == 4 * 256);

    for (float v : projected) {
        assert(std::isfinite(v));
    }

    printf("  PASS  test_end_to_end_multimodal_pipeline\n");
}

int main() {
    printf("== test_vision ==\n");
    test_image_tensor_and_patchify();
    test_vision_encoder_forward();
    test_multimodal_projector();
    test_end_to_end_multimodal_pipeline();
    printf("All vision tests passed!\n");
    return 0;
}
