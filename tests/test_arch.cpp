// test_arch.cpp — multi-architecture dispatch (#7 and friends).
//
// These are *implementation* checks: architecture resolution, per-arch metadata
// discovery, and that each new arch's block behaves as designed on a toy model.
// Cross-engine numeric parity (golden vs llama.cpp) needs a real model download
// + a llama.cpp build and is not run here.
#include "llm/gguf.h"
#include "llm/gguf_writer.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

// Streaming forward over a token sequence; returns the final logits.
static std::vector<float> forward_gguf(const std::string& path,
                                       const std::vector<int64_t>& toks) {
    GgufFile g(path);
    ModelConfig cfg = ModelConfig::from_source(g);
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
    LayerLoader loader(&g, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(2);
    Transformer tf(&loader, &kv, &pool);
    const float* lg = nullptr;
    for (size_t i = 0; i < toks.size(); ++i) lg = tf.forward(toks[i], (int64_t)i);
    return std::vector<float>(lg, lg + cfg.vocab_size);
}

TEST(arch_name_resolution) {
    CHECK(arch_from_name("llama") == Arch::Llama);
    CHECK(arch_from_name("mistral") == Arch::Mistral);
    CHECK(arch_from_name("totally-made-up") == Arch::Unknown);
    CHECK(std::string(arch_name(Arch::Mistral)) == "mistral");
}

TEST(mistral_config_discovery) {
    // hparams namespaced under "mistral.*" must be discovered generically.
    ToyGgufConfig c; c.arch = "mistral";
    c.n_layers = 3; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = 13;
    std::string p = scratch("mistral_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.arch == "mistral" && cfg.arch_kind == Arch::Mistral);
    CHECK(cfg.n_layers == 3 && cfg.n_heads == 4 && cfg.n_kv_heads == 2);
    CHECK(cfg.dim == 32 && cfg.ffn_dim == 64 && cfg.vocab_size == 48);
}

TEST(mistral_matches_llama_block) {
    // Identical weights (same seed), one labeled llama and one mistral. Mistral
    // is RMSNorm+RoPE+GQA+SwiGLU like Llama and routes to the same block, so the
    // logits must be byte-identical.
    ToyGgufConfig cl; cl.arch = "llama";
    cl.n_layers = 3; cl.dim = 32; cl.n_heads = 4; cl.n_kv_heads = 2;
    cl.ffn_dim = 64; cl.vocab_size = 48; cl.seed = 13;
    ToyGgufConfig cm = cl; cm.arch = "mistral";
    std::string pl = scratch("as_llama.gguf"), pm = scratch("as_mistral.gguf");
    write_toy_gguf(pl, cl); write_toy_gguf(pm, cm);

    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(pl, toks), b = forward_gguf(pm, toks);
    CHECK(a.size() == b.size() && !a.empty());
    for (size_t i = 0; i < a.size(); ++i) APPROX(a[i], b[i], 0);
}

TEST(qwen2_config_discovery) {
    ToyGgufConfig c; c.arch = "qwen2"; c.attn_qkv_bias = true;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = 21;
    std::string p = scratch("qwen2_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.arch == "qwen2" && cfg.arch_kind == Arch::Qwen2);
    CHECK(g.find("blk.0.attn_q.bias") != nullptr);   // bias tensors present
}

TEST(qwen2_zero_bias_equals_llama) {
    // Qwen2 with zero q/k/v biases must reproduce the bias-free Llama forward
    // over identical weights — proves the bias plumbing is a clean no-op at 0.
    ToyGgufConfig cl; cl.arch = "llama";
    cl.n_layers = 3; cl.dim = 32; cl.n_heads = 4; cl.n_kv_heads = 2;
    cl.ffn_dim = 64; cl.vocab_size = 48; cl.seed = 21;
    ToyGgufConfig cq = cl; cq.arch = "qwen2"; cq.attn_qkv_bias = true; cq.zero_bias = true;
    std::string pl = scratch("q_llama.gguf"), pq = scratch("q_zero.gguf");
    write_toy_gguf(pl, cl); write_toy_gguf(pq, cq);
    std::vector<int64_t> toks = {3, 1, 4, 1, 5};
    auto a = forward_gguf(pl, toks), b = forward_gguf(pq, toks);
    for (size_t i = 0; i < a.size(); ++i) APPROX(a[i], b[i], 0);
}

TEST(qwen2_bias_changes_output) {
    // Same weights, nonzero biases: the qkv bias must actually move the logits.
    ToyGgufConfig cl; cl.arch = "llama";
    cl.n_layers = 3; cl.dim = 32; cl.n_heads = 4; cl.n_kv_heads = 2;
    cl.ffn_dim = 64; cl.vocab_size = 48; cl.seed = 21;
    ToyGgufConfig cq = cl; cq.arch = "qwen2"; cq.attn_qkv_bias = true; cq.zero_bias = false;
    std::string pl = scratch("q_llama2.gguf"), pq = scratch("q_rand.gguf");
    write_toy_gguf(pl, cl); write_toy_gguf(pq, cq);
    std::vector<int64_t> toks = {3, 1, 4, 1, 5};
    auto a = forward_gguf(pl, toks), b = forward_gguf(pq, toks);
    double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
    CHECK(maxdiff > 1e-4);   // bias demonstrably changes the output
}

// Shared Gemma-2 toy config (post-norms + soft-caps).
static ToyGgufConfig gemma2_cfg(uint32_t seed, float final_cap) {
    ToyGgufConfig c; c.arch = "gemma2"; c.post_norms = true;
    c.attn_softcap = 50.f; c.final_softcap = final_cap;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = seed;
    return c;
}

TEST(gemma2_config_discovery) {
    ToyGgufConfig c = gemma2_cfg(31, 30.f);
    std::string p = scratch("gemma2_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.arch_kind == Arch::Gemma2 && cfg.gemma_rmsnorm);
    APPROX(cfg.embedding_scale, std::sqrt(32.0), 1e-4);
    APPROX(cfg.attn_logit_softcap, 50.0, 1e-4);
    APPROX(cfg.final_logit_softcap, 30.0, 1e-4);
    CHECK(g.find("blk.0.post_attention_norm.weight") != nullptr);
    CHECK(g.find("blk.0.post_ffw_norm.weight") != nullptr);
}

TEST(gemma2_forward_finite_and_deterministic) {
    ToyGgufConfig c = gemma2_cfg(31, 30.f);
    std::string p = scratch("gemma2_fwd.gguf"); write_toy_gguf(p, c);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(p, toks), b = forward_gguf(p, toks);
    CHECK(a == b);                                  // deterministic
    for (float v : a) CHECK(std::isfinite(v));      // GeGLU + softcap path stable
}

TEST(gemma2_final_softcap_bounds_logits) {
    // An aggressive final cap of 1.0 must clamp every logit strictly inside
    // (-1, 1), since cap*tanh(.) < cap. Proves the final soft-cap is applied.
    ToyGgufConfig c = gemma2_cfg(31, 1.0f);
    std::string p = scratch("gemma2_cap.gguf"); write_toy_gguf(p, c);
    auto a = forward_gguf(p, {5, 2, 9, 1, 7});
    for (float v : a) CHECK(std::fabs(v) < 1.0f);
}

TEST(gemma3_config_discovery) {
    ToyGgufConfig c; c.arch = "gemma3"; c.post_norms = true; c.qk_norm = true;
    c.rope_local = 10000.f; c.rope_theta = 1000000.f; c.swa_pattern = 3;
    c.n_layers = 4; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = 41;
    std::string p = scratch("gemma3_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.arch_kind == Arch::Gemma3 && cfg.gemma_rmsnorm);
    APPROX(cfg.embedding_scale, std::sqrt(32.0), 1e-4);
    APPROX(cfg.rope_theta_local, 10000.0, 1e-3);
    CHECK(cfg.sliding_window_pattern == 3);
    CHECK(g.find("blk.0.attn_q_norm.weight") != nullptr);
    CHECK(g.find("blk.0.attn_k_norm.weight") != nullptr);
}

static ToyGgufConfig gemma3_base(uint32_t seed) {
    ToyGgufConfig c; c.arch = "gemma3"; c.post_norms = true;
    c.n_layers = 4; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = seed;
    return c;
}

TEST(gemma3_forward_finite_and_deterministic) {
    ToyGgufConfig c = gemma3_base(41); c.qk_norm = true;
    c.rope_local = 10000.f; c.rope_theta = 1000000.f; c.swa_pattern = 3;
    std::string p = scratch("gemma3_fwd.gguf"); write_toy_gguf(p, c);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(p, toks), b = forward_gguf(p, toks);
    CHECK(a == b);
    for (float v : a) CHECK(std::isfinite(v));
}

TEST(gemma3_qk_norm_changes_output) {
    // Same weights; QK-norm on vs off must move the logits (it renormalizes q/k).
    ToyGgufConfig off = gemma3_base(41);              // no qk_norm
    ToyGgufConfig on  = gemma3_base(41); on.qk_norm = true;
    std::string po = scratch("g3_noqk.gguf"), pn = scratch("g3_qk.gguf");
    write_toy_gguf(po, off); write_toy_gguf(pn, on);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(po, toks), b = forward_gguf(pn, toks);
    double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
    CHECK(maxdiff > 1e-4);
}

TEST(gemma3_local_rope_changes_output) {
    // Same weights; a distinct local RoPE base on sliding-window layers must
    // change the output vs using the single global base everywhere.
    ToyGgufConfig g = gemma3_base(41); g.qk_norm = true; g.rope_theta = 1000000.f;
    ToyGgufConfig same = g;                              // no local base -> all global
    ToyGgufConfig split = g; split.rope_local = 10000.f; split.swa_pattern = 3;
    std::string ps = scratch("g3_global.gguf"), pl = scratch("g3_local.gguf");
    write_toy_gguf(ps, same); write_toy_gguf(pl, split);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(ps, toks), b = forward_gguf(pl, toks);
    double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
    CHECK(maxdiff > 1e-4);
}

TEST(phi3_config_discovery) {
    ToyGgufConfig c; c.arch = "phi3"; c.fused_qkv = true;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = 51;
    std::string p = scratch("phi3_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.arch_kind == Arch::Phi3 && cfg.fused_qkv && cfg.fused_gate_up);
    CHECK(g.find("blk.0.attn_qkv.weight") != nullptr);
    CHECK(g.find("blk.0.attn_q.weight") == nullptr);   // fused, no separate q
}

TEST(phi3_full_rotary_matches_llama) {
    // Phi-3 with full rotary == Llama over the same weights: the fused
    // attn_qkv / ffn_up tensors are filled in the same RNG order as the split
    // tensors, and block_phi3 with rot=head_dim is exactly the Llama block.
    ToyGgufConfig cl; cl.arch = "llama";
    cl.n_layers = 3; cl.dim = 32; cl.n_heads = 4; cl.n_kv_heads = 2;
    cl.ffn_dim = 64; cl.vocab_size = 48; cl.seed = 51;
    ToyGgufConfig cp = cl; cp.arch = "phi3"; cp.fused_qkv = true;   // full rotary (no rope_dim)
    std::string pl = scratch("phi_llama.gguf"), pp = scratch("phi_phi3.gguf");
    write_toy_gguf(pl, cl); write_toy_gguf(pp, cp);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(pl, toks), b = forward_gguf(pp, toks);
    CHECK(a.size() == b.size() && !a.empty());
    for (size_t i = 0; i < a.size(); ++i) APPROX(a[i], b[i], 1e-4);
}

TEST(phi3_partial_rotary_changes_output) {
    // Same fused weights, but partial rotary (rope_dim < head_dim) must differ
    // from full rotary — only the first rope_dim dims of each head are rotated.
    ToyGgufConfig full; full.arch = "phi3"; full.fused_qkv = true;
    full.n_layers = 3; full.dim = 32; full.n_heads = 4; full.n_kv_heads = 2;
    full.ffn_dim = 64; full.vocab_size = 48; full.seed = 51;   // head_dim = 8
    ToyGgufConfig part = full; part.rope_dim = 4;              // rotate 4 of 8 dims
    std::string pf = scratch("phi_full.gguf"), pp = scratch("phi_part.gguf");
    write_toy_gguf(pf, full); write_toy_gguf(pp, part);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(pf, toks), b = forward_gguf(pp, toks);
    double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
    CHECK(maxdiff > 1e-4);
}

TEST(moe_config_discovery) {
    ToyGgufConfig c; c.arch = "llama"; c.n_experts = 4; c.n_experts_used = 2;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.seed = 61;
    std::string p = scratch("moe_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.is_moe() && cfg.n_experts == 4 && cfg.n_experts_used == 2);
    CHECK(g.find("blk.0.ffn_gate_inp.weight") != nullptr);
    CHECK(g.find("blk.0.ffn_gate_exps.weight") != nullptr);
    CHECK(g.find("blk.0.ffn_gate.weight") == nullptr);   // no dense FFN
}

TEST(moe_single_expert_equals_dense) {
    // A 1-expert MoE (top-1, weight 1.0) is exactly a dense FFN. The expert
    // tensors are filled in the same RNG order/size as the dense gate/up/down
    // and the router is emitted last, so the shared weights are identical =>
    // byte-identical logits.
    ToyGgufConfig cd; cd.arch = "llama";
    cd.n_layers = 3; cd.dim = 32; cd.n_heads = 4; cd.n_kv_heads = 2;
    cd.ffn_dim = 64; cd.vocab_size = 48; cd.seed = 61;
    ToyGgufConfig cm = cd; cm.n_experts = 1; cm.n_experts_used = 1;
    std::string pd = scratch("moe_dense.gguf"), pm = scratch("moe_one.gguf");
    write_toy_gguf(pd, cd); write_toy_gguf(pm, cm);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(pd, toks), b = forward_gguf(pm, toks);
    CHECK(a.size() == b.size() && !a.empty());
    for (size_t i = 0; i < a.size(); ++i) APPROX(a[i], b[i], 1e-4);
}

TEST(moe_expert_count_changes_output) {
    // Same weights; routing 1 expert vs all 4 must produce different logits.
    ToyGgufConfig base; base.arch = "llama"; base.n_experts = 4;
    base.n_layers = 3; base.dim = 32; base.n_heads = 4; base.n_kv_heads = 2;
    base.ffn_dim = 64; base.vocab_size = 48; base.seed = 61;
    ToyGgufConfig one = base; one.n_experts_used = 1;
    ToyGgufConfig all = base; all.n_experts_used = 4;
    std::string p1 = scratch("moe_k1.gguf"), p4 = scratch("moe_k4.gguf");
    write_toy_gguf(p1, one); write_toy_gguf(p4, all);
    std::vector<int64_t> toks = {5, 2, 9, 1, 7};
    auto a = forward_gguf(p1, toks), b = forward_gguf(p4, toks);
    double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
    CHECK(maxdiff > 1e-4);
}

int main() {
    printf("== test_arch ==\n");
    return llmtest::run_all();
}
