// test_gemma4.cpp — real Gemma 4 (dense) architecture support.
//
// A synthetic Gemma 4 GGUF that mirrors the 12B's layout at toy scale:
// per-layer head_count_kv / sliding_window_pattern ARRAYS, sliding-window
// layers with small heads and their own attn_v, a global layer with wider
// heads, a single kv head and NO attn_v (K doubles as V), proportional RoPE
// via rope_freqs.weight, and a per-layer layer_output_scale. The real model is
// golden-tested against llama.cpp (golden/); these tests pin the engine-side
// invariants that a 7 GB model can't run in CI:
//   * config discovery of the per-layer geometry,
//   * the sliding-window ring KV cache is bitwise-equivalent to a full cache,
//   * batched prefill == token-by-token forward,
//   * session save/load round-trips through a wrapped ring,
//   * the "gemma4" SPM-style BPE tokenizer (▁ escaping, newline runs, merges
//     by rank, <0xNN> byte fallback, literal special tokens).
#include "llm/gguf_writer.h"
#include "llm/runtime.h"
#include "llm/session.h"
#include "llm/tokenizer.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace llm;
using llmtest::check;

namespace {

constexpr int64_t kLayers = 4, kDim = 32, kHeads = 2, kFfn = 64, kVocab = 64;
constexpr int64_t kHdSwa = 8, kHdGlobal = 16, kWindow = 4;
const std::vector<int32_t> kSwa = {1, 1, 1, 0};     // layer 3 is global
const std::vector<int32_t> kKvHeads = {2, 2, 2, 1};

std::string write_toy_gemma4(const std::string& name, int64_t window = kWindow) {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    GgufWriter w;
    const std::string a = "gemma4.";
    w.str("general.architecture", "gemma4");
    w.u32(a + "block_count", kLayers);
    w.u32(a + "attention.head_count", kHeads);
    w.i32_array(a + "attention.head_count_kv", kKvHeads);
    w.u32(a + "embedding_length", kDim);
    w.u32(a + "feed_forward_length", kFfn);
    w.u32(a + "context_length", 256);
    w.f32(a + "rope.freq_base", 1e6f);
    w.f32(a + "rope.freq_base_swa", 1e4f);
    w.u32(a + "rope.dimension_count", kHdGlobal);
    w.u32(a + "rope.dimension_count_swa", kHdSwa);
    w.u32(a + "attention.key_length", kHdGlobal);
    w.u32(a + "attention.value_length", kHdGlobal);
    w.u32(a + "attention.key_length_swa", kHdSwa);
    w.u32(a + "attention.value_length_swa", kHdSwa);
    w.u32(a + "attention.sliding_window", (uint32_t)window);
    w.i32_array(a + "attention.sliding_window_pattern", kSwa);
    w.f32(a + "attention.layer_norm_rms_epsilon", 1e-6f);
    w.f32(a + "final_logit_softcapping", 30.f);
    w.u32("general.alignment", 32);

    auto mat = [&](const std::string& n, int64_t out, int64_t in) {
        std::vector<float> f((size_t)(out * in));
        for (auto& v : f) v = nd(rng);
        w.add_tensor(n, DType::F32, {out, in}, f.data(), f.size() * 4);
    };
    auto vec = [&](const std::string& n, int64_t len, bool random_gain) {
        std::vector<float> f((size_t)len, 1.f);
        if (random_gain) for (auto& v : f) v = ud(rng);
        w.add_tensor(n, DType::F32, {len}, f.data(), f.size() * 4);
    };
    mat(names::token_embd, kVocab, kDim);
    for (int64_t l = 0; l < kLayers; ++l) {
        const int64_t hd = kSwa[l] ? kHdSwa : kHdGlobal, kv = kKvHeads[l] * hd;
        vec(names::attn_norm(l), kDim, true);
        mat(names::attn_q(l), kHeads * hd, kDim);
        mat(names::attn_k(l), kv, kDim);
        if (kSwa[l]) mat(names::attn_v(l), kv, kDim);    // global: K doubles as V
        mat(names::attn_out(l), kDim, kHeads * hd);
        vec(names::blk(l, "attn_q_norm.weight"), hd, true);
        vec(names::blk(l, "attn_k_norm.weight"), hd, true);
        vec(names::blk(l, "post_attention_norm.weight"), kDim, true);
        vec(names::ffn_norm(l), kDim, true);
        mat(names::ffn_gate(l), kFfn, kDim);
        mat(names::ffn_up(l), kFfn, kDim);
        mat(names::ffn_down(l), kDim, kFfn);
        vec(names::blk(l, "post_ffw_norm.weight"), kDim, true);
        const float scale = 0.5f + 0.25f * (float)l;
        w.add_tensor(names::blk(l, "layer_output_scale.weight"), DType::F32, {1}, &scale, 4);
    }
    vec(names::output_norm, kDim, true);
    // Proportional RoPE for the global layer: rotate the first 2 of 8 pairs.
    std::vector<float> rf((size_t)kHdGlobal / 2, 1e30f);
    rf[0] = rf[1] = 1.f;
    w.add_tensor("rope_freqs.weight", DType::F32, {(int64_t)rf.size()}, rf.data(), rf.size() * 4);
    std::string path = llmtest::scratch_path(name);
    w.write(path);
    return path;
}

std::vector<float> run_tokens(WeightSource& src, const ModelConfig& cfg, KVCache& kv,
                              const std::vector<int64_t>& toks, bool batched) {
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    LayerLoader loader(&src, cfg, opt);
    Transformer tf(&loader, &kv, nullptr);
    const float* lg = nullptr;
    if (batched) lg = tf.prefill(toks.data(), (int64_t)toks.size(), 0);
    else for (size_t i = 0; i < toks.size(); ++i) lg = tf.forward(toks[i], (int64_t)i);
    return std::vector<float>(lg, lg + cfg.vocab_size);
}

std::vector<int64_t> seq(int n) {
    std::vector<int64_t> t;
    for (int i = 0; i < n; ++i) t.push_back((i * 7 + 3) % kVocab);
    return t;
}

KVCache ring_cache(const ModelConfig& cfg, int64_t ctx) {
    std::vector<int64_t> dims, wins;
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
        dims.push_back(cfg.kv_dim_at(l));
        wins.push_back(cfg.layer_swa[(size_t)l] ? cfg.sliding_window : 0);
    }
    return KVCache(dims, wins, ctx);
}

} // namespace

TEST(gemma4_per_layer_config_discovery) {
    auto src = open_model(write_toy_gemma4("g4_cfg.gguf"));
    ModelConfig c = ModelConfig::from_source(*src);
    check(c.arch_kind == Arch::Gemma4, "arch");
    check(c.layer_swa == std::vector<uint8_t>({1, 1, 1, 0}), "swa pattern");
    check(c.head_dim_at(0) == kHdSwa && c.head_dim_at(3) == kHdGlobal, "per-layer head dim");
    check(c.kv_heads_at(0) == 2 && c.kv_heads_at(3) == 1, "per-layer kv heads");
    check(c.n_kv_heads == 2, "scalar kv heads = widest (never 0)");
    check(c.kv_dim() == 16 && c.q_dim() == kHeads * kHdGlobal, "max dims size buffers");
    check(c.is_swa_layer(0) && !c.is_swa_layer(3), "is_swa_layer");
    check(c.rope_theta == 1e6f && c.rope_theta_local == 1e4f, "dual rope base");
    check(c.rope_freq_factors.size() == (size_t)kHdGlobal / 2, "rope_freqs read");
    check(c.attn_scale == 1.0f, "attention scale 1.0");
    const BlockSpec& b = c.block_spec;
    check(b.norm == NormKind::RMSNorm && b.plain_norm_weights, "plain RMSNorm (no 1+w)");
    check(b.rope_pairing == RopePairing::NeoX && b.v_norm && b.layer_out_scale, "gemma4 flags");
    check(b.ffn == FfnKind::GeGLU && c.final_logit_softcap == 30.f, "geglu + softcap");
    // Ring layers hold at most `window` positions; the global layer holds all.
    const size_t want = 3 * 2 * kWindow * 16 * 4 + 2 * 100 * 16 * 4;
    check(c.kv_cache_bytes(100, false) == want, "kv_cache_bytes per-layer + ring");
}

TEST(gemma4_ring_kv_bitwise_equals_full_cache) {
    auto src = open_model(write_toy_gemma4("g4_ring.gguf"));
    ModelConfig cfg = ModelConfig::from_source(*src);
    const auto toks = seq(19);   // wraps the window-4 ring several times
    KVCache full(cfg.n_layers, cfg.kv_dim(), 64);
    KVCache ring = ring_cache(cfg, 64);
    check(ring.has_ring() && !full.has_ring(), "ring layout");
    auto a = run_tokens(*src, cfg, full, toks, false);
    auto b = run_tokens(*src, cfg, ring, toks, false);
    check(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "ring logits != full-cache logits");
    for (float v : a) check(std::isfinite(v), "non-finite logit");
    check(ring.bytes() < full.bytes(), "ring must use less memory");
}

TEST(gemma4_prefill_equals_token_by_token) {
    auto src = open_model(write_toy_gemma4("g4_prefill.gguf"));
    ModelConfig cfg = ModelConfig::from_source(*src);
    const auto toks = seq(41);   // > one 32-position prefill chunk, wraps the ring
    KVCache r1 = ring_cache(cfg, 64), r2 = ring_cache(cfg, 64);
    auto a = run_tokens(*src, cfg, r1, toks, false);
    auto b = run_tokens(*src, cfg, r2, toks, true);
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i]));
    llmtest::approx(d, 0.0, 1e-4, "prefill vs forward max|Δ|");
}

TEST(gemma4_sliding_window_only_on_swa_layers) {
    // Shrinking the window changes the output once the sequence exceeds it...
    auto s4 = open_model(write_toy_gemma4("g4_w4.gguf", 4));
    auto s64 = open_model(write_toy_gemma4("g4_w64.gguf", 64));
    ModelConfig c4 = ModelConfig::from_source(*s4), c64 = ModelConfig::from_source(*s64);
    KVCache k4 = ring_cache(c4, 64), k64 = ring_cache(c64, 64);
    auto a = run_tokens(*s4, c4, k4, seq(12), false);
    auto b = run_tokens(*s64, c64, k64, seq(12), false);
    check(std::memcmp(a.data(), b.data(), a.size() * 4) != 0, "window had no effect");
    // ...but not while the whole sequence still fits inside it.
    KVCache k4b = ring_cache(c4, 64), k64b = ring_cache(c64, 64);
    auto c = run_tokens(*s4, c4, k4b, seq(4), false);
    auto d = run_tokens(*s64, c64, k64b, seq(4), false);
    check(std::memcmp(c.data(), d.data(), c.size() * 4) == 0, "window affected a short sequence");
}

TEST(gemma4_kv_ring_rewind_rules) {
    KVCache kv({8, 8}, {4, 0}, 64);
    check(kv.can_rewind(0, 4), "no wrap yet: any rewind ok");
    check(kv.can_rewind(9, 10) && kv.can_rewind(10, 10), "rewind by <= 1 after wrap ok");
    check(!kv.can_rewind(5, 10), "rewind past the ring after wrap must be refused");
    check(kv.first_valid(0, 10) == 6 && kv.first_valid(1, 10) == 0, "first_valid");
    check(kv.kv_dim() == 8 && kv.window(0) == 4 && kv.window(1) == 0, "layout accessors");
}

TEST(gemma4_session_roundtrip_through_wrapped_ring) {
    auto src = open_model(write_toy_gemma4("g4_sess.gguf"));
    ModelConfig cfg = ModelConfig::from_source(*src);
    const auto toks = seq(11);
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    LayerLoader loader(src.get(), cfg, opt);

    // Reference: one uninterrupted run over all 11 tokens.
    KVCache ref = ring_cache(cfg, 64);
    Transformer tref(&loader, &ref, nullptr);
    const float* lr = nullptr;
    for (size_t i = 0; i < toks.size(); ++i) lr = tref.forward(toks[i], (int64_t)i);
    std::vector<float> want(lr, lr + cfg.vocab_size);

    // Run 9 tokens (ring wrapped), save, load into a fresh cache, finish.
    KVCache a = ring_cache(cfg, 64);
    Transformer ta(&loader, &a, nullptr);
    for (int64_t i = 0; i < 9; ++i) ta.forward(toks[(size_t)i], i);
    const std::string path = llmtest::scratch_path("g4_sess.bin");
    std::vector<int64_t> committed(toks.begin(), toks.begin() + 9);
    check(session_write(path, committed, a, 9, 42), "session_write");
    KVCache b = ring_cache(cfg, 64);
    std::vector<int64_t> got_toks; int64_t n = 0;
    check(session_read(path, got_toks, b, &n, 42) && n == 9 && got_toks == committed, "session_read");
    Transformer tb(&loader, &b, nullptr);
    const float* lb = nullptr;
    for (int64_t i = 9; i < 11; ++i) lb = tb.forward(toks[(size_t)i], i);
    check(std::memcmp(lb, want.data(), want.size() * 4) == 0, "resumed logits differ");
}

TEST(gemma4_spm_bpe_tokenizer) {
    // Vocab: specials, a byte-fallback token, pieces with ▁, and newline runs.
    const std::string SP = "\xE2\x96\x81";
    std::vector<std::string> toks = {"<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>",
                                     "<0x5A>", "a", "b", "ab", SP, SP + "ab", "\n", "\n\n",
                                     SP + SP};
    std::vector<int32_t> types = {3, 1, 3, 3, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1};
    GgufWriter w;
    w.str("general.architecture", "gemma4");
    w.str("tokenizer.ggml.model", "gemma4");
    w.str_array("tokenizer.ggml.tokens", toks);
    w.i32_array("tokenizer.ggml.token_type", types);
    w.str_array("tokenizer.ggml.merges", {"a b", SP + " ab", SP + " " + SP});
    w.u32("tokenizer.ggml.bos_token_id", 2);
    w.u32("tokenizer.ggml.eos_token_id", 4);
    const std::string path = llmtest::scratch_path("g4_tok.gguf");
    w.write(path);
    auto src = open_model(path);
    Tokenizer tk = Tokenizer::from_source(*src);
    check(tk.kind() == Tokenizer::Kind::SpmBpe, "kind");

    // "ab ab": no dummy prefix; ' '->▁; "a b" (rank 0) merges before "▁ ab".
    auto ids = tk.encode("ab ab", true);
    check(ids == std::vector<int64_t>({2, 8, 10}), "ab ab");
    // newline runs are their own chunks and map to a whole token.
    ids = tk.encode("ab\n\nab", false);
    check(ids == std::vector<int64_t>({8, 12, 8}), "newline run");
    // unknown byte -> <0xNN>; special tokens are matched literally.
    ids = tk.encode("<|turn>Z<turn|>", false);
    check(ids == std::vector<int64_t>({3, 5, 4}), "specials + byte fallback");
    // "  ab" -> ▁▁ab: rank order merges "a b" then "▁ ab" (rank 1) before
    // "▁ ▁" (rank 2) is ever adjacent; decode restores the text.
    ids = tk.encode("  ab", false);
    check(ids == std::vector<int64_t>({9, 10}), "double space (rank order)");
    ids = tk.encode("  ", false);
    check(ids == std::vector<int64_t>({13}), "space run merges to ▁▁");
    check(tk.decode(tk.encode("ab ab\n\nZ", false)) == "ab ab\n\nZ", "decode roundtrip");
    check(tk.is_eog(4) && tk.is_eog(1), "<turn|> and <eos> end generation");
}

int main() {
    printf("== test_gemma4 ==\n");
    return llmtest::run_all();
}
