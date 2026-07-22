// toy_model.h — generate a tiny random Llama-shaped model in .llmw format.
//
// Phase 4: "Instead of Llama-7B, build a transformer with 2 layers, hidden=32,
// heads=2, random weights. If inference works entirely through LayerLoader,
// you've proven the architecture." This header is that generator, shared by the
// make_toy_model tool and the end-to-end tests so the fixture is identical.
#pragma once

#include "llm/format.h"
#include "llm/model.h"

#include <random>
#include <vector>

namespace llm {

struct ToyConfig {
    int64_t n_layers   = 2;
    int64_t dim        = 32;
    int64_t n_heads    = 2;
    int64_t n_kv_heads = 2;      // set < n_heads to exercise GQA
    int64_t ffn_dim    = 64;
    int64_t vocab_size = 48;
    int64_t ctx_len    = 128;
    float   rope_theta = 10000.f;
    float   rms_eps    = 1e-5f;
    bool    tied       = false;  // omit output.weight -> tied to embeddings
    uint32_t seed      = 1234;
};

inline void write_toy_model(const std::string& path, const ToyConfig& c) {
    const int64_t head_dim = c.dim / c.n_heads;
    const int64_t q_dim  = c.n_heads * head_dim;
    const int64_t kv_dim = c.n_kv_heads * head_dim;

    std::mt19937 rng(c.seed);
    std::normal_distribution<float> nd(0.f, 0.02f);
    auto randbuf = [&](int64_t n) {
        std::vector<float> b(n);
        for (auto& v : b) v = nd(rng);
        return b;
    };
    auto ones = [&](int64_t n) { return std::vector<float>(n, 1.0f); };

    ModelWriter w;
    w.set_meta("arch", std::string("llama"));
    w.set_meta("n_layers", c.n_layers);
    w.set_meta("n_heads", c.n_heads);
    w.set_meta("n_kv_heads", c.n_kv_heads);
    w.set_meta("dim", c.dim);
    w.set_meta("head_dim", head_dim);
    w.set_meta("ffn_dim", c.ffn_dim);
    w.set_meta("ctx_len", c.ctx_len);
    w.set_meta("rope_theta", (double)c.rope_theta);
    w.set_meta("rms_eps", (double)c.rms_eps);

    { auto b = randbuf(c.vocab_size * c.dim);
      w.add_f32(names::token_embd, {c.vocab_size, c.dim}, b.data()); }

    for (int64_t i = 0; i < c.n_layers; ++i) {
        { auto b = ones(c.dim);           w.add_f32(names::attn_norm(i), {c.dim}, b.data()); }
        { auto b = randbuf(q_dim * c.dim);  w.add_f32(names::attn_q(i), {q_dim, c.dim}, b.data()); }
        { auto b = randbuf(kv_dim * c.dim); w.add_f32(names::attn_k(i), {kv_dim, c.dim}, b.data()); }
        { auto b = randbuf(kv_dim * c.dim); w.add_f32(names::attn_v(i), {kv_dim, c.dim}, b.data()); }
        { auto b = randbuf(c.dim * q_dim);  w.add_f32(names::attn_out(i), {c.dim, q_dim}, b.data()); }
        { auto b = ones(c.dim);           w.add_f32(names::ffn_norm(i), {c.dim}, b.data()); }
        { auto b = randbuf(c.ffn_dim * c.dim); w.add_f32(names::ffn_gate(i), {c.ffn_dim, c.dim}, b.data()); }
        { auto b = randbuf(c.ffn_dim * c.dim); w.add_f32(names::ffn_up(i), {c.ffn_dim, c.dim}, b.data()); }
        { auto b = randbuf(c.dim * c.ffn_dim); w.add_f32(names::ffn_down(i), {c.dim, c.ffn_dim}, b.data()); }
    }

    { auto b = ones(c.dim); w.add_f32(names::output_norm, {c.dim}, b.data()); }
    if (!c.tied) {
        auto b = randbuf(c.vocab_size * c.dim);
        w.add_f32(names::output, {c.vocab_size, c.dim}, b.data());
    }
    w.write(path);
}

} // namespace llm
