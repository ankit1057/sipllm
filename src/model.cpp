// model.cpp — ModelConfig discovery from a WeightSource's metadata.
#include "llm/model.h"
#include "llm/common.h"

#include <cstdio>

namespace llm {

// Try several candidate keys (GGUF long form first, then toy short form).
static bool first_int(const WeightSource& s, std::initializer_list<const char*> keys,
                      int64_t& out) {
    for (auto k : keys) if (s.has_meta(k)) { out = s.meta_int(k); return true; }
    return false;
}
static bool first_float(const WeightSource& s, std::initializer_list<const char*> keys,
                        double& out) {
    for (auto k : keys) if (s.has_meta(k)) { out = s.meta_float(k); return true; }
    return false;
}

ModelConfig ModelConfig::from_source(const WeightSource& src) {
    ModelConfig c;
    int64_t v = 0;
    double f = 0;

    if (first_int(src, {"llama.block_count", "block_count", "n_layers"}, v)) c.n_layers = v;
    if (first_int(src, {"llama.attention.head_count", "n_heads"}, v)) c.n_heads = v;
    if (first_int(src, {"llama.attention.head_count_kv", "n_kv_heads"}, v)) c.n_kv_heads = v;
    else c.n_kv_heads = c.n_heads;
    if (first_int(src, {"llama.embedding_length", "dim"}, v)) c.dim = v;
    if (first_int(src, {"llama.feed_forward_length", "ffn_dim"}, v)) c.ffn_dim = v;
    if (first_int(src, {"llama.context_length", "ctx_len"}, v)) c.ctx_len = v;
    if (first_float(src, {"llama.rope.freq_base", "rope_theta"}, f)) c.rope_theta = (float)f;
    if (first_float(src, {"llama.attention.layer_norm_rms_epsilon", "rms_eps"}, f))
        c.rms_eps = (float)f;

    // head_dim: explicit key, else dim / n_heads.
    if (first_int(src, {"llama.attention.key_length", "llama.rope.dimension_count", "head_dim"}, v))
        c.head_dim = v;
    else if (c.n_heads > 0)
        c.head_dim = c.dim / c.n_heads;

    // vocab: prefer token_embd's outer dim (authoritative), else metadata.
    if (const TensorInfo* te = src.find(names::token_embd))
        c.vocab_size = te->shape.empty() ? 0 : te->shape[0];
    if (c.vocab_size == 0 && first_int(src, {"llama.vocab_size"}, v)) c.vocab_size = v;

    // output tied to embeddings when there is no separate output.weight.
    c.tie_embeddings = (src.find(names::output) == nullptr);

    if (c.head_dim == 0 && c.n_heads > 0 && c.dim > 0) c.head_dim = c.dim / c.n_heads;
    return c;
}

std::string ModelConfig::summary() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "layers=%lld heads=%lld kv_heads=%lld dim=%lld head_dim=%lld "
             "ffn=%lld vocab=%lld ctx=%lld theta=%.1f eps=%.1e tied=%d",
             (long long)n_layers, (long long)n_heads, (long long)n_kv_heads,
             (long long)dim, (long long)head_dim, (long long)ffn_dim,
             (long long)vocab_size, (long long)ctx_len, rope_theta, rms_eps,
             (int)tie_embeddings);
    return buf;
}

} // namespace llm
