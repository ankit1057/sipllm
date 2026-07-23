// model.cpp — ModelConfig discovery from a WeightSource's metadata.
#include "llm/model.h"
#include "llm/common.h"

#include <cstdio>

namespace llm {

Arch arch_from_name(const std::string& name) {
    if (name == "llama")   return Arch::Llama;
    if (name == "mistral") return Arch::Mistral;
    if (name == "qwen2")   return Arch::Qwen2;
    return Arch::Unknown;
}

const char* arch_name(Arch a) {
    switch (a) {
        case Arch::Llama:   return "llama";
        case Arch::Mistral: return "mistral";
        case Arch::Qwen2:   return "qwen2";
        case Arch::Unknown: return "unknown";
    }
    return "unknown";
}

// Try several candidate keys in order; first present one wins.
static bool first_int(const WeightSource& s, std::initializer_list<std::string> keys,
                      int64_t& out) {
    for (const auto& k : keys) if (s.has_meta(k)) { out = s.meta_int(k); return true; }
    return false;
}
static bool first_float(const WeightSource& s, std::initializer_list<std::string> keys,
                        double& out) {
    for (const auto& k : keys) if (s.has_meta(k)) { out = s.meta_float(k); return true; }
    return false;
}

ModelConfig ModelConfig::from_source(const WeightSource& src) {
    ModelConfig c;
    int64_t v = 0;
    double f = 0;

    // Architecture identity. GGUF namespaces every hyperparameter under
    // "<arch>.*" (llama.block_count, qwen2.block_count, ...). Resolving keys by
    // the declared architecture — with a "llama.*" fallback for cross-arch
    // shared names and the toy short keys — makes config discovery work for any
    // architecture without a per-arch reader. Toy .llmw files omit the meta and
    // default to "llama", so their behavior is unchanged.
    c.arch = src.meta_str("general.architecture", "llama");
    c.arch_kind = arch_from_name(c.arch);
    const std::string& a = c.arch;                       // e.g. "llama", "qwen2"
    auto K = [&](const char* suffix) { return a + "." + suffix; };

    if (first_int(src, {K("block_count"), "llama.block_count", "block_count", "n_layers"}, v)) c.n_layers = v;
    if (first_int(src, {K("attention.head_count"), "llama.attention.head_count", "n_heads"}, v)) c.n_heads = v;
    if (first_int(src, {K("attention.head_count_kv"), "llama.attention.head_count_kv", "n_kv_heads"}, v)) c.n_kv_heads = v;
    else c.n_kv_heads = c.n_heads;
    if (first_int(src, {K("embedding_length"), "llama.embedding_length", "dim"}, v)) c.dim = v;
    if (first_int(src, {K("feed_forward_length"), "llama.feed_forward_length", "ffn_dim"}, v)) c.ffn_dim = v;
    if (first_int(src, {K("context_length"), "llama.context_length", "ctx_len"}, v)) c.ctx_len = v;
    if (first_float(src, {K("rope.freq_base"), "llama.rope.freq_base", "rope_theta"}, f)) c.rope_theta = (float)f;
    if (first_float(src, {K("attention.layer_norm_rms_epsilon"),
                          "llama.attention.layer_norm_rms_epsilon", "rms_eps"}, f))
        c.rms_eps = (float)f;

    // head_dim: explicit key, else dim / n_heads.
    if (first_int(src, {K("attention.key_length"), K("rope.dimension_count"),
                        "llama.attention.key_length", "llama.rope.dimension_count", "head_dim"}, v))
        c.head_dim = v;
    else if (c.n_heads > 0)
        c.head_dim = c.dim / c.n_heads;

    // vocab: prefer token_embd's outer dim (authoritative), else metadata.
    if (const TensorInfo* te = src.find(names::token_embd))
        c.vocab_size = te->shape.empty() ? 0 : te->shape[0];
    if (c.vocab_size == 0 && first_int(src, {K("vocab_size"), "llama.vocab_size"}, v)) c.vocab_size = v;

    // output tied to embeddings when there is no separate output.weight.
    c.tie_embeddings = (src.find(names::output) == nullptr);

    // llama3 RoPE frequency scaling (Llama-3.x). Keys are namespaced under the
    // architecture, e.g. "llama.rope.scaling.type" = "llama3" plus factor,
    // low/high freq factors, and the original (pre-scaling) context length.
    c.rope_scaling_type = src.meta_str(K("rope.scaling.type"),
                          src.meta_str("llama.rope.scaling.type", ""));
    if (first_float(src, {K("rope.scaling.factor"), "llama.rope.scaling.factor"}, f))
        c.rope_scale_factor = (float)f;
    if (first_float(src, {K("rope.scaling.low_freq_factor"),
                          "llama.rope.scaling.low_freq_factor"}, f))
        c.rope_low_freq_factor = (float)f;
    if (first_float(src, {K("rope.scaling.high_freq_factor"),
                          "llama.rope.scaling.high_freq_factor"}, f))
        c.rope_high_freq_factor = (float)f;
    if (first_int(src, {K("rope.scaling.original_context_length"),
                        "llama.rope.scaling.original_context_length"}, v))
        c.rope_orig_ctx_len = v;

    if (c.head_dim == 0 && c.n_heads > 0 && c.dim > 0) c.head_dim = c.dim / c.n_heads;
    return c;
}

std::string ModelConfig::summary() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "arch=%s layers=%lld heads=%lld kv_heads=%lld dim=%lld head_dim=%lld "
             "ffn=%lld vocab=%lld ctx=%lld theta=%.1f eps=%.1e tied=%d",
             arch.c_str(), (long long)n_layers, (long long)n_heads, (long long)n_kv_heads,
             (long long)dim, (long long)head_dim, (long long)ffn_dim,
             (long long)vocab_size, (long long)ctx_len, rope_theta, rms_eps,
             (int)tie_embeddings);
    return buf;
}

} // namespace llm
