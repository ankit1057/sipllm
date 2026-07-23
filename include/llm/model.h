// model.h — model hyperparameters, tensor naming, and the weight reference the
// transformer consumes.
//
// The transformer is written once against WeightRef and never learns whether a
// weight is fp32 (toy model / small configs) or block-quantized (real GGUF).
// LayerLoader decides residency; the forward pass just calls linear().
#pragma once

#include "llm/weight_source.h"

#include <string>

namespace llm {

// Decoder architecture family. Resolved from GGUF `general.architecture`. The
// engine started Llama-only; this enum is the dispatch seam every other
// architecture hangs off of (see Transformer::block).
enum class Arch {
    Llama,     // RMSNorm + RoPE + GQA + SwiGLU, no biases, no softcap (reference)
    Mistral,   // structurally identical to Llama (sliding-window attn not modeled)
    Qwen2,     // Llama-like + bias on q/k/v projections (Qwen2 / Qwen2.5)
    Gemma2,    // GeGLU, (1+w) RMSNorm, pre+post norms, embd scale, logit softcap
    Unknown,   // recognized string but no dedicated block yet -> treated as Llama
};

Arch arch_from_name(const std::string& name);
const char* arch_name(Arch a);

// Decoder configuration. GQA is expressed via n_kv_heads; when it equals
// n_heads there is no grouping (Llama-2 7B), when smaller there is (Llama-3 8B
// uses 8 kv heads for 32 query heads).
struct ModelConfig {
    // Architecture identity. `arch` is the raw GGUF string (e.g. "llama",
    // "qwen2"); `arch_kind` is the resolved dispatch enum. Defaults keep every
    // existing Llama/toy model on the unchanged reference path.
    std::string arch = "llama";
    Arch        arch_kind = Arch::Llama;

    int64_t n_layers   = 0;
    int64_t n_heads    = 0;
    int64_t n_kv_heads = 0;
    int64_t dim        = 0;    // hidden size / embedding length
    int64_t head_dim   = 0;    // usually dim / n_heads
    int64_t ffn_dim    = 0;    // feed-forward intermediate size
    int64_t vocab_size = 0;
    int64_t ctx_len    = 0;    // trained context length
    float   rope_theta = 10000.f;
    float   rms_eps    = 1e-5f;
    bool    tie_embeddings = false;   // output projection shares token_embd

    // "llama3" RoPE frequency scaling (Llama-3.x). The trained short-context
    // RoPE frequencies are stretched per-wavelength so the model generalizes to
    // its long context. Empty type => plain RoPE (all pre-Llama3 models).
    std::string rope_scaling_type;                 // "", "llama3", "linear", ...
    float       rope_scale_factor     = 8.f;       // "factor"
    float       rope_low_freq_factor  = 1.f;       // "low_freq_factor"
    float       rope_high_freq_factor = 4.f;       // "high_freq_factor"
    int64_t     rope_orig_ctx_len     = 0;         // "original_context_length"

    bool use_llama3_rope() const {
        return rope_scaling_type == "llama3" && rope_scale_factor > 0.f &&
               rope_high_freq_factor != rope_low_freq_factor && rope_orig_ctx_len > 0;
    }

    // Gemma-family knobs (0 / 1.0 defaults => inert for every other arch).
    bool    gemma_rmsnorm    = false;  // learned scale is (1 + weight)
    float   embedding_scale  = 1.f;    // token embeddings *= scale (Gemma: sqrt(dim))
    float   attn_logit_softcap  = 0.f; // cap on attention scores (Gemma2 ~50)
    float   final_logit_softcap = 0.f; // cap on output logits (Gemma2 ~30)
    float   query_pre_attn_scalar = 0.f; // attn scale denom; 0 => head_dim

    int64_t q_dim()  const { return n_heads * head_dim; }
    int64_t kv_dim() const { return n_kv_heads * head_dim; }
    int64_t gqa_group() const { return n_heads / n_kv_heads; }

    // Build from a WeightSource's metadata, accepting both GGUF ("<arch>.*")
    // keys and the toy .llmw short keys. Falls back to tensor shapes where a
    // hyperparameter is missing.
    static ModelConfig from_source(const WeightSource& src);

    std::string summary() const;
};

// GGUF-compatible tensor names. Using the exact llama.cpp naming everywhere
// means the toy writer and the real GGUF loader are interchangeable.
namespace names {
inline std::string blk(int64_t i, const char* suffix) {
    return "blk." + std::to_string(i) + "." + suffix;
}
inline std::string attn_norm(int64_t i)   { return blk(i, "attn_norm.weight"); }
inline std::string attn_q(int64_t i)      { return blk(i, "attn_q.weight"); }
inline std::string attn_k(int64_t i)      { return blk(i, "attn_k.weight"); }
inline std::string attn_v(int64_t i)      { return blk(i, "attn_v.weight"); }
inline std::string attn_out(int64_t i)    { return blk(i, "attn_output.weight"); }
inline std::string ffn_norm(int64_t i)    { return blk(i, "ffn_norm.weight"); }
inline std::string ffn_gate(int64_t i)    { return blk(i, "ffn_gate.weight"); }
inline std::string ffn_up(int64_t i)      { return blk(i, "ffn_up.weight"); }
inline std::string ffn_down(int64_t i)    { return blk(i, "ffn_down.weight"); }
constexpr const char* token_embd = "token_embd.weight";
constexpr const char* output_norm = "output_norm.weight";
constexpr const char* output = "output.weight";
} // namespace names

// A resident weight: pointer to bytes (fp32 or quantized), its type, and the
// logical [n_out, n_in] shape. For 1-D norm weights n_out==1 and data is fp32.
struct WeightRef {
    const void* data   = nullptr;
    DType       dtype  = DType::F32;
    int64_t     n_out  = 0;
    int64_t     n_in   = 0;
    bool valid() const { return data != nullptr; }
};

} // namespace llm
