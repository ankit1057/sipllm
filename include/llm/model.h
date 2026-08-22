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
    Gemma3,    // Gemma2 shape + QK-norm + per-layer local/global RoPE, no softcap
    Gemma4,    // Gemma 4 arch (Gemma2/3 evolution with adaptive RMSNorm & scaling)
    Kimi,      // Moonshot Kimi / Kimi-k1.5 (SwiGLU + GQA + RMSNorm / ChatML)
    DeepSeek,  // DeepSeek V2 / V3 / R1 (MLA / SwiGLU / MoE or dense)
    Yi,        // 01.AI Yi / Yi-1.5 (Llama-like SwiGLU + RMSNorm)
    Baichuan,  // Baichuan 2 (Alibi/RoPE + SwiGLU)
    InternLM2, // InternLM 2 / 2.5 (Llama-like + custom QKV layout)
    GLM4,      // Zhipu GLM-4 / ChatGLM (LayerNorm/RMSNorm + SwiGLU)
    Phi3,      // Llama-like + fused QKV, fused gate/up, partial-rotary RoPE
    Phi4,      // Microsoft Phi-4 (Llama-like + fused QKV / gate+up)
    Phi2,      // parallel block + LayerNorm + partial-rotary RoPE + GELU MLP
    GPT2,      // LayerNorm + learned positional embeddings + GELU MLP, no RoPE
    Unknown,   // recognized string but no dedicated block yet -> treated as Llama
};

Arch arch_from_name(const std::string& name);
const char* arch_name(Arch a);

// Normalization applied at a block's sublayer inputs and the final norm.
enum class NormKind {
    RMSNorm,        // x / rms(x) * w                       (Llama, Mistral, Qwen2, Phi3)
    RMSNormGemma,   // x / rms(x) * (1 + w)                 (Gemma 2 / 3)
    LayerNorm,      // (x - mean) / std * w + b             (GPT-2, Phi-2)
};

// Feed-forward network structure.
enum class FfnKind {
    SwiGLU,   // down(silu(gate(x)) * up(x))                (Llama family)
    GeGLU,    // down(gelu(gate(x)) * up(x))                (Gemma)
    GeluMLP,  // down(gelu(up(x)))  — non-gated             (GPT-2, Phi-2)
};

// Rotary position embedding mode.
enum class RopeKind {
    None,          // no RoPE; positions come from a learned table (GPT-2)
    Full,          // rotate all head_dim dims
    Partial,       // rotate only the first rope_dim dims/head (Phi-2/3)
    Llama3Scaled,  // Full + per-wavelength frequency stretch (Llama 3.x)
};

const char* norm_kind_name(NormKind);
const char* ffn_kind_name(FfnKind);
const char* rope_kind_name(RopeKind);

// Declarative recipe for one transformer block. This is the single source of
// truth for what a block does; the executor's dispatch is a pure function of
// these fields. Deriving it from the architecture (below) replaces the scattered
// `if (arch_kind == ...)` checks with data (#44).
struct BlockSpec {
    NormKind norm = NormKind::RMSNorm;   // attn/ffn input norm
    bool     qkv_fused = false;          // one attn_qkv tensor split into q,k,v (Phi)
    bool     qkv_bias  = false;          // separate q/k/v projection biases (Qwen2)
    bool     qk_norm   = false;          // per-head q/k norm before RoPE (Gemma3)
    RopeKind rope = RopeKind::Full;
    int64_t  rope_dim = 0;               // rotary dims/head when rope == Partial
    bool     rope_dual_base = false;     // separate local/global RoPE base (Gemma3)
    bool     attn_softcap = false;       // cap attention logits before softmax (Gemma2)
    bool     post_attn_norm = false;     // norm the attn output before the residual (Gemma2)
    FfnKind  ffn = FfnKind::SwiGLU;
    bool     ffn_fused_gate_up = false;  // ffn_up packs [gate; up] (Phi3)
    bool     post_ffn_norm = false;      // norm the ffn output before the residual (Gemma2)
    bool     parallel_residual = false;  // attn & ffn read one shared norm, both added (Phi2)
    bool     proj_bias = false;          // attn_output / ffn projections are biased (GPT-2, Phi-2)
    bool     moe = false;                // router + top-k expert FFNs (Mixtral)
    int64_t  n_experts = 0;              // total experts (MoE)
    int64_t  n_experts_used = 0;         // experts per token (MoE)
};

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

    // The data-driven BlockSpec for this model (Issue #44).
    BlockSpec block_spec;

    // Remaining global configuration flags (not specific to a single block).
    float   embedding_scale  = 1.f;    // token embeddings *= scale (Gemma: sqrt(dim))
    float   attn_logit_softcap = 0.f;  // cap on attention scores (Gemma2 ~50)
    float   final_logit_softcap = 0.f; // cap on output logits (Gemma2 ~30)
    float   query_pre_attn_scalar = 0.f; // attn scale denom; 0 => head_dim
    float   rope_theta_local = 0.f;      // RoPE base for local (sliding) layers
    int64_t sliding_window_pattern = 0;  // global layer every Nth (Gemma3: 6)
    int64_t sliding_window = 0;          // max context attention size (Mistral: 4096)
    bool    learned_pos_emb  = false;   // add position_embd[pos] to the embedding
    float   layernorm_eps    = 1e-5f;   // LayerNorm epsilon

    bool is_moe() const { return block_spec.moe; }

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
