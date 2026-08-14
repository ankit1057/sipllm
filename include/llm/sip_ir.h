// sip_ir.h — Sip IR: SipLLM's stable, backend-agnostic model representation.
//
// The runtime should not be "a GGUF loader." Sip IR is the seam that makes it a
// platform: importers (GGUF today; Hugging Face / ONNX / PyTorch tomorrow) all
// produce the SAME `SipModel`, and the executor + memory planner + backends
// consume only that. Nothing downstream of the IR knows what file format the
// model came from.
//
// A `SipModel` is three things, all resolved from a `WeightSource`:
//   1. config      — the hyperparameters the executor already used (ModelConfig).
//   2. block plan  — a DECLARATIVE recipe for one transformer block (the
//                    data-driven form of Transformer::block_* — see #44). Every
//                    per-architecture quirk (norm kind, fused/biased QKV, RoPE
//                    mode, soft-cap, GeGLU vs SwiGLU, pre/post norms, parallel
//                    residual, MoE) is a field, not a hand-written function.
//   3. tensor schema — which per-role tensors the model actually contains, with
//                    dtype/shape, probed against the source.
// Plus a tokenizer descriptor and a stable JSON serialization (the inspect /
// trace primitive).
//
// Sip IR v1 is ADDITIVE and READ-ONLY: importing a model does not change a
// single number in the forward pass. It formalizes what the engine already does
// so the platform layers (Phase 3 importers, Phase 4 backends) have one stable
// contract to build on.
#pragma once

#include "llm/loader.h"        // Role
#include "llm/model.h"
#include "llm/weight_source.h"

#include <string>
#include <vector>

namespace llm {

// Bumped only on an incompatible change to the SipModel shape below. Consumers
// pin against this; importers stamp it into every model they emit.
constexpr int kSipIRVersion = 1;

// NormKind / FfnKind / RopeKind and their *_kind_name() helpers now live in
// model.h as part of the BlockSpec refactor (#44); this header consumes them
// through the include above rather than redefining them.

// Declarative recipe for one transformer block. This is the single source of
// truth for what a block does; the executor's dispatch is a pure function of
// these fields. Deriving it from the architecture (below) replaces the scattered
// `if (arch_kind == ...)` checks with data (#44).
struct SipBlockPlan {
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

// One per-role tensor the model contains, resolved against the source.
struct SipTensor {
    Role                 role;
    std::string          name;    // resolved GGUF name, e.g. "blk.0.attn_q.weight"
    DType                dtype = DType::F32;
    std::vector<int64_t> shape;
    bool                 present = false;
};

// Tokenizer descriptor (pipeline + vocab + special ids).
struct SipTokenizer {
    std::string kind;             // "byte" | "spm" | "bpe"
    int64_t     vocab_size = 0;
    int64_t     bos = -1, eos = -1;
};

// The Sip IR model. Produced by an importer, consumed by the executor.
struct SipModel {
    int          ir_version   = kSipIRVersion;
    std::string  arch;            // raw architecture string ("llama", "qwen2", ...)
    Arch         arch_kind = Arch::Llama;
    std::string  source_format;   // which importer produced this ("gguf" | "llmw")

    ModelConfig  config;          // resolved hyperparameters (the executor's view)
    SipBlockPlan block;           // declarative per-block recipe
    NormKind     final_norm = NormKind::RMSNorm;
    bool         tied_embeddings = false;
    bool         learned_pos_emb = false;
    float        embedding_scale = 1.f;   // token embeddings *= scale (Gemma: sqrt(dim))
    float        final_logit_softcap = 0.f;

    SipTokenizer tokenizer;
    std::vector<SipTensor> block_tensors;   // layer-0 roles that are present
    std::vector<SipTensor> global_tensors;  // token_embd, output_norm, output, position_embd
    size_t       bytes_per_layer = 0;       // resident weight bytes for one block (quantized)
    int64_t      total_tensors = 0;         // tensors in the source directory

    // Stable serializations. `to_json` is the machine/inspect form; `summary`
    // is a one-line human digest for logs.
    std::string to_json(int indent = 2) const;
    std::string summary() const;
};

// The importer seam. Builds a SipModel from any WeightSource (GGUF or the toy
// .llmw). Read-only: does not load weight data, only the directory + metadata.
// Every future importer (HF, ONNX, PyTorch) produces this same structure.
SipModel import_model(const WeightSource& src);

// ============================================================================
// Sip IR Binary Format Specification (On-Disk Layout)
// ============================================================================

// 'SIPR' in little-endian ASCII.
constexpr uint32_t kSipIRMagic = 0x53495052;

// The file starts with this header. All integers are little-endian.
struct SipIRHeader {
    uint32_t magic;          // Must be kSipIRMagic (0x53495052)
    uint32_t version;        // Format version. Starts at 1.
    uint64_t n_tensors;      // Number of tensors.
    uint64_t metadata_bytes; // Size of the metadata block following the header.
    uint64_t payload_offset; // Absolute file offset where tensor payloads begin.
};

// Immediately following the header is the metadata section.
// It consists of a sequence of key-value pairs encoded sequentially.
// Format:
// [uint64_t key_len] [key_len bytes: key_string]
// [uint32_t value_type] (from GgufType)
// [value_len bytes: value_data] (same encoding as GGUF)

// A simpler 8-byte-aligned struct for the tensor descriptor.
struct SipIRTensorDescriptor {
    // 64-char fixed tensor name (null terminated).
    char name[64];
    
    // Data type (from DType enum).
    uint32_t dtype;
    
    // Padding to ensure 8-byte alignment for shape.
    uint32_t pad;
    
    // 4D shape (row-major).
    int64_t shape[4];
    
    // Absolute offset to the tensor payload.
    uint64_t offset;
    
    // Size of the payload in bytes (allows skipping unknown tensors).
    uint64_t nbytes;
};

} // namespace llm
