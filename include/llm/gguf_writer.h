// gguf_writer.h — emit valid GGUF v3 files.
//
// Used by tests and tools to synthesize models: a byte-exact GGUF the parser
// and the streaming engine can consume, so Phase 6 (real GGUF-backed inference)
// is provable without downloading a 5 GB model. Also builds a complete toy
// Llama in GGUF, optionally quantizing the projection weights so the quantized
// streaming path is exercised end to end.
#pragma once

#include "llm/gguf.h"
#include "llm/model.h"
#include "llm/quant.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace llm {

class GgufWriter {
public:
    void u32(const std::string& k, uint32_t v)  { kv(k, GgufType::UINT32, [&]{ pod(v); }); }
    void i32(const std::string& k, int32_t v)   { kv(k, GgufType::INT32, [&]{ pod(v); }); }
    void f32(const std::string& k, float v)     { kv(k, GgufType::FLOAT32, [&]{ pod(v); }); }
    void str(const std::string& k, const std::string& v) {
        kv(k, GgufType::STRING, [&]{ gstr(v); });
    }
    void str_array(const std::string& k, const std::vector<std::string>& v) {
        kv(k, GgufType::ARRAY, [&]{
            pod((uint32_t)GgufType::STRING); pod((uint64_t)v.size());
            for (auto& s : v) gstr(s);
        });
    }
    void i32_array(const std::string& k, const std::vector<int32_t>& v) {
        kv(k, GgufType::ARRAY, [&]{
            pod((uint32_t)GgufType::INT32); pod((uint64_t)v.size());
            for (auto x : v) pod(x);
        });
    }

    // shape is our row-major order [n_out, n_in]; stored ne-order (reversed).
    void add_tensor(const std::string& name, DType dtype,
                    std::vector<int64_t> shape, const void* data, uint64_t nbytes) {
        Tensor t;
        t.name = name; t.dtype = dtype; t.shape = std::move(shape);
        t.bytes.assign((const uint8_t*)data, (const uint8_t*)data + nbytes);
        tensors_.push_back(std::move(t));
    }

    void write(const std::string& path, uint32_t alignment = 32) const {
        std::vector<uint8_t> hdr;
        auto p32 = [&](uint32_t v){ append(hdr, &v, 4); };
        auto p64 = [&](uint64_t v){ append(hdr, &v, 8); };
        p32(kGGUFMagic); p32(3);
        p64(tensors_.size()); p64(n_meta_);
        hdr.insert(hdr.end(), meta_.begin(), meta_.end());

        // tensor directory
        for (auto& t : tensors_) {
            uint64_t nlen = t.name.size();
            append(hdr, &nlen, 8); append(hdr, t.name.data(), nlen);
            uint32_t nd = (uint32_t)t.shape.size(); append(hdr, &nd, 4);
            for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it) {  // ne-order
                uint64_t d = (uint64_t)*it; append(hdr, &d, 8);
            }
            uint32_t ty = (uint32_t)t.dtype; append(hdr, &ty, 4);
            append(hdr, &t.rel_offset, 8);   // patched below
        }

        // compute data section + relative offsets
        uint64_t data_start = round_up(hdr.size(), alignment);
        std::vector<uint64_t> rel(tensors_.size());
        uint64_t cur = 0;
        for (size_t i = 0; i < tensors_.size(); ++i) {
            rel[i] = cur;
            cur = round_up(cur + tensors_[i].bytes.size(), alignment);
        }
        // patch offsets in directory: re-walk to find each offset field
        {
            size_t pos = 24 + meta_.size();  // header(24) + metadata blob
            for (size_t i = 0; i < tensors_.size(); ++i) {
                pos += 8 + tensors_[i].name.size();      // name
                pos += 4;                                 // ndim
                pos += 8 * tensors_[i].shape.size();      // dims
                pos += 4;                                 // type
                std::memcpy(&hdr[pos], &rel[i], 8);       // offset
                pos += 8;
            }
        }

        FILE* fp = std::fopen(path.c_str(), "wb");
        LLM_CHECK(fp, "gguf write: cannot open " + path);
        std::fwrite(hdr.data(), 1, hdr.size(), fp);
        // pad to data_start
        std::vector<uint8_t> pad(data_start - hdr.size(), 0);
        std::fwrite(pad.data(), 1, pad.size(), fp);
        // tensor data with alignment padding
        uint64_t written = 0;
        for (size_t i = 0; i < tensors_.size(); ++i) {
            while (written < rel[i]) { uint8_t z = 0; std::fwrite(&z, 1, 1, fp); ++written; }
            std::fwrite(tensors_[i].bytes.data(), 1, tensors_[i].bytes.size(), fp);
            written += tensors_[i].bytes.size();
        }
        std::fclose(fp);
    }

private:
    struct Tensor {
        std::string name; DType dtype; std::vector<int64_t> shape;
        std::vector<uint8_t> bytes; uint64_t rel_offset = 0;
    };
    template <class T> void pod(T v) { append(meta_, &v, sizeof(T)); }
    void gstr(const std::string& s) {
        uint64_t n = s.size(); append(meta_, &n, 8); append(meta_, s.data(), n);
    }
    template <class F> void kv(const std::string& key, GgufType type, F emit) {
        gstr(key);
        uint32_t t = (uint32_t)type; append(meta_, &t, 4);
        emit();
        ++n_meta_;
    }
    static void append(std::vector<uint8_t>& b, const void* p, size_t n) {
        const uint8_t* c = (const uint8_t*)p; b.insert(b.end(), c, c + n);
    }

    std::vector<uint8_t> meta_;
    uint64_t n_meta_ = 0;
    std::vector<Tensor> tensors_;
};

// Build a complete toy Llama in GGUF. weight_type: F32 (exact) or a quantized
// type with a reference quantizer (Q8_0/Q4_0) applied to the 2-D projections.
inline void write_toy_gguf(const std::string& path, const struct ToyGgufConfig& c);

struct ToyGgufConfig {
    int64_t n_layers = 2, dim = 32, n_heads = 2, n_kv_heads = 2, ffn_dim = 64,
            vocab_size = 48, ctx_len = 128;
    float rope_theta = 10000.f, rms_eps = 1e-5f;
    DType weight_type = DType::F32;   // projections; norms always F32
    uint32_t seed = 4321;
    bool with_tokenizer = false;      // emit a byte-level tokenizer vocab
    std::string arch = "llama";       // general.architecture + hparam key prefix
    bool attn_qkv_bias = false;       // emit q/k/v bias tensors (Qwen2)
    bool zero_bias = false;           // if biased, fill biases with zeros
    bool post_norms = false;          // emit Gemma2 post-attention / post-ffn norms
    float attn_softcap = 0.f;         // Gemma2 attn logit softcap (0 = none)
    float final_softcap = 0.f;        // Gemma2 final logit softcap (0 = none)
    bool qk_norm = false;             // emit Gemma3 per-head q/k norms
    float rope_local = 0.f;           // Gemma3 local (sliding) RoPE base (0 = none)
    int64_t swa_pattern = 0;          // Gemma3 global-layer period (0 = all global)
    bool fused_qkv = false;           // Phi3: one attn_qkv + fused ffn_up (gate;up)
    int64_t rope_dim = 0;             // Phi3 partial rotary dims (0 = full head_dim)
};

inline void write_toy_gguf(const std::string& path, const ToyGgufConfig& c) {
    const int64_t head_dim = c.dim / c.n_heads;
    const int64_t q_dim = c.n_heads * head_dim, kv_dim = c.n_kv_heads * head_dim;
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> nd(0.f, 0.02f);

    GgufWriter w;
    const std::string a = c.arch + ".";   // hparams are namespaced under the arch
    w.str("general.architecture", c.arch);
    w.u32(a + "block_count", (uint32_t)c.n_layers);
    w.u32(a + "attention.head_count", (uint32_t)c.n_heads);
    w.u32(a + "attention.head_count_kv", (uint32_t)c.n_kv_heads);
    w.u32(a + "embedding_length", (uint32_t)c.dim);
    w.u32(a + "feed_forward_length", (uint32_t)c.ffn_dim);
    w.u32(a + "context_length", (uint32_t)c.ctx_len);
    w.f32(a + "rope.freq_base", c.rope_theta);
    w.f32(a + "attention.layer_norm_rms_epsilon", c.rms_eps);
    w.u32("general.alignment", 32);
    if (c.attn_softcap > 0.f)  w.f32(a + "attn_logit_softcapping", c.attn_softcap);
    if (c.final_softcap > 0.f) w.f32(a + "final_logit_softcapping", c.final_softcap);
    if (c.rope_local > 0.f)    w.f32(a + "rope.local_freq_base", c.rope_local);
    if (c.swa_pattern > 0)     w.u32(a + "attention.sliding_window_pattern", (uint32_t)c.swa_pattern);
    if (c.fused_qkv) {
        // Pin head_dim explicitly so rope.dimension_count (partial rotary) is
        // not mistaken for it during config discovery.
        w.u32(a + "attention.key_length", (uint32_t)head_dim);
        if (c.rope_dim > 0) w.u32(a + "rope.dimension_count", (uint32_t)c.rope_dim);
    }
    if (c.with_tokenizer) {
        // Minimal byte-level vocab: one token per byte value 0..vocab-1.
        std::vector<std::string> toks; std::vector<int32_t> types;
        for (int64_t i = 0; i < c.vocab_size; ++i) {
            toks.push_back(std::string("<") + std::to_string(i) + ">");
            types.push_back(1);
        }
        w.str("tokenizer.ggml.model", "llama");
        w.str_array("tokenizer.ggml.tokens", toks);
        w.i32_array("tokenizer.ggml.token_type", types);
        w.u32("tokenizer.ggml.bos_token_id", 1);
        w.u32("tokenizer.ggml.eos_token_id", 2);
    }

    auto emit = [&](const std::string& name, std::vector<int64_t> shape, bool quant) {
        int64_t n = 1; for (auto d : shape) n *= d;
        std::vector<float> f(n);
        for (auto& v : f) v = nd(rng);
        DType dt = quant ? c.weight_type : DType::F32;
        if (dt == DType::F32) {
            w.add_tensor(name, DType::F32, shape, f.data(), n * 4);
        } else {
            std::vector<uint8_t> q(type_nbytes(dt, n));
            if (dt == DType::Q8_0) quantize_q8_0(f.data(), q.data(), n);
            else if (dt == DType::Q4_0) quantize_q4_0(f.data(), q.data(), n);
            else throw Error("write_toy_gguf: no reference quantizer for this type");
            w.add_tensor(name, dt, shape, q.data(), q.size());
        }
    };
    auto ones = [&](const std::string& name, int64_t n) {
        std::vector<float> o(n, 1.f); w.add_tensor(name, DType::F32, {n}, o.data(), n * 4);
    };

    emit(names::token_embd, {c.vocab_size, c.dim}, true);
    for (int64_t i = 0; i < c.n_layers; ++i) {
        ones(names::attn_norm(i), c.dim);
        if (c.fused_qkv) {
            // Fused q/k/v in one tensor. Filled in the same RNG order as the
            // separate q,k,v below, so the values are identical.
            emit(names::blk(i, "attn_qkv.weight"), {q_dim + 2 * kv_dim, c.dim}, true);
        } else {
            emit(names::attn_q(i), {q_dim, c.dim}, true);
            emit(names::attn_k(i), {kv_dim, c.dim}, true);
            emit(names::attn_v(i), {kv_dim, c.dim}, true);
        }
        emit(names::attn_out(i), {c.dim, q_dim}, true);
        if (c.qk_norm) { ones(names::blk(i, "attn_q_norm.weight"), head_dim);
                         ones(names::blk(i, "attn_k_norm.weight"), head_dim); }
        if (c.post_norms) ones(names::blk(i, "post_attention_norm.weight"), c.dim);
        ones(names::ffn_norm(i), c.dim);
        if (c.fused_qkv) {
            // Fused gate+up ([gate; up]) — same RNG order as separate gate,up.
            emit(names::ffn_up(i), {2 * c.ffn_dim, c.dim}, true);
        } else {
            emit(names::ffn_gate(i), {c.ffn_dim, c.dim}, true);
            emit(names::ffn_up(i), {c.ffn_dim, c.dim}, true);
        }
        emit(names::ffn_down(i), {c.dim, c.ffn_dim}, true);
        if (c.post_norms) ones(names::blk(i, "post_ffw_norm.weight"), c.dim);
    }
    ones(names::output_norm, c.dim);
    emit(names::output, {c.vocab_size, c.dim}, true);

    // Optional q/k/v biases (Qwen2). Emitted AFTER all weights so the weight
    // RNG stream is identical to a bias-free model with the same seed — that
    // lets tests isolate the bias effect. Zero biases must reproduce the
    // bias-free forward exactly.
    if (c.attn_qkv_bias) {
        auto vecf = [&](const std::string& name, int64_t n) {
            std::vector<float> f(n);
            for (auto& v : f) v = c.zero_bias ? 0.f : nd(rng);
            w.add_tensor(name, DType::F32, {n}, f.data(), n * 4);
        };
        for (int64_t i = 0; i < c.n_layers; ++i) {
            vecf(names::blk(i, "attn_q.bias"), q_dim);
            vecf(names::blk(i, "attn_k.bias"), kv_dim);
            vecf(names::blk(i, "attn_v.bias"), kv_dim);
        }
    }
    w.write(path);
}

} // namespace llm
