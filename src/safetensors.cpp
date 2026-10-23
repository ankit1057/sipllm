// safetensors.cpp — HF safetensors + config.json WeightSource (see safetensors.h).
#include "llm/safetensors.h"
#include "llm/common.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

namespace llm {

// ============================================================================
// Minimal zero-dependency JSON parser (sufficient for safetensors headers and
// HF config.json: objects, arrays, strings w/ escapes, numbers, bool, null).
// ============================================================================
namespace {

struct JsonValue {
    enum class T { Null, Bool, Num, Str, Arr, Obj } type = T::Null;
    bool               b = false;
    double             num = 0;
    std::string        str;
    std::vector<JsonValue>                       arr;
    std::vector<std::pair<std::string, JsonValue>> obj;   // preserves order

    const JsonValue* find(const std::string& k) const {
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    bool   is_num() const { return type == T::Num; }
    bool   is_str() const { return type == T::Str; }
    double as_num(double d = 0) const { return type == T::Num ? num : d; }
};

class JsonParser {
public:
    explicit JsonParser(const char* p, size_t n) : p_(p), n_(n) {}
    JsonValue parse() { ws(); JsonValue v = value(); return v; }

private:
    const char* p_; size_t n_, i_ = 0;

    [[noreturn]] void fail(const char* m) { throw Error(std::string("json: ") + m); }
    void ws() { while (i_ < n_ && std::isspace((unsigned char)p_[i_])) ++i_; }
    char peek() { return i_ < n_ ? p_[i_] : '\0'; }
    char get()  { return i_ < n_ ? p_[i_++] : '\0'; }
    bool eat(char c) { if (peek() == c) { ++i_; return true; } return false; }

    JsonValue value() {
        ws();
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { JsonValue v; v.type = JsonValue::T::Str; v.str = string(); return v; }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') { expect("null"); return JsonValue{}; }
        return number();
    }
    void expect(const char* lit) {
        for (const char* q = lit; *q; ++q) if (get() != *q) fail("bad literal");
    }
    JsonValue boolean() {
        JsonValue v; v.type = JsonValue::T::Bool;
        if (peek() == 't') { expect("true");  v.b = true; }
        else               { expect("false"); v.b = false; }
        return v;
    }
    JsonValue number() {
        size_t s = i_;
        if (peek() == '-' || peek() == '+') ++i_;
        while (i_ < n_ && (std::isdigit((unsigned char)p_[i_]) || p_[i_] == '.' ||
               p_[i_] == 'e' || p_[i_] == 'E' || p_[i_] == '+' || p_[i_] == '-')) ++i_;
        if (i_ == s) fail("expected number");
        JsonValue v; v.type = JsonValue::T::Num;
        v.num = std::strtod(std::string(p_ + s, i_ - s).c_str(), nullptr);
        return v;
    }
    std::string string() {
        if (!eat('"')) fail("expected string");
        std::string out;
        while (i_ < n_) {
            char c = get();
            if (c == '"') return out;
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break;
                    case 'r': out += '\r'; break; case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break; case '/': out += '/'; break;
                    case '"': out += '"'; break; case '\\': out += '\\'; break;
                    case 'u': {   // \uXXXX -> UTF-8 (BMP only; enough for configs)
                        if (i_ + 4 > n_) fail("bad \\u");
                        int cp = (int)std::strtol(std::string(p_ + i_, 4).c_str(), nullptr, 16);
                        i_ += 4;
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                        else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                        break;
                    }
                    default: out += e;
                }
            } else out += c;
        }
        fail("unterminated string");
    }
    JsonValue array() {
        JsonValue v; v.type = JsonValue::T::Arr;
        eat('['); ws();
        if (eat(']')) return v;
        for (;;) {
            v.arr.push_back(value()); ws();
            if (eat(',')) { ws(); continue; }
            if (eat(']')) break;
            fail("expected , or ] in array");
        }
        return v;
    }
    JsonValue object() {
        JsonValue v; v.type = JsonValue::T::Obj;
        eat('{'); ws();
        if (eat('}')) return v;
        for (;;) {
            ws(); std::string key = string(); ws();
            if (!eat(':')) fail("expected : in object");
            v.obj.emplace_back(key, value()); ws();
            if (eat(',')) { ws(); continue; }
            if (eat('}')) break;
            fail("expected , or } in object");
        }
        return v;
    }
};

DType dtype_from_st(const std::string& s) {
    if (s == "F32")  return DType::F32;
    if (s == "F16")  return DType::F16;
    if (s == "BF16") return DType::BF16;
    // I8/U8/I32/F64/... are not weight dtypes we execute; surface a clear error.
    throw Error("safetensors: unsupported tensor dtype '" + s + "'");
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("safetensors: cannot open " + path);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

} // namespace

// ============================================================================
// HF tensor-name -> GGUF name mapping
// ============================================================================
std::string hf_to_gguf_name(const std::string& n) {
    // Global (non-layer) tensors.
    if (n == "model.embed_tokens.weight") return "token_embd.weight";
    if (n == "model.norm.weight")         return "output_norm.weight";
    if (n == "lm_head.weight")            return "output.weight";

    // Per-layer: "model.layers.<i>.<rest>"
    const std::string pfx = "model.layers.";
    if (n.rfind(pfx, 0) == 0) {
        size_t dot = n.find('.', pfx.size());
        if (dot != std::string::npos) {
            std::string idx  = n.substr(pfx.size(), dot - pfx.size());
            std::string rest = n.substr(dot + 1);
            auto blk = [&](const char* suf) { return "blk." + idx + "." + suf; };
            // attention
            if (rest == "self_attn.q_proj.weight") return blk("attn_q.weight");
            if (rest == "self_attn.k_proj.weight") return blk("attn_k.weight");
            if (rest == "self_attn.v_proj.weight") return blk("attn_v.weight");
            if (rest == "self_attn.o_proj.weight") return blk("attn_output.weight");
            if (rest == "self_attn.q_proj.bias")   return blk("attn_q.bias");
            if (rest == "self_attn.k_proj.bias")   return blk("attn_k.bias");
            if (rest == "self_attn.v_proj.bias")   return blk("attn_v.bias");
            if (rest == "self_attn.q_norm.weight") return blk("attn_q_norm.weight");
            if (rest == "self_attn.k_norm.weight") return blk("attn_k_norm.weight");
            // norms
            if (rest == "input_layernorm.weight")          return blk("attn_norm.weight");
            if (rest == "post_attention_layernorm.weight") return blk("ffn_norm.weight");
            if (rest == "pre_feedforward_layernorm.weight")  return blk("ffn_norm.weight");   // Gemma2
            if (rest == "post_feedforward_layernorm.weight") return blk("post_ffw_norm.weight");
            // mlp
            if (rest == "mlp.gate_proj.weight") return blk("ffn_gate.weight");
            if (rest == "mlp.up_proj.weight")   return blk("ffn_up.weight");
            if (rest == "mlp.down_proj.weight") return blk("ffn_down.weight");
        }
    }
    return n;   // unknown -> pass through (vision towers, aux tensors)
}

// ============================================================================
// SafetensorsSource
// ============================================================================
SafetensorsSource::SafetensorsSource(const std::string& st_path,
                                     const std::string& cfg_path) {
    file_ = std::make_unique<FileBacking>(st_path, false);
    parse_header();

    std::string cfg = cfg_path;
    if (cfg.empty()) {
        // Look for a sibling config.json next to the weights.
        size_t slash = st_path.find_last_of('/');
        std::string dir = (slash == std::string::npos) ? "" : st_path.substr(0, slash + 1);
        std::string sib = dir + "config.json";
        std::ifstream probe(sib);
        if (probe.good()) cfg = sib;
    }
    if (!cfg.empty()) load_config(cfg);
}

void SafetensorsSource::parse_header() {
    // Layout: u64 LE header_len | header_len bytes JSON | tensor data blob.
    uint64_t header_len = 0;
    file_->pread_exact(0, &header_len, 8);
    LLM_CHECK(header_len > 0 && header_len + 8 <= file_->size(),
              "safetensors: bad header length");
    std::vector<char> hdr(header_len);
    file_->pread_exact(8, hdr.data(), header_len);
    data_start_ = 8 + header_len;

    JsonValue root = JsonParser(hdr.data(), hdr.size()).parse();
    LLM_CHECK(root.type == JsonValue::T::Obj, "safetensors: header is not a JSON object");

    for (auto& kv : root.obj) {
        const std::string& hf_name = kv.first;
        if (hf_name == "__metadata__") continue;
        const JsonValue& e = kv.second;
        const JsonValue* jd = e.find("dtype");
        const JsonValue* js = e.find("shape");
        const JsonValue* jo = e.find("data_offsets");
        if (!jd || !js || !jo) continue;

        TensorInfo ti;
        ti.name  = hf_to_gguf_name(hf_name);
        ti.dtype = dtype_from_st(jd->str);
        for (const JsonValue& d : js->arr) ti.shape.push_back((int64_t)d.as_num());
        // safetensors stores 1-D as [n]; a scalar as []. Our engine treats norm
        // vectors as 1-D [n], which matches.
        uint64_t begin = (uint64_t)jo->arr.at(0).as_num();
        uint64_t end   = (uint64_t)jo->arr.at(1).as_num();
        ti.offset = data_start_ + begin;
        ti.nbytes = end - begin;

        index_[ti.name] = (int)tensors_.size();
        tensors_.push_back(std::move(ti));
    }
    LLM_CHECK(!tensors_.empty(), "safetensors: no tensors in header");
}

void SafetensorsSource::load_config(const std::string& path) {
    std::string blob = read_file(path);
    JsonValue c = JsonParser(blob.data(), blob.size()).parse();
    LLM_CHECK(c.type == JsonValue::T::Obj, "config.json is not a JSON object");

    auto set_int = [&](const std::string& key, int64_t v) {
        MetaValue m; m.kind = MetaValue::Kind::Int; m.i = v; meta_[key] = m;
    };
    auto set_flt = [&](const std::string& key, double v) {
        MetaValue m; m.kind = MetaValue::Kind::Float; m.f = v; meta_[key] = m;
    };
    auto set_str = [&](const std::string& key, const std::string& v) {
        MetaValue m; m.kind = MetaValue::Kind::Str; m.s = v; meta_[key] = m;
    };

    // model_type -> architecture (GGUF-style). Default "llama".
    arch_ = "llama";
    if (const JsonValue* mt = c.find("model_type"); mt && mt->is_str()) arch_ = mt->str;
    set_str("general.architecture", arch_);
    // Norm weights are as HF stores them (Gemma's +1 is not baked in as in GGUF).
    set_int("sipllm.raw_hf_norm_weights", 1);
    const std::string A = arch_ + ".";
    auto ci = [&](const char* k, const std::string& gguf) {
        if (const JsonValue* v = c.find(k); v && v->is_num()) set_int(gguf, (int64_t)v->as_num());
    };
    auto cf = [&](const char* k, const std::string& gguf) {
        if (const JsonValue* v = c.find(k); v && v->is_num()) set_flt(gguf, v->as_num());
    };

    ci("num_hidden_layers",      A + "block_count");
    ci("num_attention_heads",    A + "attention.head_count");
    ci("num_key_value_heads",    A + "attention.head_count_kv");
    ci("hidden_size",            A + "embedding_length");
    ci("intermediate_size",      A + "feed_forward_length");
    ci("max_position_embeddings",A + "context_length");
    ci("vocab_size",             A + "vocab_size");
    ci("head_dim",               A + "attention.key_length");
    cf("rope_theta",             A + "rope.freq_base");
    cf("rms_norm_eps",           A + "attention.layer_norm_rms_epsilon");
    cf("layer_norm_eps",         A + "attention.layer_norm_epsilon");
    cf("layer_norm_epsilon",     A + "attention.layer_norm_epsilon");
    // Gemma soft-caps + experts (best-effort; inert when absent).
    cf("attn_logit_softcapping",  A + "attn_logit_softcapping");
    cf("final_logit_softcapping", A + "final_logit_softcapping");
    cf("query_pre_attn_scalar",   A + "attention.query_pre_attn_scalar");
    ci("num_local_experts",       A + "expert_count");
    ci("num_experts_per_tok",     A + "expert_used_count");

    // rope_scaling: {"rope_type"|"type":"llama3", factor, low/high_freq_factor,
    // original_max_position_embeddings}
    if (const JsonValue* rs = c.find("rope_scaling"); rs && rs->type == JsonValue::T::Obj) {
        const JsonValue* rt = rs->find("rope_type"); if (!rt) rt = rs->find("type");
        if (rt && rt->is_str()) set_str(A + "rope.scaling.type", rt->str);
        auto rsf = [&](const char* k, const std::string& gguf) {
            if (const JsonValue* v = rs->find(k); v && v->is_num()) set_flt(gguf, v->as_num());
        };
        rsf("factor",           A + "rope.scaling.factor");
        rsf("low_freq_factor",  A + "rope.scaling.low_freq_factor");
        rsf("high_freq_factor", A + "rope.scaling.high_freq_factor");
        if (const JsonValue* v = rs->find("original_max_position_embeddings"); v && v->is_num())
            set_int(A + "rope.scaling.original_context_length", (int64_t)v->as_num());
    }

    // tokenizer hint: HF stores the tokenizer separately; expose a sensible
    // default so downstream picks BPE for Llama-3/Qwen and SPM for Llama-2.
    if (const JsonValue* tc = c.find("tokenizer_class"); tc && tc->is_str()) {
        std::string t = tc->str;
        std::string kind = (t.find("Llama") != std::string::npos && t.find("Fast") == std::string::npos)
                         ? "llama" : "gpt2";
        set_str("tokenizer.ggml.model", kind);
    }
    if (const JsonValue* v = c.find("bos_token_id"); v && v->is_num()) set_int("tokenizer.ggml.bos_token_id", (int64_t)v->as_num());
    if (const JsonValue* v = c.find("eos_token_id"); v && v->is_num()) set_int("tokenizer.ggml.eos_token_id", (int64_t)v->as_num());
}

const TensorInfo* SafetensorsSource::find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

void SafetensorsSource::read_raw(const TensorInfo& t, void* dst) const {
    file_->pread_exact(t.offset, dst, t.nbytes);
}
void SafetensorsSource::read_raw_at(uint64_t offset, void* dst, uint64_t n) const {
    file_->pread_exact(offset, dst, n);
}

} // namespace llm
