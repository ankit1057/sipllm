// weight_source.h — the abstraction the loader streams weights through.
//
// Phase 2/3 insight: the transformer must not care whether a weight lives in a
// hand-rolled .llmw file or a real GGUF. Both expose the same three things —
// a tensor directory, typed metadata (hyperparameters), and a way to read one
// tensor's raw bytes from disk on demand. LayerLoader talks only to this
// interface, so swapping the toy loader for GGUF (Phase 6) touches nothing in
// the transformer.
#pragma once

#include "llm/dtype.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// Directory entry for one tensor: where it is and how to interpret it.
struct TensorInfo {
    std::string          name;
    DType                dtype = DType::F32;
    std::vector<int64_t> shape;    // row-major; shape[0] is the outer dim
    uint64_t             offset = 0;   // absolute file offset of raw bytes
    uint64_t             nbytes = 0;   // bytes on disk (post-quant)

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    // Row length in logical elements (product of all dims except the first).
    int64_t row_elems() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (size_t i = 1; i < shape.size(); ++i) n *= shape[i];
        return n;
    }
};

// A tagged metadata value (hyperparameters, names, tokenizer bits).
struct MetaValue {
    enum class Kind { Int, Float, Str, IntArr, FloatArr, StrArr } kind = Kind::Int;
    int64_t                  i = 0;
    double                   f = 0;
    std::string              s;
    std::vector<int64_t>     ia;
    std::vector<double>      fa;
    std::vector<std::string> sa;
};

// Abstract weight store. Implemented by ModelFile (.llmw) and GgufFile (.gguf).
class WeightSource {
public:
    virtual ~WeightSource() = default;

    virtual const std::vector<TensorInfo>& tensors() const = 0;
    virtual const TensorInfo* find(const std::string& name) const = 0;

    // Read the tensor's raw on-disk bytes into `dst` (capacity >= t.nbytes).
    // Positional read (pread) — thread-safe, never loads the whole file.
    virtual void read_raw(const TensorInfo& t, void* dst) const = 0;

    // Read `n` bytes at an absolute file offset. Used to stream a single row
    // (e.g. one token's embedding) without materializing a whole tensor.
    virtual void read_raw_at(uint64_t offset, void* dst, uint64_t n) const = 0;

    // Optional zero-copy base pointer if the file is mmap'd; nullptr otherwise.
    // Callers add TensorInfo::offset to reach a tensor's bytes.
    virtual const uint8_t* mmap_base() const { return nullptr; }

    virtual uint64_t file_size() const = 0;

    // ---- typed metadata --------------------------------------------------
    virtual bool has_meta(const std::string& key) const = 0;
    virtual const MetaValue* meta(const std::string& key) const = 0;

    int64_t meta_int(const std::string& key, int64_t def = 0) const {
        const MetaValue* m = meta(key);
        if (!m) return def;
        return m->kind == MetaValue::Kind::Float ? (int64_t)m->f : m->i;
    }
    double meta_float(const std::string& key, double def = 0) const {
        const MetaValue* m = meta(key);
        if (!m) return def;
        return m->kind == MetaValue::Kind::Int ? (double)m->i : m->f;
    }
    std::string meta_str(const std::string& key, const std::string& def = "") const {
        const MetaValue* m = meta(key);
        return m ? m->s : def;
    }
};

} // namespace llm
