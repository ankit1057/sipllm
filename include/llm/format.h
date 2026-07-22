// format.h — Task 3 / Phase 2: a custom binary weight container (.llmw).
//
// On-disk layout (all little-endian):
//
//   magic   : 4 bytes  "LLMW"
//   version : u32      (=1)
//   n_meta  : u32      number of metadata key/value pairs
//   n_tensor: u32      number of tensor descriptors
//   --- metadata section ---
//   repeat n_meta:
//     key   : u32 len + bytes
//     type  : u8   (0=i64, 1=f64, 2=str)
//     value : i64 | f64 | (u32 len + bytes)
//   --- tensor directory ---
//   repeat n_tensor:
//     name  : u32 len + bytes
//     dtype : i32  (DType)
//     ndim  : u32
//     dims  : i64 * ndim
//     offset: u64  absolute file offset of this tensor's raw bytes
//     nbytes: u64
//   --- padding to 64 ---
//   --- raw tensor data, each tensor 64-byte aligned ---
//
// The header + directory are tiny and read fully; tensor *data* is only ever
// touched via WeightSource::read_raw (pread) — the whole point of streaming.
#pragma once

#include "llm/file_backing.h"
#include "llm/weight_source.h"

#include <map>
#include <memory>

namespace llm {

constexpr uint32_t kLLMWMagic   = 0x574D4C4C; // 'L''L''M''W' little-endian
constexpr uint32_t kLLMWVersion = 1;

// ---- reader ---------------------------------------------------------------
class ModelFile : public WeightSource {
public:
    explicit ModelFile(const std::string& path, bool use_mmap = false);

    const std::vector<TensorInfo>& tensors() const override { return tensors_; }
    const TensorInfo* find(const std::string& name) const override;
    void read_raw(const TensorInfo& t, void* dst) const override;
    void read_raw_at(uint64_t offset, void* dst, uint64_t n) const override;
    const uint8_t* mmap_base() const override { return file_->map_base(); }
    uint64_t file_size() const override { return file_->size(); }

    bool has_meta(const std::string& key) const override {
        return meta_.count(key) != 0;
    }
    const MetaValue* meta(const std::string& key) const override {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }

    FileBacking& backing() { return *file_; }

private:
    std::unique_ptr<FileBacking>       file_;
    std::vector<TensorInfo>            tensors_;
    std::map<std::string, int>         index_;
    std::map<std::string, MetaValue>   meta_;
};

// ---- writer ---------------------------------------------------------------
// Accumulates metadata + tensors in memory, then lays them out and writes the
// file with correct 64-aligned offsets. Used by tools/make_toy_model.
class ModelWriter {
public:
    void set_meta(const std::string& key, int64_t v);
    void set_meta(const std::string& key, double v);
    void set_meta(const std::string& key, const std::string& v);

    // Adds a tensor; `data` points to `type_nbytes(dtype, numel)` bytes which
    // are copied into an internal staging buffer.
    void add_tensor(const std::string& name, DType dtype,
                    std::vector<int64_t> shape, const void* data);

    // Convenience for fp32 tensors.
    void add_f32(const std::string& name, std::vector<int64_t> shape,
                 const float* data);

    void write(const std::string& path) const;

private:
    struct Pending {
        TensorInfo         info;
        std::vector<uint8_t> bytes;
    };
    std::vector<Pending>              pending_;
    std::map<std::string, MetaValue>  meta_;
};

} // namespace llm
