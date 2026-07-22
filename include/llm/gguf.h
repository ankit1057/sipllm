// gguf.h — Task 8 / Phase 5: a GGUF v2/v3 parser.
//
// GGUF is the container llama.cpp uses. Layout:
//   magic "GGUF" | u32 version | u64 tensor_count | u64 metadata_kv_count
//   metadata_kv_count × { gguf_string key; u32 value_type; value }
//   tensor_count     × { gguf_string name; u32 n_dims; u64 dims[]; u32 type; u64 offset }
//   padding to general.alignment (default 32)
//   tensor data blob (each tensor at data_start + its relative offset)
//
// gguf_string = u64 length + raw bytes. All little-endian.
//
// This class only *parses* — it reads the header/metadata/tensor directory and
// exposes them via WeightSource. Tensor *data* is never loaded here; it is
// streamed later through read_raw (pread), exactly as Phase 6 requires.
#pragma once

#include "llm/file_backing.h"
#include "llm/weight_source.h"

#include <map>
#include <memory>

namespace llm {

constexpr uint32_t kGGUFMagic = 0x46554747;   // 'G''G''U''F'

// GGUF metadata value type codes.
enum class GgufType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9,
    UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

class GgufFile : public WeightSource {
public:
    explicit GgufFile(const std::string& path, bool use_mmap = false);

    const std::vector<TensorInfo>& tensors() const override { return tensors_; }
    const TensorInfo* find(const std::string& name) const override;
    void read_raw(const TensorInfo& t, void* dst) const override;
    void read_raw_at(uint64_t offset, void* dst, uint64_t n) const override;
    const uint8_t* mmap_base() const override { return file_->map_base(); }
    uint64_t file_size() const override { return file_->size(); }

    bool has_meta(const std::string& key) const override { return meta_.count(key) != 0; }
    const MetaValue* meta(const std::string& key) const override {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }

    uint32_t version() const { return version_; }
    uint64_t data_offset() const { return data_offset_; }
    FileBacking& backing() { return *file_; }

private:
    std::unique_ptr<FileBacking>      file_;
    std::vector<TensorInfo>           tensors_;
    std::map<std::string, int>        index_;
    std::map<std::string, MetaValue>  meta_;
    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;
    uint32_t alignment_ = 32;
};

} // namespace llm
