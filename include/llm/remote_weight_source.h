#pragma once

#include "llm/weight_source.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <mutex>

namespace llm {

class RemoteWeightSource : public WeightSource {
public:
    RemoteWeightSource(const std::string& host, int port);
    ~RemoteWeightSource() override;

    const std::vector<TensorInfo>& tensors() const override { return tensors_; }
    const TensorInfo* find(const std::string& name) const override;

    void read_raw(const TensorInfo& t, void* dst) const override;
    void read_raw_at(uint64_t offset, void* dst, uint64_t n) const override;

    uint64_t file_size() const override { return file_size_; }

    bool has_meta(const std::string& key) const override;
    const MetaValue* meta(const std::string& key) const override;

private:
    int sock_ = -1;
    uint64_t file_size_ = 0;
    std::vector<TensorInfo> tensors_;
    mutable std::unordered_map<std::string, bool> has_meta_cache_;
    mutable std::unordered_map<std::string, MetaValue> meta_cache_;
    
    // File-based cache for bounded RSS
    mutable FILE* cache_file_ = nullptr;
    mutable std::unordered_map<uint64_t, bool> cached_blocks_;
    static constexpr uint64_t BLOCK_SIZE = 1024 * 1024; // 1MB blocks

    mutable std::mutex mu_;

    void fetch_tensors();
    void fetch_file_size();
    void fetch_block(uint64_t block_idx) const;
};

} // namespace llm
