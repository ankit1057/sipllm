// sip_ir_reader.h
#pragma once

#include "llm/sip_ir.h"
#include "llm/weight_source.h"
#include "llm/file_backing.h"
#include "llm/gguf.h" // For GgufType in parsing
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llm {

class SipIRReader : public WeightSource {
public:
    explicit SipIRReader(const std::string& path, bool use_mmap = false);

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

private:
    std::unique_ptr<FileBacking> file_;
    std::vector<TensorInfo> tensors_;
    std::map<std::string, int> index_;
    std::map<std::string, MetaValue> meta_;
};

} // namespace llm
