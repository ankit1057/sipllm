// sip_ir_writer.h
#pragma once

#include "llm/sip_ir.h"
#include "llm/gguf.h"
#include <string>
#include <vector>
#include <fstream>

namespace llm {

class SipIRWriter {
public:
    explicit SipIRWriter(const std::string& path);
    ~SipIRWriter();

    // Write a typed metadata key-value pair.
    void write_meta_kv(const std::string& key, GgufType type, const void* val_data, size_t val_size);

    // Add a tensor to the directory.
    void write_tensor_desc(const SipIRTensorDescriptor& desc);

    // Finalize the header and directory section.
    // Must be called exactly once before write_tensor_data.
    void finalize_headers(uint64_t n_tensors);

    // Write a block of tensor payload data.
    void write_tensor_data(const void* data, size_t size);

    uint64_t tell();

private:
    std::ofstream out_;
    SipIRHeader header_;
    uint64_t meta_bytes_written_ = 0;
    bool finalized_ = false;
};

} // namespace llm
