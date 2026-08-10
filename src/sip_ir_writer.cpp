// sip_ir_writer.cpp
#include "llm/sip_ir_writer.h"
#include "llm/common.h"
#include <cstring>
#include <iostream>

namespace llm {

SipIRWriter::SipIRWriter(const std::string& path) {
    out_.open(path, std::ios::binary);
    LLM_CHECK(out_.is_open(), "SipIRWriter: failed to open " + path);
    
    // Write placeholder header
    header_.magic = kSipIRMagic;
    header_.version = kSipIRVersion;
    header_.n_tensors = 0;
    header_.metadata_bytes = 0;
    header_.payload_offset = 0;
    out_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
}

SipIRWriter::~SipIRWriter() {
    if (out_.is_open()) {
        out_.close();
    }
}

void SipIRWriter::write_meta_kv(const std::string& key, GgufType type, const void* val_data, size_t val_size) {
    LLM_CHECK(!finalized_, "SipIRWriter: cannot write metadata after finalize_headers");
    uint64_t key_len = key.size();
    out_.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
    out_.write(key.data(), key_len);
    
    uint32_t val_type = static_cast<uint32_t>(type);
    out_.write(reinterpret_cast<const char*>(&val_type), sizeof(val_type));
    out_.write(reinterpret_cast<const char*>(val_data), val_size);
    
    meta_bytes_written_ += sizeof(key_len) + key_len + sizeof(val_type) + val_size;
}

void SipIRWriter::write_tensor_desc(const SipIRTensorDescriptor& desc) {
    LLM_CHECK(!finalized_, "SipIRWriter: cannot write tensor descriptors after finalize_headers");
    out_.write(reinterpret_cast<const char*>(&desc), sizeof(desc));
}

void SipIRWriter::finalize_headers(uint64_t n_tensors) {
    LLM_CHECK(!finalized_, "SipIRWriter: already finalized");
    
    header_.n_tensors = n_tensors;
    header_.metadata_bytes = meta_bytes_written_;
    
    // Calculate payload offset: header + metadata + descriptors
    uint64_t offset = sizeof(SipIRHeader) + meta_bytes_written_ + n_tensors * sizeof(SipIRTensorDescriptor);
    
    // Align payload offset to 32 bytes (general.alignment default in GGUF)
    uint32_t alignment = 32;
    uint64_t rem = offset % alignment;
    uint64_t padding = rem == 0 ? 0 : alignment - rem;
    
    header_.payload_offset = offset + padding;
    
    // Write padding
    if (padding > 0) {
        std::vector<char> pad(padding, 0);
        out_.write(pad.data(), padding);
    }
    
    // Seek back and rewrite header
    out_.seekp(0, std::ios::beg);
    out_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
    
    // Seek to end of headers/padding
    out_.seekp(header_.payload_offset, std::ios::beg);
    finalized_ = true;
}

void SipIRWriter::write_tensor_data(const void* data, size_t size) {
    LLM_CHECK(finalized_, "SipIRWriter: must finalize_headers before writing tensor data");
    out_.write(reinterpret_cast<const char*>(data), size);
}

uint64_t SipIRWriter::tell() {
    return static_cast<uint64_t>(out_.tellp());
}

} // namespace llm
