// tools/gguf_to_sipir.cpp
#include "llm/gguf.h"
#include "llm/sip_ir_writer.h"
#include "llm/common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

using namespace llm;

size_t get_value_size(std::ifstream& in, GgufType type) {
    size_t start = in.tellg();
    switch (type) {
        case GgufType::UINT8: case GgufType::INT8: case GgufType::BOOL: in.seekg(1, std::ios::cur); break;
        case GgufType::UINT16: case GgufType::INT16: in.seekg(2, std::ios::cur); break;
        case GgufType::UINT32: case GgufType::INT32: case GgufType::FLOAT32: in.seekg(4, std::ios::cur); break;
        case GgufType::UINT64: case GgufType::INT64: case GgufType::FLOAT64: in.seekg(8, std::ios::cur); break;
        case GgufType::STRING: {
            uint64_t len;
            in.read((char*)&len, sizeof(len));
            in.seekg(len, std::ios::cur);
            break;
        }
        case GgufType::ARRAY: {
            uint32_t elem_type;
            uint64_t count;
            in.read((char*)&elem_type, sizeof(elem_type));
            in.read((char*)&count, sizeof(count));
            for (uint64_t i = 0; i < count; ++i) {
                get_value_size(in, (GgufType)elem_type);
            }
            break;
        }
        default: throw Error("Unknown GgufType in get_value_size");
    }
    size_t end = in.tellg();
    return end - start;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    std::string in_path = argv[1];
    std::string out_path = in_path + ".sipr";

    GgufFile src(in_path);

    std::ifstream in(in_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open input GGUF\n";
        return 1;
    }

    uint32_t magic, version;
    uint64_t n_tensors, n_meta;
    in.read((char*)&magic, 4);
    in.read((char*)&version, 4);
    in.read((char*)&n_tensors, 8);
    in.read((char*)&n_meta, 8);

    if (magic != kGGUFMagic) {
        std::cerr << "Input is not a GGUF file\n";
        return 1;
    }

    SipIRWriter writer(out_path);

    std::cout << "Writing metadata (" << n_meta << " pairs)...\n";
    for (uint64_t i = 0; i < n_meta; ++i) {
        uint64_t key_len;
        in.read((char*)&key_len, 8);
        std::string key(key_len, '\0');
        in.read(&key[0], key_len);

        uint32_t type;
        in.read((char*)&type, 4);

        size_t v_start = in.tellg();
        size_t val_size = get_value_size(in, (GgufType)type);
        in.seekg(v_start, std::ios::beg);

        std::vector<char> val_data(val_size);
        in.read(val_data.data(), val_size);

        writer.write_meta_kv(key, (GgufType)type, val_data.data(), val_size);
    }

    std::cout << "Writing " << n_tensors << " tensor descriptors...\n";
    std::vector<TensorInfo> tensors = src.tensors();
    
    // We must figure out the alignment padding for the first tensor.
    // SipIRWriter handles it via finalize_headers. But we need to know the offset to write in desc.offset.
    uint64_t meta_bytes_written = writer.tell() - sizeof(SipIRHeader);
    uint64_t payload_offset = sizeof(SipIRHeader) + meta_bytes_written + tensors.size() * sizeof(SipIRTensorDescriptor);
    
    uint32_t alignment = 32;
    uint64_t rem = payload_offset % alignment;
    if (rem != 0) payload_offset += (alignment - rem);
    
    uint64_t current_payload = payload_offset;

    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorInfo& t = tensors[i];
        SipIRTensorDescriptor desc = {};
        
        size_t nlen = t.name.length();
        if (nlen > 63) nlen = 63;
        std::memcpy(desc.name, t.name.c_str(), nlen);
        desc.name[nlen] = '\0';
        
        desc.dtype = static_cast<uint32_t>(t.dtype);
        desc.pad = 0;

        desc.shape[0] = 1; desc.shape[1] = 1; desc.shape[2] = 1; desc.shape[3] = 1;
        for (size_t d = 0; d < t.shape.size() && d < 4; ++d) {
            desc.shape[d] = t.shape[d];
        }

        desc.offset = current_payload;
        desc.nbytes = t.nbytes;
        
        writer.write_tensor_desc(desc);
        
        current_payload += t.nbytes;
        uint64_t trem = current_payload % alignment;
        if (trem != 0) current_payload += (alignment - trem);
    }

    writer.finalize_headers(tensors.size());
    std::cout << "Streaming tensor payloads...\n";
    
    const size_t CHUNK = 1024 * 1024;
    std::vector<char> buf(CHUNK);

    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorInfo& t = tensors[i];
        
        in.seekg(t.offset, std::ios::beg);
        uint64_t remain = t.nbytes;
        while (remain > 0) {
            size_t take = std::min<uint64_t>(CHUNK, remain);
            in.read(buf.data(), take);
            writer.write_tensor_data(buf.data(), take);
            remain -= take;
        }
        
        uint64_t trem = writer.tell() % alignment;
        if (trem != 0) {
            uint64_t pad = alignment - trem;
            std::vector<char> zeros(pad, 0);
            writer.write_tensor_data(zeros.data(), pad);
        }
    }
    
    std::cout << "Success: " << out_path << "\n";
    return 0;
}
