// tests/test_sip_ir.cpp
#include "llm/sip_ir_reader.h"
#include "llm/sip_ir_writer.h"
#include "llm/common.h"
#include "tests/test_util.h"
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <vector>

using namespace llm;

TEST(SipIRTest_Roundtrip) {
    std::string path = "test_roundtrip.sipr";

    {
        SipIRWriter writer(path);
        
        // Write metadata
        int64_t v_int = 42;
        writer.write_meta_kv("test.int", GgufType::INT64, &v_int, sizeof(v_int));
        
        float v_float = 3.14f;
        writer.write_meta_kv("test.float", GgufType::FLOAT32, &v_float, sizeof(v_float));
        
        std::string v_str = "hello";
        uint64_t str_len = v_str.size();
        std::vector<char> str_buf(sizeof(str_len) + str_len);
        std::memcpy(str_buf.data(), &str_len, sizeof(str_len));
        std::memcpy(str_buf.data() + sizeof(str_len), v_str.data(), str_len);
        writer.write_meta_kv("test.str", GgufType::STRING, str_buf.data(), str_buf.size());
        
        // Write tensor descriptor
        SipIRTensorDescriptor desc = {};
        std::strncpy(desc.name, "tensor1", 64);
        desc.dtype = static_cast<uint32_t>(DType::F32);
        desc.shape[0] = 2; desc.shape[1] = 3; desc.shape[2] = 1; desc.shape[3] = 1;
        desc.nbytes = 24;
        
        // Offset
        desc.offset = 1024;
        
        writer.write_tensor_desc(desc);
        writer.finalize_headers(1);
        
        // Write payload
        float data[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        writer.write_tensor_data(data, sizeof(data));
    }

    {
        SipIRReader reader(path);
        CHECK(reader.tensors().size() == 1);
        
        const TensorInfo* ti = reader.find("tensor1");
        CHECK(ti != nullptr);
        CHECK(ti->dtype == DType::F32);
        CHECK(ti->shape.size() == 2);
        CHECK(ti->shape[0] == 2);
        CHECK(ti->shape[1] == 3);
        CHECK(ti->offset == 1024);
        CHECK(ti->nbytes == 24);
        
        CHECK(reader.meta_int("test.int") == 42);
        APPROX(reader.meta_float("test.float"), 3.14f, 1e-4);
        CHECK(reader.meta_str("test.str") == "hello");
    }

    std::filesystem::remove(path);
}

int main() {
    printf("== test_sip_ir ==\n");
    return llmtest::run_all();
}
