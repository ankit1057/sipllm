#pragma once
#include <cstdint>
namespace llm {
namespace remote {
enum CmdType : uint32_t {
    CMD_FILE_SIZE = 1,
    CMD_TENSORS = 2,
    CMD_HAS_META = 3,
    CMD_GET_META = 4,
    CMD_READ_RAW = 5,
};
struct Cmd {
    uint32_t type;
    uint32_t arg_len;
    uint64_t offset;
    uint64_t size;
};
}
}
