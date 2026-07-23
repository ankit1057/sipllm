// dtype.h — tensor element types, shared by the custom format, the GGUF parser
// and the quantizer.
//
// The numeric values deliberately mirror ggml's `enum ggml_type` so the GGUF
// parser can store the on-disk type code directly and the quant dispatcher can
// switch on the same enum. Only the subset we actually handle is named; the
// rest pass through as their integer code.
#pragma once

#include <cstdint>
#include <cstddef>

namespace llm {

enum class DType : int32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,   // the "K" in Q4_K_M
    Q5_K  = 13,
    Q6_K  = 14,   // Q4_K_M mixes Q4_K and Q6_K tensors
    Q8_K  = 15,
    IQ4_NL = 20,  // non-linear 4-bit: fp16 scale + 16-entry lookup, 32/block
    BF16  = 30,
    COUNT = 39,
};

// Per-type block geometry: a "block" of `block_size` logical elements is stored
// in `type_size` bytes. For unquantized types block_size==1.
struct TypeTraits {
    const char* name;
    int64_t     block_size;   // logical elements per block
    int64_t     type_size;    // bytes per block on disk
    bool        quantized;
};

const TypeTraits& type_traits(DType t);

// Bytes required to store `n_elements` of type `t` (must be a multiple of the
// block size for quantized types).
inline int64_t type_nbytes(DType t, int64_t n_elements) {
    const TypeTraits& tr = type_traits(t);
    return n_elements / tr.block_size * tr.type_size;
}

inline const char* dtype_name(DType t) { return type_traits(t).name; }

} // namespace llm
