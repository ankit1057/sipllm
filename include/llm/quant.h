// quant.h — Task 9: GGUF quantization formats.
//
// Real Q4_K_M models are a mix of tensor types: most weights are Q4_K, a few
// (output, some attention) are Q6_K, norms are F32, embeddings often Q6_K/F16.
// To run one we must dequantize each block back to fp32 during inference.
//
// This module implements dequantization for the whole common GGUF family plus
// reference *quantizers* for Q8_0/Q4_0 (used to synthesize test fixtures and,
// later, to fabricate a tiny valid GGUF for the parser tests). The block
// structs and formulas mirror ggml-quants.c byte-for-byte so offsets computed
// from dtype.cpp line up with real files.
#pragma once

#include "llm/dtype.h"
#include "llm/threadpool.h"

#include <cstdint>

namespace llm {

// ---- half precision -------------------------------------------------------
float    fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float    bf16_to_fp32(uint16_t h);

// ---- dequantization -------------------------------------------------------
// Expand `n` logical elements of type `t` from `src` into `dst` (fp32).
// `n` must be a multiple of the type's block size for quantized types.
void dequantize_row(DType t, const void* src, float* dst, int64_t n);

// ---- fused quantized matmul ----------------------------------------------
// y = W @ x where W is [n_out, n_in] stored as type `t` (row-major, each row a
// whole number of blocks). Each row is dequantized into a small scratch buffer
// and dotted with x — this keeps only one row's worth of fp32 live at a time,
// which is why a whole layer can stay quantized in RAM.
void matmul_quant(float* y, const void* W, DType t, const float* x,
                  int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// ---- reference quantizers (for tests / fixtures) --------------------------
// dst must hold type_nbytes(Q*, n) bytes; n a multiple of 32.
void quantize_q8_0(const float* src, void* dst, int64_t n);
void quantize_q4_0(const float* src, void* dst, int64_t n);

} // namespace llm
