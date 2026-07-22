// vulkan_backend.h — Task 11: optional GPU compute offload.
//
// Compile-time optional (build with `make VULKAN=1`, which requires a
// GLSL->SPIR-V compiler for the shader). When not compiled, or when no usable
// device is found at runtime, everything transparently falls back to the CPU
// kernels — the engine never depends on the GPU for correctness.
//
// The backend enumerates the Vulkan device (the POCO X6 Pro's Mali-G615) and,
// when a compiled compute shader is available, can offload the fp32 matmul.
#pragma once

#include "llm/threadpool.h"

#include <string>

namespace llm {

// True if the binary was built with the Vulkan backend compiled in.
bool vulkan_compiled();

// True if a usable Vulkan device was found at runtime (false if not compiled).
bool vulkan_available();

// Human-readable status: device name + whether offload is active, or the
// reason the CPU fallback is in use.
std::string vulkan_backend_info();

// y = W @ x on the GPU when available; otherwise runs the CPU matmul. Always
// safe to call — returns true if it actually used the GPU.
bool vulkan_matmul(float* y, const float* W, const float* x,
                   int64_t n_out, int64_t n_in);

} // namespace llm
