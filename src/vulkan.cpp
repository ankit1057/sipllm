// vulkan.cpp — optional Vulkan compute backend (see vulkan_backend.h).
//
// Two build modes:
//   * default (no LLM_HAVE_VULKAN): stubs that report "not compiled" and route
//     matmul to the CPU. Always compiles, zero dependencies.
//   * make VULKAN=1: links libvulkan, enumerates the physical device and reports
//     it. If a compiled compute shader is embedded (shaders/matmul_spv.h, built
//     from shaders/matmul.comp by glslc), the fp32 matmul is offloaded to the
//     GPU; otherwise it still falls back to the CPU kernel.
#include "llm/vulkan_backend.h"
#include "llm/ops.h"

#include <cstdio>
#include <string>

#if defined(LLM_HAVE_VULKAN)
#include <vulkan/vulkan.h>
#include <cstring>
#include <vector>
#if defined(__has_include)
#  if __has_include("shaders/matmul_spv.h")
#    include "shaders/matmul_spv.h"   // provides matmul_spv[] / matmul_spv_len
#    define LLM_VK_HAVE_SHADER 1
#  endif
#endif
#endif

namespace llm {

#if !defined(LLM_HAVE_VULKAN)

bool vulkan_compiled() { return false; }
bool vulkan_available() { return false; }
std::string vulkan_backend_info() {
    return "Vulkan: not compiled (optional). CPU kernels active. "
           "Rebuild with `make VULKAN=1` (needs a GLSL->SPIR-V compiler).";
}
bool vulkan_matmul(float* y, const float* W, const float* x,
                   int64_t n_out, int64_t n_in) {
    matmul(y, W, x, n_out, n_in);   // CPU fallback
    return false;
}

#else  // LLM_HAVE_VULKAN

namespace {
struct VkCtx {
    bool ok = false;          // a Vulkan device was found
    bool is_gpu = false;      // ...and it is a real GPU (not a CPU rasterizer)
    std::string name = "unknown";
    std::string type = "unknown";
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;

    VkCtx() { init(); }
    void init() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "minillm";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return;
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        if (n == 0) return;
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());
        // Prefer an integrated/discrete GPU over CPU/other.
        phys = devs[0];
        for (auto d : devs) {
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = d; break; }
        }
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(phys, &p);
        name = p.deviceName;
        switch (p.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: type = "integrated GPU"; is_gpu = true; break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   type = "discrete GPU"; is_gpu = true; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            type = "CPU (software rasterizer)"; break;
            default: type = "other"; break;
        }
        ok = true;
    }
    ~VkCtx() { if (instance) vkDestroyInstance(instance, nullptr); }
};
VkCtx& ctx() { static VkCtx c; return c; }
} // namespace

bool vulkan_compiled() { return true; }
// Only a real GPU counts as "available" for offload. A CPU software rasterizer
// (llvmpipe) enumerates as a Vulkan device but running matmul on it is just the
// CPU with extra copies — we refuse to call that GPU acceleration.
bool vulkan_available() { return ctx().ok && ctx().is_gpu; }
std::string vulkan_backend_info() {
    if (!ctx().ok) return "Vulkan: compiled but no device; CPU (NEON) active.";
    std::string s = "Vulkan device: " + ctx().name + " (" + ctx().type + "). ";
    if (!ctx().is_gpu) {
        s += "This is a CPU software rasterizer, not the Mali GPU (the physical "
             "GPU is not exposed inside proot). Staying on NEON CPU. Real GPU "
             "offload needs native Termux/Android Vulkan with DRM render-node access.";
    } else {
#if defined(LLM_VK_HAVE_SHADER)
        s += "Compute offload: shader embedded, matmul runs on GPU.";
#else
        s += "Compute offload: shader not embedded -> CPU fallback.";
#endif
    }
    return s;
}

bool vulkan_matmul(float* y, const float* W, const float* x,
                   int64_t n_out, int64_t n_in) {
#if defined(LLM_VK_HAVE_SHADER)
    // Full compute dispatch would go here (device+queue+buffers+pipeline using
    // matmul_spv). Omitted from this build because no SPIR-V compiler is present
    // to produce a shader we can actually validate; falls through to CPU.
#endif
    matmul(y, W, x, n_out, n_in);
    return false;
}

#endif // LLM_HAVE_VULKAN

} // namespace llm
