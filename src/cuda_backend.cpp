// cuda_backend.cpp — runtime-loaded CUDA driver API + embedded-PTX matvec.
// See include/llm/cuda_backend.h. No CUDA headers, no link-time dependency:
// every driver entry point is resolved with dlsym from libcuda.so.1.
#include "llm/cuda_backend.h"
#include "llm/common.h"
#include "llm/model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace llm {
namespace cuda {

#include "cuda_ptx.inc"   // static const char kCudaKernelsPTX[]

namespace {

// ---- minimal driver-API ABI (stable since CUDA 4; the _v2 symbols are what
// cuda.h maps the public names to on every 64-bit platform) ------------------
using CUresult    = int;
using CUdevice    = int;
using CUdeviceptr = unsigned long long;
using CUcontext   = void*;
using CUmodule    = void*;
using CUfunction  = void*;
using CUstream    = void*;
constexpr CUresult CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

struct Api {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGetCount)(int*);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetName)(char*, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned, int*, void**);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemAllocHost)(void**, size_t);
    CUresult (*cuMemFreeHost)(void*);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemGetInfo)(size_t*, size_t*);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, CUstream, void**, void**);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuGetErrorString)(CUresult, const char**);
};

enum KType { K_F32 = 0, K_F16, K_Q4K, K_Q5K, K_Q6K, K_COUNT };
const char* const kMvNames[K_COUNT] = {"mv_f32", "mv_f16", "mv_q4k", "mv_q5k", "mv_q6k"};
const char* const kMmNames[K_COUNT] = {"mm_f32", "mm_f16", "mm_q4k", "mm_q5k", "mm_q6k"};

struct State {
    std::once_flag once;
    bool ok = false;
    std::string why;
    std::string name;
    void* lib = nullptr;
    Api api{};
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction mv[K_COUNT]{}, mm[K_COUNT]{};
    // Activation / output scratch: device + pinned host staging, grown on demand.
    std::mutex mu;
    CUdeviceptr dx = 0, dy = 0;
    size_t dx_cap = 0, dy_cap = 0;
    float* hx = nullptr; float* hy = nullptr;
    size_t hx_cap = 0, hy_cap = 0;
};

State& st() { static State s; return s; }

thread_local CUcontext tls_ctx = nullptr;   // context made current on this thread

const char* errstr(CUresult r) {
    const char* s = nullptr;
    if (st().api.cuGetErrorString) st().api.cuGetErrorString(r, &s);
    return s ? s : "unknown CUDA error";
}

void check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS)
        throw Error(std::string("cuda: ") + what + " failed: " + errstr(r));
}

void make_current() {
    State& s = st();
    if (tls_ctx != s.ctx) {
        check(s.api.cuCtxSetCurrent(s.ctx), "cuCtxSetCurrent");
        tls_ctx = s.ctx;
    }
}

int ktype(DType t) {
    switch (t) {
        case DType::F32:  return K_F32;
        case DType::F16:  return K_F16;
        case DType::Q4_K: return K_Q4K;
        case DType::Q5_K: return K_Q5K;
        case DType::Q6_K: return K_Q6K;
        default:          return -1;
    }
}

// Device bytes per 256-element super-block.
size_t dev_block_bytes(DType t) {
    switch (t) {
        case DType::F32:  return 1024;
        case DType::F16:  return 512;
        case DType::Q4_K: return 144;
        case DType::Q5_K: return 176;
        case DType::Q6_K: return 212;   // GGUF 210 + 2 pad (4-byte alignment)
        default:          return 0;
    }
}

template <class F> bool sym(void* lib, F& fn, const char* name) {
#if !defined(_WIN32)
    fn = reinterpret_cast<F>(dlsym(lib, name));
#else
    fn = nullptr;
#endif
    return fn != nullptr;
}

void init() {
    State& s = st();
    const char* off = std::getenv("SIPLLM_NO_CUDA");
    if (off && *off && std::strcmp(off, "0") != 0) { s.why = "disabled by SIPLLM_NO_CUDA"; return; }
#if defined(_WIN32)
    s.why = "CUDA backend not supported on this platform"; return;
#else
    s.lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!s.lib) s.lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!s.lib) { s.why = "libcuda.so.1 not found (no NVIDIA driver)"; return; }
    Api& a = s.api;
    bool okp = sym(s.lib, a.cuInit, "cuInit") &&
               sym(s.lib, a.cuDeviceGetCount, "cuDeviceGetCount") &&
               sym(s.lib, a.cuDeviceGet, "cuDeviceGet") &&
               sym(s.lib, a.cuDeviceGetName, "cuDeviceGetName") &&
               sym(s.lib, a.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
               sym(s.lib, a.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
               sym(s.lib, a.cuCtxSetCurrent, "cuCtxSetCurrent") &&
               sym(s.lib, a.cuModuleLoadDataEx, "cuModuleLoadDataEx") &&
               sym(s.lib, a.cuModuleGetFunction, "cuModuleGetFunction") &&
               sym(s.lib, a.cuMemAlloc, "cuMemAlloc_v2") &&
               sym(s.lib, a.cuMemFree, "cuMemFree_v2") &&
               sym(s.lib, a.cuMemAllocHost, "cuMemAllocHost_v2") &&
               sym(s.lib, a.cuMemFreeHost, "cuMemFreeHost") &&
               sym(s.lib, a.cuMemcpyHtoD, "cuMemcpyHtoD_v2") &&
               sym(s.lib, a.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2") &&
               sym(s.lib, a.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2") &&
               sym(s.lib, a.cuMemGetInfo, "cuMemGetInfo_v2") &&
               sym(s.lib, a.cuLaunchKernel, "cuLaunchKernel") &&
               sym(s.lib, a.cuStreamSynchronize, "cuStreamSynchronize") &&
               sym(s.lib, a.cuGetErrorString, "cuGetErrorString");
    if (!okp) { s.why = "libcuda is missing required driver-API symbols"; return; }

    CUresult r = a.cuInit(0);
    if (r != CUDA_SUCCESS) { s.why = std::string("cuInit: ") + errstr(r); return; }
    int n = 0;
    if (a.cuDeviceGetCount(&n) != CUDA_SUCCESS || n <= 0) { s.why = "no CUDA device"; return; }
    CUdevice dev = 0;
    if ((r = a.cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) { s.why = std::string("cuDeviceGet: ") + errstr(r); return; }
    char nm[256] = {0};
    a.cuDeviceGetName(nm, sizeof(nm) - 1, dev);
    int maj = 0, mnr = 0;
    a.cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    a.cuDeviceGetAttribute(&mnr, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    s.name = std::string(nm) + " (sm_" + std::to_string(maj) + std::to_string(mnr) + ")";
    if (maj * 10 + mnr < 61) { s.why = s.name + ": compute capability < 6.1 unsupported"; return; }

    // The primary context is shareable by every thread (cuCtxSetCurrent per
    // thread in make_current()), unlike a cuCtxCreate'd one.
    if ((r = a.cuDevicePrimaryCtxRetain(&s.ctx, dev)) != CUDA_SUCCESS) {
        s.why = std::string("cuDevicePrimaryCtxRetain: ") + errstr(r); return;
    }
    if ((r = a.cuCtxSetCurrent(s.ctx)) != CUDA_SUCCESS) {
        s.why = std::string("cuCtxSetCurrent: ") + errstr(r); return;
    }
    tls_ctx = s.ctx;

    char elog[8192] = {0}, ilog[4096] = {0};
    int   opts[4] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                     CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
    void* vals[4] = {elog, reinterpret_cast<void*>((uintptr_t)sizeof(elog)),
                     ilog, reinterpret_cast<void*>((uintptr_t)sizeof(ilog))};
    r = a.cuModuleLoadDataEx(&s.mod, kCudaKernelsPTX, 4, opts, vals);
    if (r != CUDA_SUCCESS) {
        s.why = std::string("PTX JIT failed: ") + errstr(r) + (elog[0] ? std::string(": ") + elog : "");
        return;
    }
    for (int k = 0; k < K_COUNT; ++k) {
        if (a.cuModuleGetFunction(&s.mv[k], s.mod, kMvNames[k]) != CUDA_SUCCESS ||
            a.cuModuleGetFunction(&s.mm[k], s.mod, kMmNames[k]) != CUDA_SUCCESS) {
            s.why = std::string("missing kernel ") + kMvNames[k]; return;
        }
    }
    s.ok = true;
#endif
}

void ensure_scratch(size_t xbytes, size_t ybytes) {
    State& s = st();
    Api& a = s.api;
    if (xbytes > s.dx_cap) {
        if (s.dx) a.cuMemFree(s.dx);
        s.dx = 0; s.dx_cap = 0;
        check(a.cuMemAlloc(&s.dx, xbytes), "cuMemAlloc(x)");
        s.dx_cap = xbytes;
    }
    if (ybytes > s.dy_cap) {
        if (s.dy) a.cuMemFree(s.dy);
        s.dy = 0; s.dy_cap = 0;
        check(a.cuMemAlloc(&s.dy, ybytes), "cuMemAlloc(y)");
        s.dy_cap = ybytes;
    }
    if (xbytes > s.hx_cap) {
        if (s.hx) a.cuMemFreeHost(s.hx);
        s.hx = nullptr; s.hx_cap = 0;
        void* p = nullptr;
        check(a.cuMemAllocHost(&p, xbytes), "cuMemAllocHost(x)");
        s.hx = static_cast<float*>(p); s.hx_cap = xbytes;
    }
    if (ybytes > s.hy_cap) {
        if (s.hy) a.cuMemFreeHost(s.hy);
        s.hy = nullptr; s.hy_cap = 0;
        void* p = nullptr;
        check(a.cuMemAllocHost(&p, ybytes), "cuMemAllocHost(y)");
        s.hy = static_cast<float*>(p); s.hy_cap = ybytes;
    }
}

} // namespace

bool available() {
    State& s = st();
    std::call_once(s.once, init);
    return s.ok;
}

const std::string& unavailable_reason() {
    available();
    return st().why;
}

std::string device_name() { return available() ? st().name : std::string(); }

bool mem_info(size_t* free_bytes, size_t* total_bytes) {
    if (!available()) return false;
    std::lock_guard<std::mutex> lk(st().mu);
    make_current();
    size_t f = 0, t = 0;
    if (st().api.cuMemGetInfo(&f, &t) != CUDA_SUCCESS) return false;
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return true;
}

bool supports(DType t, int64_t n_in) {
    return ktype(t) >= 0 && n_in > 0 && (n_in % 256) == 0;
}

size_t weight_bytes(DType t, int64_t n_out, int64_t n_in) {
    return (size_t)n_out * (size_t)(n_in / 256) * dev_block_bytes(t);
}

uint64_t upload_weight(const void* host, DType t, int64_t n_out, int64_t n_in) {
    if (!available() || !supports(t, n_in) || n_out <= 0) return 0;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    make_current();
    const size_t bytes = weight_bytes(t, n_out, n_in);
    CUdeviceptr d = 0;
    if (s.api.cuMemAlloc(&d, bytes) != CUDA_SUCCESS) return 0;
    CUresult r;
    if (t == DType::Q6_K) {
        // Re-pack 210-byte GGUF blocks into 212-byte (4-byte aligned) blocks.
        const size_t nblk = (size_t)n_out * (size_t)(n_in / 256);
        std::vector<uint8_t> tmp(bytes, 0);
        const uint8_t* src = static_cast<const uint8_t*>(host);
        for (size_t b = 0; b < nblk; ++b) std::memcpy(&tmp[b * 212], src + b * 210, 210);
        r = s.api.cuMemcpyHtoD(d, tmp.data(), bytes);
    } else {
        r = s.api.cuMemcpyHtoD(d, host, bytes);
    }
    if (r != CUDA_SUCCESS) { s.api.cuMemFree(d); return 0; }
    return (uint64_t)d;
}

void free_weight(uint64_t dev) {
    if (!dev || !available()) return;
    std::lock_guard<std::mutex> lk(st().mu);
    make_current();
    st().api.cuMemFree((CUdeviceptr)dev);
}

void linear(float* Y, const WeightRef& W, const float* X, int64_t m) {
    LLM_CHECK(W.dev != 0, "cuda::linear: weight is not on the device");
    LLM_CHECK(available(), "cuda::linear: CUDA unavailable");
    const int k = ktype(W.dtype);
    LLM_CHECK(k >= 0 && supports(W.dtype, W.n_in), "cuda::linear: unsupported weight");
    State& s = st();
    Api& a = s.api;
    std::lock_guard<std::mutex> lk(s.mu);
    make_current();

    // Chunk long prompts so scratch stays small (64 x 15360 x 4 B ~ 4 MB).
    const int64_t CH = 64;
    const int64_t mc = m < CH ? m : CH;
    ensure_scratch((size_t)mc * W.n_in * sizeof(float), (size_t)mc * W.n_out * sizeof(float));

    CUdeviceptr dw = (CUdeviceptr)W.dev;
    int n_out = (int)W.n_out, nb = (int)(W.n_in / 256);
    for (int64_t t0 = 0; t0 < m; t0 += CH) {
        const int64_t mm = (m - t0) < CH ? (m - t0) : CH;
        const size_t xb = (size_t)mm * W.n_in * sizeof(float);
        const size_t yb = (size_t)mm * W.n_out * sizeof(float);
        std::memcpy(s.hx, X + t0 * W.n_in, xb);
        check(a.cuMemcpyHtoDAsync(s.dx, s.hx, xb, nullptr), "cuMemcpyHtoDAsync");
        if (mm == 1) {
            void* args[] = {&dw, &s.dx, &s.dy, &n_out, &nb};
            check(a.cuLaunchKernel(s.mv[k], (unsigned)((n_out + 3) / 4), 1, 1, 128, 1, 1,
                                   0, nullptr, args, nullptr), "cuLaunchKernel(mv)");
        } else {
            int mi = (int)mm;
            void* args[] = {&dw, &s.dx, &s.dy, &n_out, &nb, &mi};
            // 8 warps x RPW rows per CTA (RPW = 4, Q6_K 2); 16 tokens per CTA
            // — must mirror RPWOf / TB in kernels/cuda_kernels.cu.
            const int rows_per_cta = 8 * (k == K_Q6K ? 2 : 4);
            check(a.cuLaunchKernel(s.mm[k], (unsigned)((n_out + rows_per_cta - 1) / rows_per_cta),
                                   (unsigned)((mi + 15) / 16), 1, 256, 1, 1,
                                   0, nullptr, args, nullptr), "cuLaunchKernel(mm)");
        }
        check(a.cuMemcpyDtoHAsync(s.hy, s.dy, yb, nullptr), "cuMemcpyDtoHAsync");
        check(a.cuStreamSynchronize(nullptr), "cuStreamSynchronize");
        std::memcpy(Y + t0 * W.n_out, s.hy, yb);
    }
}

} // namespace cuda
} // namespace llm
