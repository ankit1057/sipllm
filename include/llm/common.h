// common.h — shared primitives: logging, timing, aligned allocation, errors.
//
// Kept header-only and dependency-free so every translation unit can pull it in
// cheaply. The whole engine targets C++17 and nothing beyond the standard
// library + pthreads (+ optional Vulkan behind a compile flag).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <stdexcept>

// Platform RSS query: mach task_info on Apple, /proc on Linux/Android.
#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

namespace llm {

// ---- errors ---------------------------------------------------------------
// A single exception type. Callers that stream weights off disk hit plenty of
// recoverable failure modes (short reads, bad magic, truncated tensors); we
// surface them with context rather than aborting.
struct Error : std::runtime_error {
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

// LLM_CHECK: like assert, but always on and throws with file:line context.
#define LLM_CHECK(cond, msg)                                                   \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw ::llm::Error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) + ": " + (msg));       \
        }                                                                      \
    } while (0)

// ---- logging --------------------------------------------------------------
enum class LogLevel { Silent = 0, Error = 1, Warn = 2, Info = 3, Debug = 4 };

inline LogLevel& log_level() {
    static LogLevel lvl = LogLevel::Info;
    return lvl;
}

inline void log_impl(LogLevel lvl, const char* tag, const char* fmt, ...) {
    if (static_cast<int>(lvl) > static_cast<int>(log_level())) return;
    fprintf(stderr, "[%s] ", tag);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

#define LOG_ERROR(...) ::llm::log_impl(::llm::LogLevel::Error, "err ", __VA_ARGS__)
#define LOG_WARN(...)  ::llm::log_impl(::llm::LogLevel::Warn,  "warn", __VA_ARGS__)
#define LOG_INFO(...)  ::llm::log_impl(::llm::LogLevel::Info,  "info", __VA_ARGS__)
#define LOG_DEBUG(...) ::llm::log_impl(::llm::LogLevel::Debug, "dbg ", __VA_ARGS__)

// ---- timing ---------------------------------------------------------------
// Monotonic seconds as a double. Used for TTFT / tokens-per-second reporting.
inline double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// ---- aligned allocation ---------------------------------------------------
// 64-byte alignment keeps NEON/SVE loads happy and avoids straddling cache
// lines. We wrap posix_memalign so tensor buffers are always vectorizable.
inline void* aligned_alloc64(size_t bytes) {
    void* p = nullptr;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) throw Error("aligned_alloc64 failed");
    return p;
}
inline void aligned_free(void* p) { free(p); }

// Round `n` up to a multiple of `mult`.
inline size_t round_up(size_t n, size_t mult) {
    return (n + mult - 1) / mult * mult;
}

// ---- resident set size (actual RAM) ---------------------------------------
// The number that proves the streaming claim: it stays flat as layers rotate
// through instead of climbing to the full model size. Returns 0 if
// unavailable.
//   * macOS/iOS: mach task_info(MACH_TASK_BASIC_INFO).resident_size.
//   * Linux/Android: /proc/self/statm resident pages * page size.
inline size_t current_rss_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        return 0;
    return static_cast<size_t>(info.resident_size);
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long total_pages = 0, resident_pages = 0;
    if (std::fscanf(f, "%ld %ld", &total_pages, &resident_pages) != 2) {
        std::fclose(f); return 0;
    }
    std::fclose(f);
    const long page = sysconf(_SC_PAGESIZE);
    return static_cast<size_t>(resident_pages) * page;
#endif
}

} // namespace llm
