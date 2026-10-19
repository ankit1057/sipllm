// mem_manager.h — Byte-accounting and memory pressure management layer.
//
// Tracks memory consumption across functional categories (Weights, KV Cache,
// Workspace, Tensors, General), bounds allocations against a global memory ceiling,
// and signals memory pressure when approaching limits.
#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>

namespace llm {

enum class MemCategory {
    Weights,
    KVCache,
    Workspace,
    Tensors,
    General,
    Count
};

const char* mem_category_name(MemCategory cat);

class MemoryManager {
public:
    explicit MemoryManager(size_t limit_bytes = 0);

    // Set or update the memory limit (0 = unlimited).
    void set_limit(size_t limit_bytes);
    size_t limit() const { return limit_bytes_.load(); }

    // Request an allocation under `cat`. Returns true if accepted, false if over hard limit.
    bool allocate(size_t bytes, MemCategory cat = MemCategory::General);

    // Release previously allocated bytes.
    void deallocate(size_t bytes, MemCategory cat = MemCategory::General);

    // Current usage queries.
    size_t allocated(MemCategory cat) const;
    size_t total_allocated() const;
    size_t peak_allocated() const { return peak_allocated_.load(); }

    // Memory pressure: true if total allocation exceeds 85% of limit (when limit > 0).
    bool has_pressure() const;
    float pressure_ratio() const;

    // Reset peak tracking.
    void reset_peak();

private:
    std::atomic<size_t> limit_bytes_{0};
    std::atomic<size_t> allocated_by_cat_[static_cast<size_t>(MemCategory::Count)];
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_allocated_{0};
    mutable std::mutex mutex_;
};

// Singleton instance for global process byte accounting.
MemoryManager& global_mem_manager();

} // namespace llm
