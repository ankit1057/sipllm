#include "llm/mem_manager.h"
#include <algorithm>

namespace llm {

const char* mem_category_name(MemCategory cat) {
    switch (cat) {
        case MemCategory::Weights:   return "Weights";
        case MemCategory::KVCache:   return "KVCache";
        case MemCategory::Workspace: return "Workspace";
        case MemCategory::Tensors:   return "Tensors";
        case MemCategory::General:   return "General";
        default:                     return "Unknown";
    }
}

MemoryManager::MemoryManager(size_t limit_bytes) {
    limit_bytes_.store(limit_bytes);
    for (size_t i = 0; i < static_cast<size_t>(MemCategory::Count); ++i) {
        allocated_by_cat_[i].store(0);
    }
    total_allocated_.store(0);
    peak_allocated_.store(0);
}

void MemoryManager::set_limit(size_t limit_bytes) {
    limit_bytes_.store(limit_bytes);
}

bool MemoryManager::allocate(size_t bytes, MemCategory cat) {
    if (bytes == 0) return true;
    size_t lim = limit_bytes_.load();

    std::lock_guard<std::mutex> lock(mutex_);
    size_t current = total_allocated_.load();
    if (lim > 0 && current + bytes > lim) {
        return false;
    }

    size_t idx = static_cast<size_t>(cat);
    if (idx < static_cast<size_t>(MemCategory::Count)) {
        allocated_by_cat_[idx].fetch_add(bytes);
    }

    size_t new_total = total_allocated_.fetch_add(bytes) + bytes;
    size_t peak = peak_allocated_.load();
    while (new_total > peak && !peak_allocated_.compare_exchange_weak(peak, new_total)) {}
    return true;
}

void MemoryManager::deallocate(size_t bytes, MemCategory cat) {
    if (bytes == 0) return;
    size_t idx = static_cast<size_t>(cat);
    if (idx < static_cast<size_t>(MemCategory::Count)) {
        size_t cur = allocated_by_cat_[idx].load();
        size_t sub = std::min(cur, bytes);
        allocated_by_cat_[idx].fetch_sub(sub);
    }

    size_t cur_tot = total_allocated_.load();
    size_t sub_tot = std::min(cur_tot, bytes);
    total_allocated_.fetch_sub(sub_tot);
}

size_t MemoryManager::allocated(MemCategory cat) const {
    size_t idx = static_cast<size_t>(cat);
    if (idx < static_cast<size_t>(MemCategory::Count)) {
        return allocated_by_cat_[idx].load();
    }
    return 0;
}

size_t MemoryManager::total_allocated() const {
    return total_allocated_.load();
}

bool MemoryManager::has_pressure() const {
    size_t lim = limit_bytes_.load();
    if (lim == 0) return false;
    return total_allocated_.load() >= (lim * 85) / 100;
}

float MemoryManager::pressure_ratio() const {
    size_t lim = limit_bytes_.load();
    if (lim == 0) return 0.0f;
    return static_cast<float>(total_allocated_.load()) / static_cast<float>(lim);
}

void MemoryManager::reset_peak() {
    peak_allocated_.store(total_allocated_.load());
}

MemoryManager& global_mem_manager() {
    static MemoryManager instance;
    return instance;
}

} // namespace llm
