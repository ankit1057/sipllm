// kv_cache.h — Task 6: reuse past keys and values.
//
// Without a KV cache, generating token T re-attends over all T positions by
// recomputing every past key/value — O(T^2) per step and O(T^3) overall. The
// cache stores each position's projected K and V once, so a decode step only
// projects the *new* token and attends against stored history.
//
// Because weights stream one layer at a time but the cache must survive across
// every layer and every token, it is indexed [layer][pos][kv_dim]. This is
// activation memory, not weights: n_layers · ctx · kv_dim · 2 · 4 bytes.
#pragma once

#include "llm/common.h"

#include <vector>

namespace llm {

class KVCache {
public:
    KVCache(int64_t n_layers, int64_t kv_dim, int64_t max_ctx)
        : n_layers_(n_layers), kv_dim_(kv_dim), max_ctx_(max_ctx) {
        const size_t per = (size_t)n_layers * max_ctx * kv_dim;
        k_.assign(per, 0.f);
        v_.assign(per, 0.f);
    }

    int64_t max_ctx()  const { return max_ctx_; }
    int64_t kv_dim()   const { return kv_dim_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t seq_len()  const { return seq_len_; }

    // Advance the filled length after writing position `pos`.
    void set_seq_len(int64_t n) { seq_len_ = n; }
    void clear() { seq_len_ = 0; }

    float* k(int64_t layer, int64_t pos) {
        return k_.data() + offset(layer, pos);
    }
    float* v(int64_t layer, int64_t pos) {
        return v_.data() + offset(layer, pos);
    }
    const float* k(int64_t layer, int64_t pos) const {
        return k_.data() + offset(layer, pos);
    }
    const float* v(int64_t layer, int64_t pos) const {
        return v_.data() + offset(layer, pos);
    }

    size_t bytes() const { return (k_.size() + v_.size()) * sizeof(float); }

private:
    size_t offset(int64_t layer, int64_t pos) const {
        return (((size_t)layer * max_ctx_) + pos) * kv_dim_;
    }
    int64_t n_layers_, kv_dim_, max_ctx_;
    int64_t seq_len_ = 0;
    std::vector<float> k_, v_;
};

} // namespace llm
