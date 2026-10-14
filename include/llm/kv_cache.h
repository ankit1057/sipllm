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
//
// RFC-003 Phase 1 — grow-on-demand allocation. The cache used to commit the
// FULL max_ctx footprint up front (`assign(n_layers·max_ctx·kv_dim, 0)`), so a
// 16-token chat paid for the whole 8k window: on smollm2-135m that dense fp32
// KV was 188.7 MB = 79% of peak RSS. Instead we track a live capacity `cap_`
// (in positions), start small, and double it (capped at max_ctx) only as the
// sequence advances. The [layer][pos][kv_dim] layout uses `cap_` as the
// per-layer stride, so a grow changes that stride: we re-lay-out (memcpy) the
// existing rows into the wider buffer. Values are copied verbatim, so results
// are bitwise-identical to the old full-preallocation path — this is a pure RAM
// win with zero accuracy cost. `bytes()` reports the true resident footprint so
// GenStats.kv_bytes shrinks accordingly.
#pragma once

#include "llm/common.h"
#include "llm/dtype.h"

#include <cstring>
#include <vector>

namespace llm {

enum class KVPrecision { FP32, Q8_0 };

class KVCache {
public:
    KVCache(int64_t n_layers, int64_t kv_dim, int64_t max_ctx, KVPrecision precision = KVPrecision::FP32)
        : n_layers_(n_layers), kv_dim_(kv_dim), max_ctx_(max_ctx), precision_(precision) {
        // Start with a small capacity instead of the full max_ctx window. Most
        // sessions are far shorter than the trained context, and growth is
        // amortized O(1) via doubling. Never exceed max_ctx (the hard ceiling
        // the runtime enforces before every forward) or underflow to zero.
        cap_ = max_ctx_ < kInitialCap ? max_ctx_ : kInitialCap;
        if (cap_ < 1) cap_ = max_ctx_ < 1 ? 1 : max_ctx_;
        
        if (precision_ == KVPrecision::Q8_0) {
            bytes_per_row_ = type_nbytes(DType::Q8_0, kv_dim_);
        } else {
            bytes_per_row_ = kv_dim_ * sizeof(float);
        }
        alloc(cap_);
    }

    int64_t max_ctx()  const { return max_ctx_; }
    int64_t kv_dim()   const { return kv_dim_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t seq_len()  const { return seq_len_; }
    int64_t capacity() const { return cap_; }   // currently-resident positions
    KVPrecision precision() const { return precision_; }

    // Advance the filled length after writing position `pos`. Also ensures the
    // backing store can hold `n` positions (writes normally grow it first via
    // k()/v(), but this keeps the invariant if a caller sets it ahead).
    void set_seq_len(int64_t n) {
        if (n > cap_) grow_to(n);
        seq_len_ = n;
    }
    void clear() { seq_len_ = 0; }

    // Write/read accessors. The non-const overloads are the write path: they
    // grow the cache so `pos` is resident before handing back the pointer. The
    // const overloads are read-only and never grow (every position read has
    // already been written, hence is within cap_).
    void* k_ptr(int64_t layer, int64_t pos) {
        if (pos >= cap_) grow_to(pos + 1);
        return (void*)(k_.data() + offset_bytes(layer, pos));
    }
    void* v_ptr(int64_t layer, int64_t pos) {
        if (pos >= cap_) grow_to(pos + 1);
        return (void*)(v_.data() + offset_bytes(layer, pos));
    }
    const void* k_ptr(int64_t layer, int64_t pos) const {
        return (const void*)(k_.data() + offset_bytes(layer, pos));
    }
    const void* v_ptr(int64_t layer, int64_t pos) const {
        return (const void*)(v_.data() + offset_bytes(layer, pos));
    }

    float* k(int64_t layer, int64_t pos) { return (float*)k_ptr(layer, pos); }
    float* v(int64_t layer, int64_t pos) { return (float*)v_ptr(layer, pos); }
    const float* k(int64_t layer, int64_t pos) const { return (const float*)k_ptr(layer, pos); }
    const float* v(int64_t layer, int64_t pos) const { return (const float*)v_ptr(layer, pos); }

    // ---- Semantic Cache Extensions ----
    // Return raw pointers to the start of a layer's KV block (used by memcpy bypass)
    void* k_ptr_layer(int64_t layer) { return (void*)(k_.data() + offset_bytes(layer, 0)); }
    void* v_ptr_layer(int64_t layer) { return (void*)(v_.data() + offset_bytes(layer, 0)); }
    
    // Inject a contiguous block of cached memory into this KV cache
    void inject(int64_t layer, const float* k_src, const float* v_src, int64_t num_positions) {
        if (precision_ != KVPrecision::FP32) throw std::runtime_error("SemanticCache only supports FP32 currently");
        if (num_positions > cap_) grow_to(num_positions);
        std::memcpy(k_ptr_layer(layer), k_src, num_positions * bytes_per_row_);
        std::memcpy(v_ptr_layer(layer), v_src, num_positions * bytes_per_row_);
        if (num_positions > seq_len_) seq_len_ = num_positions;
    }

    size_t bytes() const { return k_.size() + v_.size(); }

private:
    static constexpr int64_t kInitialCap = 64;

    size_t offset_bytes(int64_t layer, int64_t pos) const {
        return (((size_t)layer * cap_) + pos) * bytes_per_row_;
    }

    void alloc(int64_t cap) {
        const size_t per = (size_t)n_layers_ * cap * bytes_per_row_;
        k_.assign(per, 0);
        v_.assign(per, 0);
    }

    // Grow capacity to at least `need` positions (doubling, capped at max_ctx),
    // re-laying-out existing rows because the per-layer stride is `cap_`.
    void grow_to(int64_t need) {
        if (need <= cap_) return;
        int64_t nc = cap_ > 0 ? cap_ : 1;
        while (nc < need) nc <<= 1;
        if (nc > max_ctx_) nc = max_ctx_;   // need <= max_ctx_ (runtime-enforced)

        const size_t per = (size_t)n_layers_ * nc * bytes_per_row_;
        std::vector<uint8_t> nk(per, 0), nv(per, 0);
        // Copy each layer's rows from the old stride (cap_) to the new (nc).
        const size_t old_row = (size_t)cap_ * bytes_per_row_;
        for (int64_t l = 0; l < n_layers_; ++l) {
            const size_t src = (size_t)l * cap_ * bytes_per_row_;
            const size_t dst = (size_t)l * nc * bytes_per_row_;
            std::memcpy(nk.data() + dst, k_.data() + src, old_row);
            std::memcpy(nv.data() + dst, v_.data() + src, old_row);
        }
        k_.swap(nk);
        v_.swap(nv);
        cap_ = nc;
    }

    int64_t n_layers_, kv_dim_, max_ctx_;
    KVPrecision precision_;
    size_t bytes_per_row_;
    int64_t cap_ = 0;
    int64_t seq_len_ = 0;
    std::vector<uint8_t> k_, v_;
};

} // namespace llm
