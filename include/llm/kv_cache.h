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

#include <cstring>
#include <vector>

namespace llm {

class KVCache {
public:
    KVCache(int64_t n_layers, int64_t kv_dim, int64_t max_ctx)
        : n_layers_(n_layers), kv_dim_(kv_dim), max_ctx_(max_ctx) {
        // Start with a small capacity instead of the full max_ctx window. Most
        // sessions are far shorter than the trained context, and growth is
        // amortized O(1) via doubling. Never exceed max_ctx (the hard ceiling
        // the runtime enforces before every forward) or underflow to zero.
        cap_ = max_ctx_ < kInitialCap ? max_ctx_ : kInitialCap;
        if (cap_ < 1) cap_ = max_ctx_ < 1 ? 1 : max_ctx_;
        alloc(cap_);
    }

    int64_t max_ctx()  const { return max_ctx_; }
    int64_t kv_dim()   const { return kv_dim_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t seq_len()  const { return seq_len_; }
    int64_t capacity() const { return cap_; }   // currently-resident positions

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
    float* k(int64_t layer, int64_t pos) {
        if (pos >= cap_) grow_to(pos + 1);
        return k_.data() + offset(layer, pos);
    }
    float* v(int64_t layer, int64_t pos) {
        if (pos >= cap_) grow_to(pos + 1);
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
    static constexpr int64_t kInitialCap = 64;

    size_t offset(int64_t layer, int64_t pos) const {
        return (((size_t)layer * cap_) + pos) * kv_dim_;
    }

    void alloc(int64_t cap) {
        const size_t per = (size_t)n_layers_ * cap * kv_dim_;
        k_.assign(per, 0.f);
        v_.assign(per, 0.f);
    }

    // Grow capacity to at least `need` positions (doubling, capped at max_ctx),
    // re-laying-out existing rows because the per-layer stride is `cap_`.
    void grow_to(int64_t need) {
        if (need <= cap_) return;
        int64_t nc = cap_ > 0 ? cap_ : 1;
        while (nc < need) nc <<= 1;
        if (nc > max_ctx_) nc = max_ctx_;   // need <= max_ctx_ (runtime-enforced)

        const size_t per = (size_t)n_layers_ * nc * kv_dim_;
        std::vector<float> nk(per, 0.f), nv(per, 0.f);
        // Copy each layer's rows from the old stride (cap_) to the new (nc).
        const size_t old_row = (size_t)cap_ * kv_dim_;
        for (int64_t l = 0; l < n_layers_; ++l) {
            const size_t src = (size_t)l * cap_ * kv_dim_;
            const size_t dst = (size_t)l * nc * kv_dim_;
            std::memcpy(nk.data() + dst, k_.data() + src, old_row * sizeof(float));
            std::memcpy(nv.data() + dst, v_.data() + src, old_row * sizeof(float));
        }
        k_.swap(nk);
        v_.swap(nv);
        cap_ = nc;
    }

    int64_t n_layers_, kv_dim_, max_ctx_;
    int64_t cap_ = 0;
    int64_t seq_len_ = 0;
    std::vector<float> k_, v_;
};

} // namespace llm
