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

#include <algorithm>
#include <cstring>
#include <vector>

namespace llm {

enum class KVPrecision { FP32, Q8_0 };

// Per-layer layout (Gemma 4). Every layer may have its own row width, and a
// layer with window W > 0 is a RING of min(W, max_ctx) rows: position p lives
// in slot p % W, because a sliding-window layer never attends further back
// than W-1 positions. The uniform constructor gives every layer the same width
// and no window, which is byte-for-byte the classic [layer][pos][kv_dim] cache.
// kv_dim() reports the widest row (what a per-position buffer must hold).
class KVCache {
public:
    // `layer_base` (Issue #70): a node that runs only layers
    // [layer_base, layer_base + n_layers) allocates KV for those alone; callers
    // keep using global layer ids. 0 == today's layout, exactly.
    KVCache(int64_t n_layers, int64_t kv_dim, int64_t max_ctx, KVPrecision precision = KVPrecision::FP32,
            int64_t layer_base = 0)
        : KVCache(std::vector<int64_t>((size_t)n_layers, kv_dim),
                  std::vector<int64_t>((size_t)n_layers, 0), max_ctx, precision, layer_base) {}

    // Entry i of the per-layer vectors describes global layer layer_base + i.
    KVCache(const std::vector<int64_t>& layer_kv_dim, const std::vector<int64_t>& layer_window,
            int64_t max_ctx, KVPrecision precision = KVPrecision::FP32, int64_t layer_base = 0)
        : n_layers_((int64_t)layer_kv_dim.size()), max_ctx_(max_ctx), precision_(precision),
          layer_base_(layer_base) {
        // Start with a small capacity instead of the full max_ctx window. Most
        // sessions are far shorter than the trained context, and growth is
        // amortized O(1) via doubling. Never exceed max_ctx (the hard ceiling
        // the runtime enforces before every forward) or underflow to zero.
        cap_ = max_ctx_ < kInitialCap ? max_ctx_ : kInitialCap;
        if (cap_ < 1) cap_ = max_ctx_ < 1 ? 1 : max_ctx_;
        layers_.resize((size_t)n_layers_);
        for (int64_t l = 0; l < n_layers_; ++l) {
            Layer& L = layers_[(size_t)l];
            L.kv_dim = layer_kv_dim[(size_t)l];
            const int64_t w = l < (int64_t)layer_window.size() ? layer_window[(size_t)l] : 0;
            L.window = (w > 0 && w < max_ctx_) ? w : 0;   // a window >= ctx is no ring
            L.row_bytes = precision_ == KVPrecision::Q8_0 ? type_nbytes(DType::Q8_0, L.kv_dim)
                                                          : (size_t)L.kv_dim * sizeof(float);
            kv_dim_ = std::max(kv_dim_, L.kv_dim);
            if (L.window > 0) has_ring_ = true;
            L.cap = eff_cap(L, cap_);
            L.k.assign((size_t)L.cap * L.row_bytes, 0);
            L.v.assign((size_t)L.cap * L.row_bytes, 0);
        }
    }

    int64_t max_ctx()  const { return max_ctx_; }
    int64_t kv_dim()   const { return kv_dim_; }
    int64_t kv_dim(int64_t layer) const { return at(layer).kv_dim; }
    int64_t window(int64_t layer) const { return at(layer).window; }
    size_t  row_bytes(int64_t layer) const { return at(layer).row_bytes; }
    bool    has_ring() const { return has_ring_; }
    int64_t n_layers() const { return n_layers_; }      // layers held (local count)
    int64_t layer_base() const { return layer_base_; }
    int64_t seq_len()  const { return seq_len_; }
    int64_t capacity() const { return cap_; }   // currently-resident positions
    KVPrecision precision() const { return precision_; }

    // Oldest position of `layer` still resident once `seq_len` positions have
    // been written (ring layers forget everything older than their window).
    int64_t first_valid(int64_t layer, int64_t seq_len) const {
        const int64_t w = at(layer).window;
        return (w > 0 && seq_len > w) ? seq_len - w : 0;
    }
    // Can the sequence be rewound to `pos` (keeping [0,pos)) after `written`
    // positions, with every position a later query can attend still resident?
    // Always true without ring layers; with them, only when nothing in the
    // window behind `pos` has been overwritten.
    bool can_rewind(int64_t pos, int64_t written) const {
        for (const Layer& L : layers_) {
            if (L.window == 0 || written <= L.window) continue;
            if (pos < written - 1) return false;
        }
        return true;
    }

    // Advance the filled length after writing position `pos`. Also ensures the
    // backing store can hold `n` positions (writes normally grow it first via
    // k()/v(), but this keeps the invariant if a caller sets it ahead).
    void set_seq_len(int64_t n) {
        if (n > cap_) grow_to(n);
        seq_len_ = n;
    }
    void clear() { seq_len_ = 0; }

    // Issue #81: allocate the whole max_ctx window now. Under --ram-budget the
    // planner prices the KV at full context; growing by doubling later would
    // hold old+new buffers at once and overshoot that price.
    void reserve_full() { grow_to(max_ctx_); }

    // Write/read accessors. The non-const overloads are the write path: they
    // grow the cache so `pos` is resident before handing back the pointer. The
    // const overloads are read-only and never grow (every position read has
    // already been written, hence is within cap_).
    void* k_ptr(int64_t layer, int64_t pos) {
        if (pos >= cap_) grow_to(pos + 1);
        Layer& L = at(layer);
        return (void*)(L.k.data() + slot(L, pos) * L.row_bytes);
    }
    void* v_ptr(int64_t layer, int64_t pos) {
        if (pos >= cap_) grow_to(pos + 1);
        Layer& L = at(layer);
        return (void*)(L.v.data() + slot(L, pos) * L.row_bytes);
    }
    const void* k_ptr(int64_t layer, int64_t pos) const {
        const Layer& L = at(layer);
        return (const void*)(L.k.data() + slot(L, pos) * L.row_bytes);
    }
    const void* v_ptr(int64_t layer, int64_t pos) const {
        const Layer& L = at(layer);
        return (const void*)(L.v.data() + slot(L, pos) * L.row_bytes);
    }

    // Float views are valid only for an FP32 cache: a Q8_0 row is shorter
    // than kv_dim floats. Byte-level users go through k_ptr()/row_bytes().
    float* k(int64_t layer, int64_t pos) { check_fp32(); return (float*)k_ptr(layer, pos); }
    float* v(int64_t layer, int64_t pos) { check_fp32(); return (float*)v_ptr(layer, pos); }
    const float* k(int64_t layer, int64_t pos) const { check_fp32(); return (const float*)k_ptr(layer, pos); }
    const float* v(int64_t layer, int64_t pos) const { check_fp32(); return (const float*)v_ptr(layer, pos); }

    size_t bytes() const {
        size_t b = 0;
        for (const Layer& L : layers_) b += L.k.size() + L.v.size();
        return b;
    }
    // Resident bytes for `n` positions under this layout (planning / reports).
    size_t bytes_for(int64_t n) const {
        size_t b = 0;
        for (const Layer& L : layers_) b += 2 * (size_t)eff_cap(L, std::min(n, max_ctx_)) * L.row_bytes;
        return b;
    }

    // Inject contiguous float KV data into layer from pos=0 to num_positions
    void inject(int64_t layer, const float* k_src, const float* v_src, int64_t num_positions) {
        check_fp32();
        if (num_positions > cap_) grow_to(num_positions);
        std::memcpy(k_ptr(layer, 0), k_src, num_positions * kv_dim_ * sizeof(float));
        std::memcpy(v_ptr(layer, 0), v_src, num_positions * kv_dim_ * sizeof(float));
        if (num_positions > seq_len_) seq_len_ = num_positions;
    }

private:
    static constexpr int64_t kInitialCap = 64;

    struct Layer {
        int64_t kv_dim = 0, window = 0, cap = 0;
        size_t row_bytes = 0;
        std::vector<uint8_t> k, v;
    };

    // Callers pass global layer ids; this node holds [layer_base_, +n_layers_).
    Layer&       at(int64_t layer)       { return layers_[(size_t)(layer - layer_base_)]; }
    const Layer& at(int64_t layer) const { return layers_[(size_t)(layer - layer_base_)]; }

    void check_fp32() const {
        LLM_CHECK(precision_ == KVPrecision::FP32, "KVCache: float access to a Q8_0 cache");
    }

    static int64_t eff_cap(const Layer& L, int64_t cap) {
        return L.window > 0 ? std::min(cap, L.window) : cap;
    }
    static size_t slot(const Layer& L, int64_t pos) {
        return (size_t)(L.window > 0 ? pos % L.window : pos);
    }

    // Grow capacity to at least `need` positions (doubling, capped at max_ctx).
    // A ring layer stops growing at its window; before it wraps, slot == pos,
    // so copying its leading rows preserves every value verbatim.
    void grow_to(int64_t need) {
        if (need <= cap_) return;
        int64_t nc = cap_ > 0 ? cap_ : 1;
        while (nc < need) nc <<= 1;
        if (nc > max_ctx_) nc = max_ctx_;   // need <= max_ctx_ (runtime-enforced)
        for (Layer& L : layers_) {
            const int64_t ncap = eff_cap(L, nc);
            if (ncap == L.cap) continue;
            L.k.resize((size_t)ncap * L.row_bytes, 0);
            L.v.resize((size_t)ncap * L.row_bytes, 0);
            L.cap = ncap;
        }
        cap_ = nc;
    }

    int64_t n_layers_, max_ctx_;
    int64_t kv_dim_ = 0;
    KVPrecision precision_;
    int64_t layer_base_ = 0;
    bool has_ring_ = false;
    int64_t cap_ = 0;
    int64_t seq_len_ = 0;
    std::vector<Layer> layers_;
};

} // namespace llm
