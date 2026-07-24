// test_kv_cache.cpp — RFC-003 Phase 1: grow-on-demand KV allocation.
//
// The KV cache no longer commits the full max_ctx footprint up front; it starts
// at a small capacity and doubles (capped at max_ctx) as the sequence advances,
// re-laying-out the [layer][pos][kv_dim] rows on each grow. These tests pin the
// two guarantees that make that a *pure* RAM win:
//
//   1. Bitwise equivalence. Values stored and read back through the grown cache
//      are byte-identical to a full-preallocated reference (this is exactly the
//      old implementation, reproduced inline), at every position, including
//      positions written *before* a grow relaid the buffer out. This is what
//      keeps logits identical to main.
//   2. Smaller resident footprint. bytes() for a short session is far below the
//      full max_ctx footprint, and never exceeds it.
//
// Plus an end-to-end check: a real Transformer forward driven through a growth
// boundary produces bitwise-identical logits whether the cache grows on demand
// or is force-preallocated to max_ctx (the main-branch behavior).
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

// An independent full-preallocation reference: this is verbatim the pre-RFC-003
// KVCache (assign the whole n_layers*max_ctx*kv_dim buffer up front, stride =
// max_ctx). Its k()/v() outputs are the ground truth the grown cache must match.
struct FullCache {
    int64_t nl, kvd, mc;
    std::vector<float> k, v;
    FullCache(int64_t nl_, int64_t kvd_, int64_t mc_) : nl(nl_), kvd(kvd_), mc(mc_) {
        const size_t per = (size_t)nl * mc * kvd;
        k.assign(per, 0.f);
        v.assign(per, 0.f);
    }
    size_t off(int64_t l, int64_t p) const { return (((size_t)l * mc) + p) * kvd; }
    float* kk(int64_t l, int64_t p) { return k.data() + off(l, p); }
    float* vv(int64_t l, int64_t p) { return v.data() + off(l, p); }
    size_t bytes() const { return (k.size() + v.size()) * sizeof(float); }
};

// Mirror a deterministic write into both caches, then assert the grown cache
// reads back byte-identical to the full reference for EVERY (layer,pos) written
// so far — this is what catches a mistaken relayout after a grow.
TEST(kv_grow_bitwise_matches_full_prealloc) {
    const int64_t NL = 5, KVD = 8, MAXC = 300;
    KVCache kv(NL, KVD, MAXC);
    FullCache ref(NL, KVD, MAXC);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);

    const int64_t N = 200;   // crosses 64 -> 128 -> 256 growth boundaries
    for (int64_t pos = 0; pos < N; ++pos) {
        // Write this position for every layer (mirrors the transformer's
        // per-layer write at a fixed pos).
        for (int64_t l = 0; l < NL; ++l) {
            float* dk = kv.k(l, pos);
            float* dv = kv.v(l, pos);
            float* rk = ref.kk(l, pos);
            float* rv = ref.vv(l, pos);
            for (int64_t d = 0; d < KVD; ++d) {
                float a = dist(rng), b = dist(rng);
                dk[d] = rk[d] = a;
                dv[d] = rv[d] = b;
            }
        }
        kv.set_seq_len(pos + 1);
        // Every previously written row must still read back bit-for-bit — this
        // exercises the relayout copy when cap_ just changed.
        for (int64_t l = 0; l < NL; ++l) {
            for (int64_t p = 0; p <= pos; ++p) {
                CHECK(std::memcmp(kv.k(l, p), ref.kk(l, p), KVD * sizeof(float)) == 0);
                CHECK(std::memcmp(kv.v(l, p), ref.vv(l, p), KVD * sizeof(float)) == 0);
            }
        }
    }
    CHECK(kv.seq_len() == N);
    // After 200 positions the grown cache holds 256 (next pow2), still < max_ctx.
    CHECK(kv.capacity() == 256);
    CHECK(kv.capacity() <= kv.max_ctx());
}

// A short session must not pay for the full context: bytes() stays at the small
// initial capacity, far below the full max_ctx footprint.
TEST(kv_short_session_footprint_is_small) {
    const int64_t NL = 30, KVD = 192, MAXC = 8192;   // smollm2-135m shape
    KVCache kv(NL, KVD, MAXC);
    FullCache full(NL, KVD, MAXC);

    // Fresh cache: capacity is the small initial block, not max_ctx.
    CHECK(kv.capacity() < MAXC);
    for (int64_t pos = 0; pos < 16; ++pos) {   // a 16-token chat
        for (int64_t l = 0; l < NL; ++l) {
            for (int64_t d = 0; d < KVD; ++d) { kv.k(l, pos)[d] = 1.f; kv.v(l, pos)[d] = 1.f; }
        }
        kv.set_seq_len(pos + 1);
    }
    // 16 tokens fit inside the initial 64-position block: no growth happened.
    CHECK(kv.capacity() == 64);
    // Resident KV is a small fraction of the full max_ctx allocation (64/8192).
    CHECK(kv.bytes() * 100 < full.bytes());   // >100x smaller
}

// Capacity doubles and is hard-capped at max_ctx; it never over-allocates.
TEST(kv_capacity_caps_at_max_ctx) {
    const int64_t NL = 2, KVD = 4, MAXC = 100;   // not a power of two
    KVCache kv(NL, KVD, MAXC);
    CHECK(kv.capacity() == 64);                  // initial block
    // Force growth past the last power-of-two step; must clamp to max_ctx.
    kv.set_seq_len(MAXC);
    CHECK(kv.capacity() == MAXC);
    CHECK(kv.bytes() == (size_t)NL * MAXC * KVD * 2 * sizeof(float));
}

// End-to-end: a real Transformer forward through a growth boundary yields
// BITWISE-identical logits whether the cache grows on demand or is
// force-preallocated to max_ctx (the pre-RFC-003 / main behavior).
TEST(kv_grow_e2e_logits_identical_to_full_prealloc) {
    ToyConfig tc;
    tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.ctx_len = 300; tc.seed = 7;
    std::string path = scratch("toy_kvgrow.llmw");
    write_toy_model(path, tc);

    ModelFile f(path);
    ModelConfig cfg = ModelConfig::from_source(f);
    const int64_t T = 100;   // > 64 -> forces at least one grow (64 -> 128)
    std::vector<int64_t> tokens(T);
    for (int64_t i = 0; i < T; ++i) tokens[i] = (i * 7 + 3) % cfg.vocab_size;

    LayerLoader::Options opt;
    opt.residency = Residency::FP32; opt.async = false;

    // Run A: grow-on-demand (default fresh cache).
    std::vector<std::vector<float>> a;
    {
        LayerLoader loader(&f, cfg, opt);
        KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
        ThreadPool pool(4);
        Transformer tf(&loader, &kv, &pool);
        for (int64_t pos = 0; pos < T; ++pos) {
            const float* lg = tf.forward(tokens[pos], pos);
            a.emplace_back(lg, lg + cfg.vocab_size);
        }
        CHECK(kv.capacity() == 128);              // grew exactly once past 64
        CHECK(kv.capacity() < kv.max_ctx());
    }

    // Run B: full preallocation — force cap_ == max_ctx up front, exactly like
    // main's `assign(n_layers*max_ctx*kv_dim, 0)`, so no growth occurs.
    std::vector<std::vector<float>> b;
    size_t full_bytes = 0;
    {
        LayerLoader loader(&f, cfg, opt);
        KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
        kv.set_seq_len(kv.max_ctx());             // grow to full window...
        kv.clear();                               // ...then reset length (cap stays)
        CHECK(kv.capacity() == kv.max_ctx());
        full_bytes = kv.bytes();
        ThreadPool pool(4);
        Transformer tf(&loader, &kv, &pool);
        for (int64_t pos = 0; pos < T; ++pos) {
            const float* lg = tf.forward(tokens[pos], pos);
            b.emplace_back(lg, lg + cfg.vocab_size);
        }
    }

    CHECK(a.size() == b.size());
    for (size_t s = 0; s < a.size(); ++s) {
        CHECK(a[s].size() == b[s].size());
        // Bitwise identical, not just approximately equal.
        CHECK(std::memcmp(a[s].data(), b[s].data(), a[s].size() * sizeof(float)) == 0);
    }

    // And the grow-on-demand peak footprint is well under the full allocation.
    const size_t grow_peak = (size_t)cfg.n_layers * 128 * cfg.kv_dim() * 2 * sizeof(float);
    CHECK(grow_peak < full_bytes);
}

int main() {
    printf("== test_kv_cache ==\n");
    return llmtest::run_all();
}
