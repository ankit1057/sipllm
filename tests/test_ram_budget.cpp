// test_ram_budget.cpp — issue #37: --ram-budget hard ceiling + partial layer
// residency.
//
// Two invariants, both exact:
//   1. Pinning is a PURE CACHE. Serving a layer from the pinned residency set
//      returns byte-identical WeightRefs to streaming it, so the forward pass is
//      bit-for-bit identical for ANY budget (including budget==0, today's path).
//   2. The budget is a HARD CEILING on weight-resident RAM: for any budget at or
//      above the streaming floor F (globals + the cold-stream ring), the loader's
//      resident_bytes() never exceeds it. Below F the loader pins nothing and
//      degrades to pure streaming (it cannot go below the physical floor).
//
// The dial is also exercised: budget 0 pins nothing, a huge budget pins every
// layer, and a mid budget pins a strict subset.
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/model.h"
#include "llm/sampler.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

struct RunOut {
    std::vector<float> logits;
    std::vector<float> kv_k, kv_v;
    size_t resident = 0;
    int    pinned = 0;
    int    n_layers = 0;
};

// Run the whole prompt through forward() (one layer stream per token per layer),
// under a given WEIGHT budget, and capture logits + KV + residency facts.
static RunOut run(const std::string& path, Residency res, bool async, int nbuf,
                  size_t weight_budget, const std::vector<int64_t>& toks) {
    ModelFile f(path, /*mmap=*/false);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader::Options opt;
    opt.residency = res;
    opt.async = async;
    opt.n_buffers = async ? nbuf : 1;
    opt.ram_budget_bytes = weight_budget;
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(4);
    Transformer tf(&loader, &kv, &pool);

    const float* logits = nullptr;
    for (int64_t pos = 0; pos < (int64_t)toks.size(); ++pos)
        logits = tf.forward(toks[pos], pos);

    RunOut o;
    o.logits.assign(logits, logits + cfg.vocab_size);
    const int64_t nl = kv.n_layers(), kvd = kv.kv_dim(), n = (int64_t)toks.size();
    for (int64_t l = 0; l < nl; ++l)
        for (int64_t p = 0; p < n; ++p) {
            const float* k = kv.k(l, p); const float* v = kv.v(l, p);
            o.kv_k.insert(o.kv_k.end(), k, k + kvd);
            o.kv_v.insert(o.kv_v.end(), v, v + kvd);
        }
    o.resident = loader.resident_bytes();
    o.pinned   = loader.pinned_layers();
    o.n_layers = (int)cfg.n_layers;
    return o;
}

static void expect_identical(const RunOut& a, const RunOut& b, const std::string& what) {
    CHECK_MSG(a.logits.size() == b.logits.size(), what + ": logits size");
    for (size_t i = 0; i < a.logits.size(); ++i)
        CHECK_MSG(a.logits[i] == b.logits[i], what + ": logits differ at " + std::to_string(i));
    CHECK_MSG(a.kv_k.size() == b.kv_k.size(), what + ": kv_k size");
    for (size_t i = 0; i < a.kv_k.size(); ++i)
        CHECK_MSG(a.kv_k[i] == b.kv_k[i], what + ": kv_k differ at " + std::to_string(i));
    CHECK_MSG(a.kv_v.size() == b.kv_v.size(), what + ": kv_v size");
    for (size_t i = 0; i < a.kv_v.size(); ++i)
        CHECK_MSG(a.kv_v[i] == b.kv_v[i], what + ": kv_v differ at " + std::to_string(i));
    CHECK_MSG(Sampler::argmax(a.logits.data(), (int64_t)a.logits.size()) ==
              Sampler::argmax(b.logits.data(), (int64_t)b.logits.size()),
              what + ": argmax differs");
}

static const size_t kHuge = (size_t)1 << 30;  // 1 GiB: pins every toy layer

// Invariant 1: any budget yields bit-identical output to unlimited streaming.
TEST(ram_budget_logits_bit_identical) {
    ToyConfig tc; tc.n_layers = 6; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 321;
    std::string path = scratch("toy_rb_ident.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> toks = {3, 1, 4, 1, 5, 9, 2, 6};

    RunOut base = run(path, Residency::Quantized, true, 2, 0, toks);        // today's path
    RunOut full = run(path, Residency::Quantized, true, 2, kHuge, toks);    // all pinned
    CHECK_MSG(base.pinned == 0, "budget 0 pins nothing");
    CHECK_MSG(full.pinned == tc.n_layers, "huge budget pins every layer");

    // Derive a budget that provably pins a strict subset: measure the streaming
    // floor (budget 1 pins nothing), then aim halfway between floor and full so a
    // few layers fit above the cold-stream ring reserve.
    const size_t floor = run(path, Residency::Quantized, true, 2, 1, toks).resident;
    const size_t mid = floor + (full.resident - floor) / 2;
    RunOut half = run(path, Residency::Quantized, true, 2, mid, toks);
    CHECK_MSG(half.pinned > 0 && half.pinned < tc.n_layers, "mid budget pins a strict subset");

    expect_identical(base, full, "quant huge");
    expect_identical(base, half, "quant mid");

    // fp32 residency and the synchronous single-buffer path must also match.
    RunOut base_fp = run(path, Residency::FP32, false, 1, 0, toks);
    RunOut full_fp = run(path, Residency::FP32, false, 1, kHuge, toks);
    CHECK_MSG(full_fp.pinned == tc.n_layers, "fp32 huge budget pins every layer");
    expect_identical(base_fp, full_fp, "fp32 sync huge");
}

// Invariant 2: for every budget >= the streaming floor, resident weight RAM
// stays at or below the budget (hard ceiling); below the floor it degrades to
// pure streaming (pins 0, runs at the floor).
TEST(ram_budget_hard_ceiling) {
    ToyConfig tc; tc.n_layers = 8; tc.dim = 64; tc.n_heads = 8; tc.n_kv_heads = 4;
    tc.ffn_dim = 128; tc.vocab_size = 48; tc.seed = 99;
    std::string path = scratch("toy_rb_ceiling.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> toks = {7, 3, 11, 42 % tc.vocab_size, 5, 1, 0, 9};

    // Streaming floor: a 1-byte budget pins nothing; resident is the floor F.
    RunOut floor = run(path, Residency::Quantized, true, 2, 1, toks);
    CHECK_MSG(floor.pinned == 0, "sub-floor budget pins nothing");
    const size_t F = floor.resident;

    RunOut full = run(path, Residency::Quantized, true, 2, kHuge, toks);
    const size_t all_weights = full.resident;
    CHECK_MSG(all_weights > F, "full residency exceeds the streaming floor");

    // Sweep budgets from the floor up past the whole model, including awkward
    // fuzzed values, and assert the ceiling holds every time.
    const size_t span = all_weights - F;
    size_t budgets[] = {
        F, F + span / 7, F + span / 3, F + span / 2,
        F + (span * 2) / 3, F + span - 1, all_weights, all_weights + span,
        F + 12345, F + 1, kHuge,
    };
    bool saw_partial = false;
    for (size_t b : budgets) {
        RunOut r = run(path, Residency::Quantized, true, 2, b, toks);
        CHECK_MSG(r.resident <= b, "ceiling: resident " + std::to_string(r.resident) +
                                   " > budget " + std::to_string(b));
        CHECK_MSG(r.pinned >= 0 && r.pinned <= tc.n_layers, "pinned in range");
        if (r.pinned > 0 && r.pinned < tc.n_layers) saw_partial = true;
    }
    CHECK_MSG(saw_partial, "sweep exercised a partial (dial) residency point");

    // Below the physical floor the loader cannot shrink further: it pins nothing
    // and runs at the floor rather than violating correctness.
    RunOut tiny = run(path, Residency::Quantized, true, 2, F / 2, toks);
    CHECK_MSG(tiny.pinned == 0 && tiny.resident == F, "sub-floor degrades to streaming");
}

int main() {
    printf("== test_ram_budget ==\n");
    return llmtest::run_all();
}
