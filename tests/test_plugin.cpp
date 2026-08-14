// tests/test_plugin.cpp — plugin seam: Kosh (context) + RTK (runtime/KV) +
// host.
//
// These exercise the plugin CONTRACT directly (no model needed). The default
// runtime path (no PluginHost installed) is proven byte-identical by the
// existing e2e / stress / sampler suites, which are unchanged by this seam.
#include "llm/kosh.h"
#include "llm/plugin.h"
#include "llm/rtk.h"
#include "tests/test_util.h"

#include <cstdio>
#include <stdexcept>

using namespace llm;

// ---- Kosh v0: token-run collapse -------------------------------------------

TEST(kosh_v0_collapses_long_run) {
  auto k = make_kosh_v0(/*max_run=*/3);
  KoshRequest req;
  req.tokens = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5}; // 10 identical
  KoshResult r = k->optimize(req);
  CHECK(r.tokens_in == 10);
  CHECK(r.tokens_out == 3);
  CHECK(r.modified);
  CHECK(r.tokens == (std::vector<int64_t>{5, 5, 5}));
}

TEST(kosh_v0_preserves_short_runs_identity) {
  auto k = make_kosh_v0(/*max_run=*/3);
  KoshRequest req;
  req.tokens = {1, 2, 2, 3, 3, 3}; // no run exceeds 3
  KoshResult r = k->optimize(req);
  CHECK(r.tokens_in == 6);
  CHECK(r.tokens_out == 6);
  CHECK(!r.modified);
  CHECK(r.tokens == req.tokens);
}

TEST(kosh_v0_collapses_only_overlong_runs) {
  auto k = make_kosh_v0(/*max_run=*/2);
  KoshRequest req;
  req.tokens = {7, 7, 7, 7, 1, 9, 9, 9, 9, 9}; // 7x4 -> 7x2 ; 9x5 -> 9x2
  KoshResult r = k->optimize(req);
  CHECK(r.tokens == (std::vector<int64_t>{7, 7, 1, 9, 9}));
  CHECK(r.tokens_in == 10);
  CHECK(r.tokens_out == 5);
  CHECK(r.modified);
}

TEST(kosh_v0_empty_is_safe) {
  auto k = make_kosh_v0();
  KoshRequest req; // empty tokens
  KoshResult r = k->optimize(req);
  CHECK(r.tokens_in == 0);
  CHECK(r.tokens_out == 0);
  CHECK(!r.modified);
  CHECK(r.tokens.empty());
}

TEST(kosh_v0_metrics_accumulate) {
  auto k = make_kosh_v0(/*max_run=*/1);
  KoshRequest a;
  a.tokens = {4, 4, 4}; // -> {4}
  KoshRequest b;
  b.tokens = {1, 2}; // -> {1,2} (no over-long run)
  k->optimize(a);
  k->optimize(b);
  KoshMetrics m = k->metrics();
  CHECK(m.requests == 2);
  CHECK(m.tokens_in == 5);  // 3 + 2
  CHECK(m.tokens_out == 3); // 1 + 2
  CHECK(m.reductions == 1); // only request a shrank
}

// ---- PluginHost lifecycle + graceful fallback ------------------------------

TEST(plugin_host_null_by_default) {
  PluginHost h;
  CHECK(h.kosh() == nullptr);
  CHECK(h.rtk() == nullptr);
  CHECK(!h.inited());
}

TEST(plugin_host_installs_and_inits) {
  PluginHost h;
  h.set_kosh(make_kosh_v0());
  h.set_rtk(make_rtk_v0());
  ModelConfig cfg;
  RtkKvView v;
  CHECK(h.init(cfg, v));
  CHECK(h.inited());
  CHECK(h.kosh() != nullptr);
  CHECK(h.rtk() != nullptr);
}

namespace {
// A Kosh whose init() reports failure — the host must DISABLE (drop) it.
struct FalseInitKosh final : KoshPlugin {
  const char *name() const override { return "false-init"; }
  bool init(const ModelConfig &) override { return false; }
  KoshResult optimize(const KoshRequest &req) override {
    KoshResult r;
    r.tokens = req.tokens;
    return r;
  }
};
// A Kosh whose init() throws — the host must catch and DISABLE it.
struct ThrowInitKosh final : KoshPlugin {
  const char *name() const override { return "throw-init"; }
  bool init(const ModelConfig &) override { throw std::runtime_error("boom"); }
  KoshResult optimize(const KoshRequest &req) override {
    KoshResult r;
    r.tokens = req.tokens;
    return r;
  }
};
} // namespace

TEST(plugin_host_disables_failing_init) {
  {
    PluginHost h;
    h.set_kosh(std::make_unique<FalseInitKosh>());
    ModelConfig cfg;
    RtkKvView v;
    h.init(cfg, v);
    CHECK(h.kosh() == nullptr); // dropped: init returned false
  }
  {
    PluginHost h;
    h.set_kosh(std::make_unique<ThrowInitKosh>());
    ModelConfig cfg;
    RtkKvView v;
    h.init(cfg, v);             // must not propagate the exception
    CHECK(h.kosh() == nullptr); // dropped: init threw
  }
}

// ---- RTK v0: runtime-state observer ----------------------------------------

TEST(rtk_v0_observes_and_accounts) {
  auto r = make_rtk_v0();
  ModelConfig cfg;
  RtkKvView v0;
  v0.seq_len = 5;
  v0.bytes = 100;
  CHECK(r->init(cfg, v0)); // observes: max_seq=5, peak=100

  RtkKvView v1;
  v1.seq_len = 8;
  v1.bytes = 160;
  r->on_prefill(v1); // max_seq=8, peak=160

  RtkKvView v2;
  v2.seq_len = 9;
  v2.bytes = 150; // seq grows, bytes below peak
  r->on_step(v2);

  RtkMetrics m = r->metrics();
  CHECK(m.steps == 1);
  CHECK(m.max_seq == 9);
  CHECK(m.peak_kv_bytes == 160);
}

int main() {
  printf("== test_plugin ==\n");
  return llmtest::run_all();
}
