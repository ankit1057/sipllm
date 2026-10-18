// test_spec_equiv.cpp — speculative decoding (prompt lookup) ≡ plain decoding.
//
// Contract: with --spec N the verify pass streams every layer once for K+1
// positions, but every emitted token is still sampled from the model's own
// distribution given its true prefix. So under greedy sampling the generated
// token sequence must be IDENTICAL to the plain one-token-per-forward path,
// and the default (spec off) path must be untouched. Uses the byte-tokenizer
// toy model (deterministic, 1 token/byte), single-threaded.
#include "llm/runtime.h"
#include "llm/toy_model.h"
#include "tests/test_util.h"

#include <memory>
#include <string>
#include <vector>

using namespace llm;

static std::string make_toy() {
  ToyConfig tc;
  tc.n_layers = 3;
  tc.dim = 32;
  tc.n_heads = 4;
  tc.n_kv_heads = 2;
  tc.ffn_dim = 64;
  tc.vocab_size = 256; // byte tokenizer
  tc.ctx_len = 256;
  tc.seed = 23;
  std::string path = llmtest::scratch_path("toy_spec.llmw");
  write_toy_model(path, tc);
  return path;
}

struct Gen {
  std::vector<int64_t> ids;
  std::string text;
  GenStats st;
};

static Gen run(const std::string& path, const std::string& prompt, int max_new,
               int spec_draft, SamplerConfig scfg) {
  auto src = open_model(path);
  LayerLoader::Options opt;
  opt.residency = Residency::FP32;
  opt.async = false;
  opt.n_buffers = 1;
  Runtime rt(std::move(src), opt, /*max_ctx=*/0, /*threads=*/1);
  if (spec_draft > 0) rt.enable_speculative_decoding(spec_draft);
  Gen g;
  g.text = rt.generate(prompt, max_new, scfg,
                       [&](const std::string&, int64_t id) { g.ids.push_back(id); return true; },
                       &g.st);
  return g;
}

// A prompt with repeated n-grams so prompt lookup actually finds drafts.
static const char* kRepetitive =
    "abcabcabcabc abcabcabcabc xyzxyzxyz abcabcabcabc";

TEST(spec_off_by_default) {
  const std::string path = make_toy();
  SamplerConfig g; g.temperature = 0.f;
  Gen plain = run(path, kRepetitive, 24, /*spec=*/0, g);
  CHECK(!plain.st.spec_active);
  CHECK(plain.st.spec_batches == 0);
  CHECK(plain.st.spec_drafted == 0);
  CHECK(plain.st.spec_accepted == 0);
  CHECK((int)plain.ids.size() == plain.st.gen_tokens);
}

TEST(spec_greedy_matches_plain) {
  const std::string path = make_toy();
  SamplerConfig g; g.temperature = 0.f;
  for (int K : {1, 3, 5}) {
    Gen plain = run(path, kRepetitive, 32, 0, g);
    Gen spec  = run(path, kRepetitive, 32, K, g);
    CHECK(spec.st.spec_active);
    CHECK(spec.st.spec_batches > 0);          // the speculative path ran
    CHECK(spec.st.spec_drafted >= spec.st.spec_accepted);
    CHECK(spec.st.gen_tokens == plain.st.gen_tokens);
    CHECK(spec.ids == plain.ids);             // token-for-token identical
    CHECK(spec.text == plain.text);
    CHECK((int)spec.ids.size() == spec.st.gen_tokens);
  }
}

TEST(spec_respects_max_new) {
  const std::string path = make_toy();
  SamplerConfig g; g.temperature = 0.f;
  for (int n : {1, 2, 5, 7}) {
    Gen spec = run(path, kRepetitive, n, /*spec=*/4, g);
    CHECK(spec.st.gen_tokens <= n);
    CHECK((int)spec.ids.size() == spec.st.gen_tokens);
    Gen plain = run(path, kRepetitive, n, 0, g);
    CHECK(spec.ids == plain.ids);
  }
}

TEST(spec_seeded_sampling_matches_plain) {
  // Each verify step consumes exactly one RNG draw from the same logits the
  // plain path would see, so a seeded temperature run is identical too. The
  // random-weight toy has near-flat logits, so any real temperature emits
  // noise that never repeats an n-gram and prompt lookup never drafts (the
  // test would be vacuous). A near-zero temperature keeps the output periodic
  // while still going through sample_temperature() and the RNG every step,
  // which is exactly what this case is meant to pin down.
  const std::string path = make_toy();
  SamplerConfig s; s.temperature = 0.01f; s.top_k = 40; s.seed = 7;
  Gen plain = run(path, kRepetitive, 24, 0, s);
  Gen spec  = run(path, kRepetitive, 24, 3, s);
  CHECK(spec.st.spec_batches > 0);
  CHECK(spec.ids == plain.ids);
}

TEST(spec_stop_callback_honoured) {
  const std::string path = make_toy();
  SamplerConfig g; g.temperature = 0.f;
  auto src = open_model(path);
  LayerLoader::Options opt;
  opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
  Runtime rt(std::move(src), opt, 0, 1);
  rt.enable_speculative_decoding(4);
  int seen = 0;
  GenStats st;
  rt.generate(kRepetitive, 32, g,
              [&](const std::string&, int64_t) { return ++seen < 5; }, &st);
  CHECK(seen == 5);              // stopped exactly when the callback said so
  CHECK(st.gen_tokens == 5);
}

int main() { return llmtest::run_all(); }
