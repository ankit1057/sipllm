// test_reuse.cpp — cross-turn context reuse: prove reuse ≡ full-prefill.
//
// The reuse path reprocesses only the changed tail, reusing the KV of the
// longest common prefix already committed. Correctness gate: the last-position
// logits after a reuse-driven prefill must equal those of a from-scratch full
// prefill of the same context (bitwise-close). Uses a byte-tokenizer .llmw toy
// model (deterministic 1 token/byte), single-threaded for reproducibility.
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
  tc.seed = 11;
  std::string path = llmtest::scratch_path("toy_reuse.llmw");
  write_toy_model(path, tc);
  return path;
}

// Run a sequence of generate() turns (max_new=0 → prefill only) and return the
// last-position logits after the final turn.
static std::vector<float>
final_prefill_logits(const std::string &path, bool reuse,
                     const std::vector<std::string> &turns,
                     GenStats *last = nullptr) {
  auto src = open_model(path);
  LayerLoader::Options opt;
  opt.residency = Residency::FP32;
  opt.async = false;
  opt.n_buffers = 1;
  Runtime rt(std::move(src), opt, /*max_ctx=*/0, /*threads=*/1);
  rt.set_context_reuse(reuse);
  GenStats st;
  for (const std::string &t : turns)
    rt.generate(t, /*max_new=*/0, SamplerConfig{}, nullptr, &st);
  if (last)
    *last = st;
  return rt.first_logits();
}

static int tok_len(const std::string &path, const std::string &s) {
  auto src = open_model(path);
  Tokenizer tok = Tokenizer::from_source(*src);
  return (int)tok.encode(s, /*add_bos=*/true).size();
}

TEST(reuse_matches_full_prefill) {
  const std::string path = make_toy();
  const std::string p1 = "The capital of France";
  const std::string pfull = "The capital of France is Paris";

  // Full from-scratch prefill of the whole context (reuse OFF).
  std::vector<float> full = final_prefill_logits(path, false, {pfull});
  // Reuse path: commit the prefix, then reprocess only the tail.
  GenStats st;
  std::vector<float> reused =
      final_prefill_logits(path, true, {p1, pfull}, &st);

  CHECK(full.size() > 0);
  CHECK(full.size() == reused.size());
  for (size_t i = 0; i < full.size(); ++i)
    APPROX(reused[i], full[i], 1e-4); // reuse must equal full prefill

  CHECK(st.reuse_active);
  CHECK(st.reused_prefix_tokens == tok_len(path, p1)); // whole prefix reused
  CHECK(st.processed_tokens == tok_len(path, pfull) - tok_len(path, p1));
}

TEST(reuse_identical_reprompt_reprocesses_one) {
  const std::string path = make_toy();
  const std::string p = "hello world";
  GenStats st;
  final_prefill_logits(path, true, {p, p}, &st); // same context twice
  const int L = tok_len(path, p);
  // Identical reprompt: reuse everything except the last token (so we still
  // obtain last-position logits).
  CHECK(st.reused_prefix_tokens == L - 1);
  CHECK(st.processed_tokens == 1);
}

TEST(reuse_first_turn_reuses_nothing) {
  const std::string path = make_toy();
  GenStats st;
  final_prefill_logits(path, true, {"fresh start"}, &st);
  CHECK(st.reuse_active);
  CHECK(st.reused_prefix_tokens == 0); // nothing committed yet
}

int main() {
  printf("== test_reuse ==\n");
  return llmtest::run_all();
}
