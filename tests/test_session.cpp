// test_session.cpp — persistent context: prove save/restore ≡ in-process reuse.
//
// Phase-5 persistence lets a Runtime serialize its committed tokens + resident
// KV cache to disk (session v1, "SIPS") and restore them into a fresh Runtime
// in a new process. Correctness gate: after load_session + reuse, the
// last-position logits of a longer prompt sharing the persisted prefix must
// equal those of a from-scratch full prefill (bitwise-close), and the reuse
// bookkeeping must show the persisted prefix was reused rather than
// reprocessed. Uses a byte-tokenizer .llmw toy model (deterministic 1
// token/byte), single-threaded for reproducibility.
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
  std::string path = llmtest::scratch_path("toy_session.llmw");
  write_toy_model(path, tc);
  return path;
}

// Standard single-threaded test loader options (matches test_reuse).
static LayerLoader::Options test_opts() {
  LayerLoader::Options opt;
  opt.residency = Residency::FP32;
  opt.async = false;
  opt.n_buffers = 1;
  return opt;
}

// Run a sequence of generate() turns (max_new=0 → prefill only) on a fresh
// Runtime and return the last-position logits after the final turn.
static std::vector<float>
final_prefill_logits(const std::string &path, bool reuse,
                     const std::vector<std::string> &turns,
                     GenStats *last = nullptr) {
  auto src = open_model(path);
  Runtime rt(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
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

TEST(session_restore_matches_in_process) {
  const std::string path = make_toy();
  const std::string sess = llmtest::scratch_path("sess.bin");
  const std::string p1 = "The capital of France";
  const std::string pfull = "The capital of France is Paris";

  // Reference: full from-scratch prefill of the whole context (reuse OFF).
  std::vector<float> ref = final_prefill_logits(path, false, {pfull});

  // Persist: commit the prefix in one Runtime, then save the session.
  {
    auto src = open_model(path);
    Runtime rtA(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
    rtA.generate(p1, /*max_new=*/0, SamplerConfig{}, nullptr, nullptr);
    CHECK(rtA.save_session(sess));
  }

  // Restore: load the session into a fresh Runtime, then reprocess only the
  // changed tail via the reuse path.
  auto src = open_model(path);
  Runtime rtB(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
  CHECK(rtB.load_session(sess));
  rtB.set_context_reuse(true);
  GenStats st;
  rtB.generate(pfull, /*max_new=*/0, SamplerConfig{}, nullptr, &st);
  std::vector<float> b = rtB.first_logits();

  CHECK(ref.size() > 0);
  CHECK(ref.size() == b.size());
  for (size_t i = 0; i < ref.size(); ++i)
    APPROX(b[i], ref[i], 1e-4); // restore + reuse must equal full prefill

  CHECK(st.reused_prefix_tokens == tok_len(path, p1)); // whole prefix reused
}

TEST(session_restore_reuses_all_but_one_on_identical) {
  const std::string path = make_toy();
  const std::string sess = llmtest::scratch_path("sess.bin");
  const std::string p1 = "The capital of France";

  // Persist the committed prefix.
  {
    auto src = open_model(path);
    Runtime rtA(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
    rtA.generate(p1, /*max_new=*/0, SamplerConfig{}, nullptr, nullptr);
    CHECK(rtA.save_session(sess));
  }

  // Restore, then reprompt with the identical context: reuse everything except
  // the last token (so we still obtain last-position logits).
  auto src = open_model(path);
  Runtime rtB(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
  CHECK(rtB.load_session(sess));
  rtB.set_context_reuse(true);
  GenStats st;
  rtB.generate(p1, /*max_new=*/0, SamplerConfig{}, nullptr, &st);

  const int L = tok_len(path, p1);
  CHECK(st.reused_prefix_tokens == L - 1);
  CHECK(st.processed_tokens == 1);
}

TEST(session_load_missing_file_fails) {
  const std::string path = make_toy();
  auto src = open_model(path);
  Runtime rt(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
  CHECK(rt.load_session(llmtest::scratch_path("does_not_exist.bin")) == false);
}

TEST(session_load_wrong_model_refused_by_model_id) {
  const std::string path = make_toy(); // model A: ffn_dim 64
  const std::string sess = llmtest::scratch_path("sess_wrong.bin");
  {
    auto src = open_model(path);
    Runtime rtA(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
    rtA.generate("hello", /*max_new=*/0, SamplerConfig{}, nullptr, nullptr);
    CHECK(rtA.save_session(sess));
  }
  // Model B shares n_layers (3) AND kv_dim (n_kv_heads*head_dim = 2*8 = 16)
  // with A, so session_read's dims check would PASS — only ffn_dim differs,
  // which feeds model_id but not the dims check. Thus a refusal isolates the
  // model_id guard (loading a same-shape-but-different model must not silently
  // corrupt).
  ToyConfig tc;
  tc.n_layers = 3;
  tc.dim = 32;
  tc.n_heads = 4;
  tc.n_kv_heads = 2;
  tc.ffn_dim = 128; // the only difference vs make_toy()
  tc.vocab_size = 256;
  tc.ctx_len = 256;
  tc.seed = 5;
  const std::string other = llmtest::scratch_path("toy_session_other.llmw");
  write_toy_model(other, tc);

  auto src = open_model(other);
  Runtime rtB(std::move(src), test_opts(), /*max_ctx=*/0, /*threads=*/1);
  CHECK(rtB.load_session(sess) == false); // refused by model_id, not by dims
}

int main() {
  printf("== test_session ==\n");
  return llmtest::run_all();
}
