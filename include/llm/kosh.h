// kosh.h — Kosh: the context / token-intelligence plugin seam.
//
// Kosh is NOT a bot. It is the token/context optimization layer: it decides
// *what context should reach the model at all* and how it is represented. It
// sits on the request path, immediately after tokenization and before prefill:
//
//     raw prompt -> tokenize -> [ Kosh::optimize ] -> prefill -> decode
//
// Guiding principle: the cheapest token is the one you never process.
//
// MESH-READINESS (locked design constraint): KoshRequest / KoshResult are the
// serialization unit for a future mesh Kosh. A remote node receives a
// KoshRequest (tokens + accounting), returns a KoshResult — the same public
// contract whether Kosh runs in-process or on another device. Do not add
// runtime-only handles to these structs; keep them serializable.
//
// This is the plugin *contract*. The v0 implementation (see src/kosh.cpp,
// make_kosh_v0) is deliberately small — a lossless token-run collapse with full
// accounting — so the boundary is proven while the algorithm evolves
// internally.
#pragma once

#include "llm/model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

// A request as it enters inference. Same struct for a local or a mesh request.
struct KoshRequest {
  const ModelConfig *cfg =
      nullptr; // target model (vocab/ctx); local convenience, not serialized
  std::string prompt;          // raw text (may be empty when tokens are preset)
  std::vector<int64_t> tokens; // tokenized context (post-tokenize)
  int64_t start_pos = 0;       // absolute KV position this context begins at
  bool add_bos = false;
};

// The optimized context to actually process, plus accounting.
struct KoshResult {
  std::vector<int64_t> tokens; // stream the runtime will prefill
  int64_t tokens_in = 0;       // count before optimization
  int64_t tokens_out = 0;      // count actually processed
  bool modified = false;       // did Kosh change the stream?
};

// Cumulative metrics for observability (Phase 3 exit criterion is measured
// here).
struct KoshMetrics {
  uint64_t requests = 0;
  uint64_t tokens_in = 0;
  uint64_t tokens_out = 0;
  uint64_t reductions = 0; // requests that actually shrank the stream
};

class KoshPlugin {
public:
  virtual ~KoshPlugin() = default;

  virtual const char *name() const = 0;

  // Lifecycle. init() may inspect the model; returning false disables the
  // plugin (the host logs and falls back to identity).
  virtual bool init(const ModelConfig & /*cfg*/) { return true; }
  virtual void shutdown() {}

  // The interception point. Implementations SHOULD be exception-safe and, on
  // any doubt, return the input unchanged. The host also guards this call.
  virtual KoshResult optimize(const KoshRequest &req) = 0;

  virtual KoshMetrics metrics() const { return {}; }
};

// v0 plugin: collapse any run of the same token id longer than `max_run` down
// to `max_run` copies (e.g. long runs of newline/space/padding tokens).
// Lossless to reason about, deterministic, and opt-in — default generation
// never invokes it.
std::unique_ptr<KoshPlugin> make_kosh_v0(int max_run = 8);

// v1 plugin: generalizes v0 to collapse consecutive repeated *blocks* (repeated
// phrases / boilerplate that tokenize to multi-token spans), not just runs of a
// single token. At each position it takes the smallest block length b in
// [1, max_block_len] whose block repeats immediately, collapses that block's
// maximal consecutive run to `keep` copies, and advances. Deterministic,
// greedy (smallest-block-first), opt-in. `keep=1, max_block_len=4` by default.
std::unique_ptr<KoshPlugin> make_kosh_v1(int keep = 1, int max_block_len = 4);

} // namespace llm
