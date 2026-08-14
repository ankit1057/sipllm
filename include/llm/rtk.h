// rtk.h — RTK: the runtime-state (KV / memory) intelligence plugin seam.
//
// NOTE ON THE NAME: in this codebase "RTK" means the RUNTIME/KV intelligence
// layer (canonical roadmap #53) — KV cache, memory allocation, KV quantization,
// cache lifecycle, eviction, prefetch, offload, device placement, scheduling.
// It answers "how should runtime state be managed?" (Kosh answers "what context
// should we process?"). This is DISTINCT from the disabled, unrelated
// tool-calling code in src/rtk_tools.cpp.bak; if that is ever revived it must
// be renamed (tools.h / ToolRuntime) to avoid a name/ODR clash with this
// contract.
//
// RTK sits over the runtime state consumed by prefill/decode:
//
//     Kosh -> optimized context -> [ RTK: KV/memory policy ] -> Sip IR -> exec
//
// MESH-READINESS (locked design constraint): the contract is expressed over an
// abstract RtkKvView (dimensions + bytes), never the concrete KVCache, so a
// future distributed or persisted KV backing implements the SAME contract.
// Later capabilities — per-region precision policy (#39 Q8_0 KV), persist() /
// restore(), offload — extend this interface additively, never a rewrite.
//
// v0 (see src/rtk.cpp, make_rtk_v0) is an OBSERVER: it reports runtime-state
// metrics without touching the numerically-validated KV math.
#pragma once

#include "llm/model.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace llm {

// A read-only snapshot of the runtime KV state. Abstract over KVCache so the
// contract survives a future distributed/persisted backing.
struct RtkKvView {
  int64_t n_layers = 0;
  int64_t kv_dim = 0;
  int64_t seq_len = 0;  // filled positions
  int64_t capacity = 0; // currently-resident positions
  int64_t max_ctx = 0;
  size_t bytes = 0; // resident KV footprint
};

struct RtkMetrics {
  uint64_t steps = 0;       // decode steps observed
  int64_t max_seq = 0;      // longest sequence length seen
  size_t peak_kv_bytes = 0; // peak resident KV footprint
};

class RtkPlugin {
public:
  virtual ~RtkPlugin() = default;

  virtual const char *name() const = 0;

  // Lifecycle. Returning false disables the plugin (host logs, falls back).
  virtual bool init(const ModelConfig & /*cfg*/, RtkKvView /*initial*/) {
    return true;
  }
  virtual void shutdown() {}

  // Observation / policy hooks. Called with the current KV view.
  virtual void on_prefill(RtkKvView /*view*/) {}
  virtual void on_step(RtkKvView /*view*/) {}

  virtual RtkMetrics metrics() const { return {}; }
};

// v0 plugin: an observer that accumulates RtkMetrics. Establishes the
// runtime-state contract Nishachar/SipMesh depend on; opt-in (default OFF).
std::unique_ptr<RtkPlugin> make_rtk_v0();

} // namespace llm
