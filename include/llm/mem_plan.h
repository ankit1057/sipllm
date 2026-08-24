// mem_plan.h — the RAM-budget *contract* planner (issue: budget must not lie).
//
// The problem it solves: `--ram-budget 256M` used to still hit ~715 MB peak RSS
// because the resident LM head and the streaming ring were never counted against
// the budget — the old code derived a weight ceiling by subtracting only the KV
// cache + a flat reserve, then allocated the head resident regardless and warned
// but ran anyway. This planner turns the budget into a contract:
//
//   peak RSS <= budget   — GUARANTEED for the whole run up to the chosen ctx,
//
// or an explicit, itemized *impossible* report with concrete suggestions. It
// does this by pricing EVERY allocation that contributes to peak RSS, then
// auto-fitting the knobs the user did not pin — in priority order:
//
//   1. stream the LM head (drop the biggest fixed resident cost)
//   2. cap the context window (shrinks the KV reservation) — only when --ctx was
//      not given explicitly; an explicit --ctx is honored or the plan is refused
//   3. reduce the streaming ring to a single buffer
//   4. spend whatever budget remains pinning hot layers resident (speed)
//
// The KV cache is priced at the FULL chosen ctx (worst case) even though it grows
// on demand, so the contract holds no matter how long the conversation runs.
//
// The planner is a pure function of (source directory, config, request): it reads
// tensor sizes from the directory but performs no I/O and mutates nothing, so it
// is cheap and unit-testable. Runtime applies the result (or throws the report).
#pragma once

#include "llm/loader.h"   // Residency
#include "llm/model.h"
#include "llm/weight_source.h"

#include <string>

#include "llm/kv_cache.h" // For KVPrecision

namespace llm {

// What the user asked for. budget_bytes == 0 means "unlimited" — the planner is
// not consulted and the legacy path runs byte-for-byte unchanged.
struct BudgetRequest {
    size_t      budget_bytes    = 0;      // total peak-RSS target (0 = unlimited)
    int         ctx_req         = 0;      // requested --ctx (0 => model/default)
    bool        ctx_explicit    = false;  // true iff the user passed --ctx
    int         default_ctx_cap = 4096;   // edge-friendly clamp for a defaulted ctx
    int         n_buffers_req   = 2;      // requested streaming ring depth
    bool        async_req       = true;   // false => --no-async (ring pinned to 1)
    Residency   residency       = Residency::Quantized;
    bool        stream_head_req = false;  // user forced --stream-lm-head
    bool        force           = false;  // --ram-budget-force: run even if impossible
    KVPrecision kv_precision    = KVPrecision::FP32;
};

// Itemized ledger (bytes). Every line is a real contributor to peak RSS; the sum
// is what the planner holds at or below the budget. Exposed for observability.
struct MemLedger {
    size_t base        = 0;   // process/allocator/tokenizer/thread-stack slop (calibrated)
    size_t scratch     = 0;   // fixed activation vectors (dim/ffn/qkv sized)
    size_t logits      = 0;   // vocab * fp32
    size_t attn        = 0;   // attention scores (ctx)
    size_t resid       = 0;   // prefill residual streams (ctx * dim), persistent
    size_t out_norm    = 0;   // final norm (+ GPT-2 learned position embeddings)
    size_t embeddings  = 0;   // input-embedding streaming window (0 if head-resident & tied)
    size_t lm_head     = 0;   // output projection: resident table or streaming window
    size_t kv          = 0;   // KV cache priced at the full chosen ctx (worst case)
    size_t ring        = 0;   // streaming ring (n_buffers * one layer)
    size_t pinned      = 0;   // hot layers pinned resident
    size_t total() const {
        return base + scratch + logits + attn + resid + out_norm + embeddings +
               lm_head + kv + ring + pinned;
    }
};

struct MemoryPlan {
    bool        feasible       = false; // fits the budget (always true if budget==0)
    bool        overridden     = false; // infeasible, but --ram-budget-force set → run anyway
    int         ctx            = 0;     // chosen (possibly capped) context window
    bool        ctx_capped     = false; // ctx reduced below request/default to fit
    bool        stream_lm_head = false; // planner's head decision (may enable it)
    int         n_buffers      = 2;     // chosen streaming ring depth
    size_t      weight_ceiling = 0;     // → LayerLoader::Options.ram_budget_bytes
    int         n_pinned_est   = 0;     // estimated pinned layers (loader is authority)
    int         n_layers       = 0;
    MemLedger   ledger;
    std::string report;                 // human-readable summary or impossible reason
};

// Compute the plan. Reads only the tensor directory + config; no I/O, no mutation.
MemoryPlan plan_memory(const WeightSource& src, const ModelConfig& cfg,
                       const BudgetRequest& req);

// Human-readable byte size, e.g. 268435456 -> "256 MB". Base-10 MB/GB to match
// how users think of "--ram-budget 256M" (= 256 MiB shown as 268 MB elsewhere is
// confusing; we print MiB-derived values consistently as MB with one decimal).
std::string human_bytes(size_t bytes);

} // namespace llm
