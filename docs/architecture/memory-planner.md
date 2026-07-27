# Memory planner — the RAM-budget contract

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** `mem_plan.cpp` is a pure function that turns `--ram-budget N` from a hope into a *contract*: either `peak RSS ≤ N` is guaranteed for the whole run, or you get an itemized reason it cannot be. It feeds the byte ceiling consumed by [layer residency](layer-residency.html) and is applied by the [runtime](runtime.html).

## Problem

`--ram-budget 256M` used to lie. The old code derived a weight ceiling by subtracting only the KV cache plus a flat reserve, then allocated the LM head resident *regardless*, warned, and ran anyway — so a "256 MB" budget could still hit ~715 MB peak RSS. A budget that silently overshoots is worse than no budget.

The fix is to price **every** allocation that contributes to peak RSS, hold the sum at or below the budget, and refuse — with a breakdown — when even the minimum configuration does not fit.

## Design

`plan_memory(src, cfg, req)` (`src/mem_plan.cpp:37`) reads only the tensor directory and config. It performs **no I/O and mutates nothing**, so it is cheap and unit-testable. Three structs carry the contract:

- **`BudgetRequest`** — what the user asked for: `budget_bytes` (0 = unlimited, planner not consulted), `ctx_req` + `ctx_explicit`, `n_buffers_req`, `async_req`, `residency`, `stream_head_req`, and `force` (`--ram-budget-force`).
- **`MemLedger`** — an itemized byte ledger, every line a real peak-RSS contributor. `total()` is what the plan holds under budget.
- **`MemoryPlan`** — the decision: `feasible`, chosen `ctx` (+ `ctx_capped`), `stream_lm_head`, `n_buffers`, `weight_ceiling`, `n_pinned_est`, the final `ledger`, and a human-readable `report`.

### Every peak-RSS term is priced

| Ledger line | What it prices |
|:--|:--|
| `base` | ~24 MB process/allocator/tokenizer/thread-stack slop (calibrated) |
| `scratch` | fixed activation vectors (`dim*8 + ffn_dim*2 + q_dim + 2*kv_dim`) × fp32 |
| `logits` | `vocab × fp32` |
| `attn` | attention scores (`ctx`) |
| `resid` | prefill residual streams (`ctx × dim`), persistent |
| `out_norm` | final norm (+ GPT-2 learned position embeddings) |
| `embeddings` | input-embedding window (0 when head-resident & tied) |
| `lm_head` | output projection: resident table **or** streaming window |
| `kv` | KV cache priced at the **full chosen ctx** — worst case |
| `ring` | streaming ring (`n_buffers × one layer`) |
| `pinned` | hot layers pinned resident |

The KV line is the linchpin: even though the cache grows on demand, it is priced at the full chosen `ctx` (`src/mem_plan.cpp:133`) so the contract holds no matter how long the conversation runs.

### The search

The planner searches the knobs the user did **not** pin and picks the highest-scoring candidate that fits:

1. **ctx** — if `--ctx` was explicit it is honored or the plan is refused; otherwise it steps **down by 128** from the default cap to 128 (`src/mem_plan.cpp:99`), shrinking the KV reservation.
2. **stream_lm_head** — when the head is non-tied, try both resident and streamed; streaming drops the biggest fixed resident cost.
3. **n_buffers** — the requested ring depth and a single-buffer fallback.

For each combination it first tries to pin **all** layers, then tries streaming with `max_pinned = (budget − base_cost − ring) / layer_size` (`src/mem_plan.cpp:160`). Candidates are ranked (`Candidate::score`): higher ctx first, then resident-over-streamed head, then more pinned layers, then a 2-buffer ring. The winner's ledger becomes the plan.

### Feasible vs impossible

- **Feasible** → `weight_ceiling = budget − base − scratch − logits − attn − resid − kv` (`src/mem_plan.cpp:223`). That remainder is exactly what [layer residency](layer-residency.html) may spend on globals + ring + pinned layers. An `Execution Plan` report lists mandatory memory, estimated peak, ctx (capped?), KV, head mode, ring, and pinned count.
- **Impossible** → the minimum configuration (lowest ctx, streamed head, 1-buffer ring, 0 pinned) still exceeds the budget. The plan is marked infeasible and carries an itemized `RAM budget impossible` report — requested vs minimum-required, broken down by base/scratch/logits/KV/resid+attn/norm+embd/head/ring (`src/mem_plan.cpp:192`).

The [runtime](runtime.html) ctor prints the report and **throws** on an infeasible plan unless `--ram-budget-force` (`overridden`) is set, in which case it runs with safe minimum defaults (`src/runtime.cpp:52`).

## Measured

The plan's pinned-layer estimate tracks what the loader actually pins across the budget sweep (loader is the final authority via its actual-size guard):

| Model | Budget | Pinned | Decode tok/s | Streamed | Peak RSS |
|:--|--:|--:|--:|--:|--:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | 22.1 | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | 65.9 | 113 MB | 161 MB |

*(M3, warm, ctx 512.)* Peak RSS stays under each budget while pinned count and decode climb.

## Alternatives considered

| Approach | Why not |
|:--|:--|
| Subtract only KV + a flat reserve (the old way) | Ignored the resident head and ring — the budget could overshoot by hundreds of MB. |
| Measure real RSS then adjust | Non-deterministic, non-testable, and too late — the allocation already happened. |
| Fail hard on any over-budget request | `--ram-budget-force` lets an operator override with eyes open; the report still tells them the true cost. |
| Auto-tune the budget itself | Out of scope — the [auto-tuner](auto-tuning.html) chooses threads + scheduler, *not* the RAM budget. |

## Tradeoffs

- **Pessimistic KV pricing.** Charging the full chosen ctx up front can refuse a plan that would have fit a short conversation — but it is what makes "peak RSS ≤ budget" true for *any* run length.
- **Homogeneous-stack assumption.** Layer size comes from `blk.0.*`; a heterogeneous stack is corrected downstream by the loader's per-layer actual-size ceiling guard, not here.
- **Estimate, not authority, for pinning.** `n_pinned_est` guides; [layer residency](layer-residency.html) has the last word once real bytes are known.

## Source files

| File | Role |
|:--|:--|
| `include/llm/mem_plan.h` | `BudgetRequest`, `MemLedger`, `MemoryPlan`, `plan_memory`, `human_bytes` |
| `src/mem_plan.cpp` | ledger pricing, ctx/head/buffer search, feasible plan + impossible report |
| `src/runtime.cpp` | builds the `BudgetRequest`, prints the report, throws unless forced, applies `weight_ceiling`/`ctx`/`stream_lm_head`/`n_buffers` |
| `src/loader.cpp` | consumes `weight_ceiling` as `ram_budget_bytes` ([layer residency](layer-residency.html)) |

## Future work

- **Grow-on-demand KV pricing** — bill KV against a projected conversation length rather than full ctx, once a safe upper bound can be enforced at run time.
- **Heterogeneous per-layer sizing** in the planner so `n_pinned_est` matches exactly.
- **Feed the [auto-tuner](auto-tuning.html)** a suggested budget from device free-RAM detection.
