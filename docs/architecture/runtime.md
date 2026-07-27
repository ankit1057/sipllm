# Runtime — end-to-end generation

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** `Runtime` owns the whole stack and drives one prompt to a stream of tokens. It is where the [memory planner](memory-planner.html), the [streaming loader](streaming-loader.html), the [KV cache](kv-cache.html), and the [transformer](transformer.html) are assembled and run.

## Problem

The layers below are deliberately narrow: the loader only knows about weights, the transformer only about a forward pass, the planner only about bytes. Something has to open a model file of unknown format, turn a byte budget into concrete loader options, wire the stack together in the right order, and run the prefill→decode loop while collecting the throughput/latency numbers the UI shows. That is `Runtime`.

## Design

### Opening a model

`open_model(path, use_mmap)` (`src/runtime.cpp:13`) sniffs the first four bytes: `kGGUFMagic` → `GgufFile` (the real GGUF parser), `kLLMWMagic` → `ModelFile` (the toy `.llmw` format), anything else throws. Both implement `WeightSource`, so nothing above cares which was opened.

### Construction — plan, then build

The constructor (`src/runtime.cpp:32`) runs a fixed sequence:

1. `ModelConfig::from_source` reads hyperparameters; `set_fast_quant` opts into the int8 SDOT kernel when `--fast` is set.
2. Build a `BudgetRequest` from the options + `ram_budget_total` and call `plan_memory` ([memory planner](memory-planner.html)).
3. If a budget was given, print `plan.report`; **throw** on an infeasible plan unless `--ram-budget-force` was passed (`src/runtime.cpp:52`).
4. Apply the plan: `ctx = plan.ctx`, `opt_.stream_lm_head`, `opt_.n_buffers`, and `opt_.ram_budget_bytes = plan.weight_ceiling`.
5. Build the stack in dependency order: `ThreadPool` → `LayerLoader` → `KVCache(n_layers, kv_dim, ctx)` → `Transformer` → `Tokenizer`.

```text
open_model(path)            magic-sniff GGUF vs .llmw -> WeightSource
  └─ plan_memory(req)       budget -> ctx / head / buffers / weight_ceiling
       └─ ThreadPool        auto-tuned threads (see auto-tuning)
            └─ LayerLoader   residency + prefetch, capped at weight_ceiling
                 └─ KVCache  sized to the chosen ctx
                      └─ Transformer  forward pass over the loader + KV
```

### Generation — one prefill, then a decode loop

`generate()` (`src/runtime.cpp:72`) encodes the prompt (adding BOS only on a fresh sequence — the engine applies **no chat template**; the caller formats conversations), then:

- **Prefill (single pass).** `tf_->prefill(ids, n, pos_)` streams the whole model **once** for the entire prompt (RFC-007), instead of the old per-token loop that re-streamed every layer for every prompt token. It advances `pos_`, seeds the sampler's repetition history, records `first_logits_`, and samples the first token. `ttft_s = prefill_s`.
- **Decode loop.** For each new token: stop on EOG or ctx exhaustion, decode the piece, invoke the `on_token` callback (return `false` to stop early), then `tf_->forward(next, pos_)` for the next logits and sample. An optional `ProfileSink` reports per-layer timings + peak RSS per step.

Finally it fills `GenStats` — `prompt/gen_tokens`, `ttft_s`, `prefill_tok_s`, `decode_tok_s`, `weights_resident_bytes`, `pinned_layers`, `kv_bytes`, `bytes_read`, `prefetch_hits`/`misses`, `ctx_used`/`ctx_max`.

## Measured

Single-pass prefill is why time-to-first-token is bounded even for long prompts — the model is streamed once, not once per prompt token:

| Workload | Metric | Before | After |
|:--|:--|--:|--:|
| smollm2, 122-tok prompt | streamed | 14,456 MB | 790 MB (18.3×) |
| smollm2, 122-tok prompt | TTFT | 3.44 s | 1.96 s |
| tinyllama, 50-tok prompt | streamed | 31,128 MB | 2,882 MB (10.8×) |

The KV cache the runtime sizes to `ctx` grows on demand, so peak RSS tracks real conversation length, not the reservation:

| Model | Peak RSS before | after | KV before | KV after |
|:--|--:|--:|--:|--:|
| SmolLM2-135M | 239 MB | 53.5 MB | 188.7 MB | 2.9 MB |
| TinyLlama-1.1B Q4_K_M | 210.6 MB | 121 MB | 92.3 MB | 2.9 MB |

*(M3, warm, t=4.)*

## Alternatives considered

| Approach | Why not |
|:--|:--|
| Per-token prefill loop (the old way) | Re-streamed the whole model for every prompt token — TTFT and bytes-streamed scaled with prompt length. Single-pass prefill collapses it to one stream. |
| Apply a built-in chat template | The engine stays raw-text; conversation formatting is the caller's job ([Flutter runtime](flutter-runtime.html) does chatml/llama3/zephyr/raw). Keeps the runtime format-agnostic. |
| Reuse one runtime for chat + embeddings | `embed()` clears KV state — use a dedicated runtime for embeddings, never one mid-conversation. |
| Let the caller wire the stack | Order and budget-derivation are load-bearing; centralizing them in the ctor prevents mis-assembly. |

## Tradeoffs

- **`runtime_reserve_bytes` is dead code** (`src/runtime.cpp:26`). The weight ceiling now comes entirely from `plan.weight_ceiling`; this older reserve helper is no longer on any path.
- **Budget failure is loud by default.** An infeasible plan throws; `--ram-budget-force` is the deliberate override, and it runs with the planner's safe minimum defaults.
- **Stats are collected once, at the end.** `GenStats` reflects the completed run; live per-step data comes through the optional `ProfileSink`, not the return value. The Dart `SipllmStats` mirror is a **subset** — it drops the per-phase `load_s`/`prefill_s`/`decode_s`.

## Source files

| File | Role |
|:--|:--|
| `include/llm/runtime.h` | `Runtime`, `GenStats`, `open_model`, `TokenCallback`/`ProfileSink` hooks |
| `src/runtime.cpp` | `open_model` magic-sniff (13), ctor plan+build (32), `generate` prefill+decode (72), dead `runtime_reserve_bytes` (26) |
| `include/llm/mem_plan.h` | the plan the ctor applies ([memory planner](memory-planner.html)) |
| `include/llm/loader.h` | loader options derived from the plan ([streaming loader](streaming-loader.html), [layer residency](layer-residency.html)) |
| `include/llm/kv_cache.h` / `include/llm/transformer.h` | KV + forward pass the runtime drives ([KV cache](kv-cache.html), [transformer](transformer.html)) |

## Future work

- **Remove the dead `runtime_reserve_bytes`** helper now that the planner owns the ceiling.
- **Incremental stats streaming** so the UI can read throughput mid-run without the `ProfileSink` plumbing.
- **A first-class embeddings runtime** so `embed()` need not share (and clear) chat KV state.
