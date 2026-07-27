# Layer residency & the `--ram-budget` dial

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** This is the residency manager inside `LayerLoader` that turns a spare RAM allowance into decode speed. It sits directly on top of the [streaming loader](streaming-loader.html) and is fed a hard byte ceiling by the [memory planner](memory-planner.html).

## Problem

Pure streaming ([streaming loader](streaming-loader.html)) bounds peak weight RSS to a single layer, but it pays for that with I/O: every layer is re-read from disk on every token, so decode is disk-bound. Most devices, though, have *some* RAM to spare above one layer's worth. The question is how to spend that headroom to buy back throughput **without ever exceeding a promised peak-RSS ceiling** — and without changing a single output bit, so `--ram-budget` is a pure speed dial, not a quality knob.

The naive version — "keep the whole model resident if it fits" — is exactly what SipLLM exists to avoid on large models. What we want is a continuous middle: pin as many layers as fit under a budget, stream the rest.

## Design

`LayerLoader::Options::ram_budget_bytes` is a hard ceiling (in bytes) on *weight*-resident RAM: globals + pinned hot layers + the streaming ring. `0` means unlimited and is the legacy path, byte-for-byte unchanged.

When the ceiling is `> 0`, the constructor calls `plan_and_pin_layers()` (`src/loader.cpp:181`), which materializes the **leading contiguous run** `[0, n_pinned)` once and keeps it resident in `pinned_[l]`; the remaining cold layers stream through the ring as before.

### Sizing the pinned run

1. `estimate_layer_bytes(0)` (`src/loader.cpp:163`) predicts one layer's resident cost straight from the tensor directory — `numel*4` for fp32 roles, on-disk `nbytes` for quantized — matching `load_weight_into`'s buffer sizing exactly, so the estimate is precise for a homogeneous block stack (the real case).
2. If the budget covers every layer's weights plus globals, `target = n_layers` — pin everything, nothing streams, and the ring stays empty (decode goes compute-bound).
3. Otherwise reserve ring headroom (`n_buffers × per_layer`) for the cold-stream path, then pin `(budget − globals − ring) / per_layer` layers.

```text
weight-resident RSS  =  globals  +  pinned[0..n_pinned)  +  ring(n_buffers)
                        ├──────── must stay <= ram_budget_bytes ────────┤
layers:  [0 1 2 ... n_pinned-1]        [n_pinned ... n_layers-1]
         pinned resident (no I/O)      streamed through the ring
```

### The per-layer ceiling guard

The estimate could be wrong for a heterogeneous layer. So the pin loop verifies against the **actual materialized size**: after `fill_slot` loads layer `l`, it sums the real buffer bytes `sb` and checks `globals + pinned_bytes_ + sb + ring_reserve > budget` (`src/loader.cpp:209`). If pinning `l` would breach the ceiling, its buffers are dropped, the run stops, and `l` (and everything after it) streams instead. This makes the contract *peak weight RSS ≤ budget* hold on the actual bytes, not just the prediction.

### Pinned layers short-circuit I/O

`loadLayer(L)` checks `pinned_mask_[L]` first (`src/loader.cpp:330`): a pinned layer sets `active_` to `pinned_[L]`, clears `current_` (the ring is not the source), counts a **prefetch hit**, and returns with zero I/O. Cold layers fall through to the sync or async ring path unchanged, and `active_` is set on every path — which is exactly why `budget == 0` stays byte-identical: the pinned branch is simply never taken.

`resident_bytes()` (`src/loader.cpp:451`) reports globals + every ring slot buffer + `pinned_bytes_`, i.e. the live weight RSS the budget governs.

## Measured

The dial is smooth: more budget → more pinned layers → more prefetch hits → fewer bytes streamed per token → higher decode tok/s, at a higher (but still bounded) peak RSS.

| Model | Budget | Pinned | Decode tok/s | Streamed | Peak RSS |
|:--|--:|--:|--:|--:|--:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | 22.1 | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | 65.9 | 113 MB | 161 MB |

*(M3, warm cache, ctx 512; `bench_ram_budget.sh`.)* At `768M` every tinyllama layer is pinned: bytes streamed collapses from 14.4 GB to 576 MB and decode nearly doubles. Output is bit-identical across the whole sweep — `--ram-budget` never touches the math.

## Alternatives considered

| Approach | Why not the default |
|:--|:--|
| Pin the *hottest* layers (LRU/frequency) | A decoder touches every layer once per token, so there is no hotness signal within a forward pass; the leading contiguous run is simpler and equally effective. |
| Evict LRU from a fixed cache | Same RAM↔speed control but with cache-management complexity and eviction races; the contiguous-pin model needs no eviction. |
| Trust the byte estimate alone | A heterogeneous layer could overshoot the ceiling; the actual-size guard closes that gap without pessimism. |
| Let the OS decide via `mmap` | No hard ceiling — the kernel can fault the whole model in. See [prefetch](prefetch.html) and [Why not mmap?](why-not-mmap.html). |

## Tradeoffs

- **Pinning is leading-contiguous only.** Layers `[0, n_pinned)` are pinned; there is no hotness heuristic, so you cannot pin, say, only the attention-heavy blocks. For a uniform decoder this is optimal anyway.
- **`unloadLayer()` frees nothing** (`src/loader.cpp:385`). It marks the current ring slot reusable; buffers are recycled by *overwrite* on the next layer, not released. Peak RSS is therefore governed by the pin plan and ring depth, not by unload timing.
- **RAM for speed, monotonically.** More budget only ever costs more resident RAM and only ever buys more tok/s; correctness is fixed. The budget is a *total peak-RSS* target that the [memory planner](memory-planner.html) converts into this weight ceiling by subtracting the KV cache and scratch first.

## Source files

| File | Role |
|:--|:--|
| `include/llm/loader.h` | `Options::ram_budget_bytes`, `pinned_` / `pinned_mask_` / `n_pinned_` residency state |
| `src/loader.cpp` | `plan_and_pin_layers` (181), `estimate_layer_bytes` (163), the pinned short-circuit in `loadLayer` (330), `resident_bytes` (451), `unloadLayer` (385) |
| `include/llm/mem_plan.h` / `src/mem_plan.cpp` | derives the byte ceiling from a total peak-RSS budget ([memory planner](memory-planner.html)) |
| `src/runtime.cpp` | wires `plan.weight_ceiling` into `Options::ram_budget_bytes` ([runtime](runtime.html)) |

## Future work

- **Non-contiguous / hotness-aware pinning** — today the pinned set is always the leading run `[0, n_pinned)`.
- **Actually free unloaded buffers** in low-water conditions instead of relying on overwrite recycling.
- **Per-layer budget shaping** — spend the ceiling on the widest (most expensive to re-stream) layers first.
