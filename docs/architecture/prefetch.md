# Predictive prefetch

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** The async double-buffered ring hides I/O latency behind compute so a *streamed* layer feels almost as cheap as a resident one. It is one of the three interchangeable backends inside the [streaming loader](streaming-loader.html); all three produce bit-identical output.

## Problem

Streaming reads each layer from disk on the critical path: `loadLayer(L)` blocks on `pread` + dequant before block `L` can run. Serially, decode is `Σ (I/O_L + compute_L)`. But the sequence is perfectly predictable — after block `L` comes block `L+1`, always — so the I/O for `L+1` can happen *while* the compute thread is busy with `L`. If the pipeline keeps up, the compute thread never waits on storage and per-token time drops toward `Σ compute_L`.

## Design

`LayerLoader` runs an optional background worker plus a small ring of slot buffers (`Options::n_buffers`, default 2). The worker is spawned only when it can help — `async && n_buffers > 1 && n_pinned_ < n_layers_` (`src/loader.cpp:154`).

```text
      compute thread                     prefetch worker (background)
      --------------                     ----------------------------
loadLayer(L)  ---- enqueue(L+1) -------> worker_loop: pread block L+1
block(L): attn + FFN                     dequant / stage into the other slot
loadLayer(L+1) -- slot Ready? ---------> HIT  (no wait)  -> prefetch_hits++
                 not finished? --------> MISS (block on cv_ready_) -> prefetch_misses++
```

### The worker loop

`worker_loop()` (`src/loader.cpp:295`) waits on `cv_job_` for a `Job{slot, layer}`, marks the slot `Loading` under `mutex_`, then runs `fill_slot` (I/O + dequant) **unlocked** so it overlaps compute. When done it flips the slot to `Ready` under the lock and notifies `cv_ready_`. `enqueue()` (`src/loader.cpp:317`) is called with `mutex_` held and de-dups jobs already queued or in flight.

### One `loadLayer` step

`loadLayer(L)` (`src/loader.cpp:326`) in the async path:

1. If some slot already holds `L` and is `Ready` → **hit**; if it is still `Loading` → **miss**, wait on `cv_ready_`.
2. If no slot holds `L` → **miss**: pick a victim slot (not `current_`), push the job to the *front* of the queue (priority), and wait until that slot is `Ready` (`src/loader.cpp:371`).
3. On becoming ready, set `current_`/`active_`, then **kick off prefetch of `L+1`** into the other slot (`src/loader.cpp:378`) so it materializes during the next compute.

The `Stats` counters `prefetch_hits` / `prefetch_misses` surface how well the pipeline keeps up; they flow out through [`GenStats`](runtime.html) for the profiler.

## Three interchangeable backends

The math only ever sees a `WeightSource`, so residency strategy is swappable with **no change to output**:

| Backend | Flag | Resident weight buffers | Notes |
|:--|:--|:--|:--|
| Synchronous single-buffer `pread` | `--no-async` (`n_buffers = 1`) | exactly one layer | strict minimum RAM, fully serial I/O + compute |
| Async double-buffered ring | default (`n_buffers = 2`) | up to two layers | worker overlaps `L+1` load with `L` compute |
| `mmap` | `--mmap` | one layer (copied from mapped pages) | OS page cache does the buffering |

All three are benchmarked side by side and yield identical logits (guarded by the e2e suite). Choosing among them is a memory/throughput decision, never a correctness one.

## Alternatives considered

| Approach | Why not the default |
|:--|:--|
| Deeper pipeline (`n_buffers > 2`) | Marginal for a strictly sequential access pattern; each extra buffer costs a full layer of RAM. Kept as a knob, not a default. |
| `posix_fadvise`/`madvise(WILLNEED)` readahead | `FileBacking::prefetch()` exists but is **unused** — the explicit worker already overlaps I/O, and hint-based readahead gave no measured win. |
| Let `mmap` + the kernel prefetch everything | No hard RSS ceiling; the OS may fault the whole model in. mmap is offered as a *backend*, paired with the [memory planner](memory-planner.html)'s budget. |
| Speculative multi-layer prefetch | Only helps in the exceeds-RAM regime; noted as future throughput work. |

## Tradeoffs

- **The async ring costs a second layer buffer.** That is the price of overlap; the single-buffer path exists for the strict-minimum-RAM case, and the [memory planner](memory-planner.html) drops the ring to 1 buffer when the budget is tight.
- **`mmap` is not zero-copy.** `read_raw_at` **memcpys** from the mapped pages into a resident buffer (`include/llm/file_backing.h`); the win is OS page-cache buffering, not a copy elided. See [Why not mmap?](why-not-mmap.html).
- **`FileBacking::prefetch()` (fadvise/madvise) is dead code** — present in `file_backing.h` but never called. The overlap comes entirely from the worker thread, not kernel hints.
- **Misses are self-correcting.** A miss pushes the needed layer to the front of the queue and blocks only until it is ready; the pipeline resynchronizes on the next layer.

## Source files

| File | Role |
|:--|:--|
| `include/llm/loader.h` | ring/worker state: `slots_`, `Job`, `cv_job_`/`cv_ready_`, `Stats` hit/miss counters, `Options::n_buffers`/`async`/`use_mmap` |
| `src/loader.cpp` | `worker_loop` (295), `enqueue` (317), `loadLayer` hit/miss + next-layer prefetch (326), the spawn guard (154) |
| `include/llm/file_backing.h` | `pread_exact`, optional `mmap` view, and the unused `prefetch()` hint |
| `src/transformer.cpp` | the forward pass whose per-layer compute the worker overlaps |

## Future work

- **Deeper prefetch pipelining** and wiring up `posix_fadvise`/`madvise(WILLNEED)` (`FileBacking::prefetch()` is ready but unused).
- **Speculative streaming** to amortize weight movement in the exceeds-RAM regime — the long-term throughput moat.
- **True zero-copy mmap** so quantized weights are dotted straight from mapped pages without the intermediate memcpy.
