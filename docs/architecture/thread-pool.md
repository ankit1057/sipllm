# Thread pool & scheduling

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** Every matmul in the [transformer](transformer.html) — fp32, [quantized](quantization.html), or SDOT — splits its output rows across this pool. The [auto-tuner](auto-tuning.html) picks how many workers and which schedule; this page is what it picks *between*.

## Problem

The hot path is matmul, and the parallel unit is the output row: worker `w` computes a disjoint slice of `y`'s rows. Spawning threads per matmul would let thread-creation dominate a per-token workload that fires dozens of small matmuls. And the right way to *split* those rows is not fixed: a big even LM-head projection wants contiguous no-overhead slices, while a lumpy quantized layer (rows dequantize at different costs) wants dynamic load balancing. One pool has to serve both without re-spawning.

## Design

`ThreadPool` (`include/llm/threadpool.h`) holds persistent workers and hands them `[begin, end)` row ranges through a `parallel_for` barrier. It spawns `n_ - 1` worker threads (the calling thread also works), defaulting to `hardware_concurrency` with headroom left so the OS and the prefetch thread stay responsive on a phone. On Apple platforms workers set `QOS_CLASS_USER_INTERACTIVE`.

`parallel_for(total, fn)` publishes the range and a per-policy **chunk size**, bumps a generation counter, and wakes the workers; each drains work and the caller waits on a barrier. The seven `SchedulePolicy` values differ *only* in that chunk size and whether draining is contiguous or work-stealing:

| Policy | chunk_ | Loop |
|:--|:--|:--|
| `Static` | `ceil(total / n_)` | contiguous — worker `tid` takes `[tid·chunk, …)`, **no stealing** |
| `Fixed8` | 8 | dynamic work-stealing |
| `Fixed16` | 16 | dynamic work-stealing |
| `Fixed32` | 32 | dynamic work-stealing |
| `Proportional2` | `max(1, total / (n_·2))` | dynamic work-stealing |
| `Proportional4` | `max(1, total / (n_·4))` | dynamic work-stealing |
| `Adaptive` | `max(16, total / (n_·4))` | dynamic work-stealing |

**Static** gives each worker one contiguous block and returns — lowest overhead, best when every row costs the same. **Every other policy** runs the dynamic loop: `next_row_.fetch_add(chunk_)` atomically claims the next chunk until the rows run out, so a worker that finishes cheap rows steals more instead of idling. Proportional sizes the chunk to the row count (finer for more workers); Adaptive is Proportional4 with a floor of 16 to avoid pathologically tiny chunks on small matmuls.

### Parallel-vs-serial thresholds

Small matmuls skip the pool entirely — barrier and wakeup cost more than the work:

- **fp32 `matmul`** parallelizes only when `pool->size() > 1 && n_out >= 64` (`src/ops.cpp`).
- **`matmul_quant`** and the **`matmul_q8_0_i8` SDOT** kernel use `n_out >= 32` (`src/quant.cpp`, `src/neon.cpp`) — quantized rows are heavier per row, so the crossover is lower.

Below the threshold the body runs inline on the calling thread.

### Defaults

- **CLI default: `Proportional2`** (`main.cpp`; the C FFI default too — `SIPLLM_SCHED_PROPORTIONAL2`).
- **Pool default: `Proportional4`** (`ThreadPool::policy` initializer) — the value a freshly constructed pool carries before anyone calls `set_policy`.

## Alternatives considered

| Approach | Why not the default |
|:--|:--|
| Spawn threads per matmul | Thread creation dominates a per-token workload of many small matmuls. |
| Pure static partition only | Idles workers on lumpy quantized layers where rows cost unequally. |
| Pure work-stealing only | The atomic `fetch_add` per chunk is pure overhead when rows are uniform and large — `Static` avoids it. |
| Always parallelize | Barrier/wakeup cost exceeds the work below ~32–64 rows; hence the thresholds. |

## Tradeoffs

- **Policy only sets chunk size + steal-vs-contiguous.** It does not change the math or the result — `tests/test_ops.cpp` checks every policy produces the same output as the serial path.
- **`steals` is a proxy.** The `Stats` counter increments on any chunk beginning past row 0, not on a true steal event — a coarse signal, not an exact steal count.
- **Only 4 of the 7 policies are auto-benchmarked.** The [auto-tuner](auto-tuning.html) sweeps `Static, Fixed16, Proportional2, Adaptive` (using `Fixed16` as the baseline while it finds the best thread count), then caches the winner. `Fixed8`, `Fixed32`, and `Proportional4` are selectable by hand but never chosen automatically.
- **Thread count is capped at 16** by the auto-tuner, with no big.LITTLE / core-cluster detection — it treats cores as uniform. The tuner picks threads + policy only; it does **not** choose the RAM budget (that is the separate memory planner).

## Source files

| File | Role |
|:--|:--|
| `include/llm/threadpool.h` | `ThreadPool`, `SchedulePolicy`, `parallel_for`, `run_chunk` (static vs work-stealing), `Stats`, `default_pool()` |
| `src/ops.cpp` | fp32 `matmul` — `n_out >= 64` parallel threshold |
| `src/quant.cpp` | `matmul_quant` — `n_out >= 32` threshold |
| `src/neon.cpp` | `matmul_q8_0_i8` SDOT — `n_out >= 32` threshold |
| `src/auto_tuner.cpp` | the 4-policy + thread-count micro-benchmark that selects the default |

## Future work

- **Auto-benchmark all 7 policies** (and expose `Proportional4`/`Fixed32` to the tuner) rather than a fixed 4-way sweep.
- **big.LITTLE awareness** — detect performance vs efficiency cores and bias chunking/affinity instead of assuming uniform cores.
- **True steal accounting** to replace the row-0 proxy, for honest scheduler telemetry.
