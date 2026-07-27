# Predictive prefetch

**Streaming made the model fit, but it also made the compute thread wait on disk.** With the synchronous single-buffer loader, the CPU runs block `L`'s attention and FFN, *then* blocks on the `pread` + dequant of block `L+1`, then computes, then waits again. Storage and arithmetic run in strict alternation, so the run is at best as slow as I/O plus compute added together. But reading and dequantizing block `L+1` needs nothing from block `L`'s computation — the two are independent. They should overlap.

## The problem: serial I/O behind serial compute

The forward pass is a chain: `loadLayer(L) → block(L) → loadLayer(L+1) → block(L+1) → …`. In the single-buffer path each `loadLayer` is a synchronous stall. On a streaming runtime, where every layer is an actual storage read, that stall is the dominant cost per layer and it is entirely sequential with the compute it precedes.

## The fix: an async double-buffer ring

`LayerLoader` runs an optional background worker over a small ring of buffers (`Options::n_buffers`, default 2). While the compute thread runs block `L`, the worker materializes block `L+1` into the *other* buffer through the three-stage pipeline **Storage (`pread`) → Dequant → ready buffer** (`src/loader.cpp`, `worker_loop`):

```text
    compute thread                    prefetch worker (background)
    --------------                    ----------------------------
loadLayer(L)  ------ enqueue L+1 --->  pread + dequant block L+1
block(L) runs attn + FFN               stage into the other buffer
loadLayer(L+1) -- already Ready? -->   HIT  (no wait)
                  not finished? --->   MISS (block until ready)
```

When compute finishes and calls `loadLayer(L+1)`, the block is usually already `Ready` — a **prefetch hit**, so the compute thread never touches I/O. If the worker hasn't finished, it's a **miss** and `loadLayer` blocks until it does. The `Stats` counters `prefetch_hits` / `prefetch_misses` (`include/llm/loader.h`) expose exactly how well the pipeline keeps up; the profiler (`build/bench`) prints per-layer I/O, dequant, compute, and RSS so you can see the overlap.

> [!KEY] Three backends, one truth
> There are three interchangeable backends: synchronous single-buffer `pread` (`--no-async`, strictly one block resident), the async double-buffer ring (default), and `mmap` (`--mmap`). All three produce **identical output**, because the math only ever sees a `WeightSource` — the [founding seam](j-streaming-layers.html). You pick the backend for the storage profile, never for correctness.

> [!NOTE] What mmap actually buys
> The `mmap` backend is *not* zero-copy: `read_raw_at` still memcpys from the mapped pages into a resident buffer. The win is OS page-cache buffering, not eliminated copies. (`FileBacking::prefetch()` for explicit `fadvise`/`madvise` readahead exists but is currently unused — a future deepening.)

## Measured: overlap, not new arithmetic

Prefetch changes *scheduling*, not results, so it shows up in the hit/miss counters and the [RAM-budget throughput sweep](j-bigger-than-ram.html) rather than in a before/after logit diff. When the ring keeps up, per-token latency approaches `max(I/O, compute)` instead of their sum; when storage is too slow, misses rise and the run degrades gracefully back toward the synchronous bound. Correctness is invariant across all three backends — the numeric oracle is unchanged.

## What it unlocked

Prefetch is why streaming isn't automatically slow: on a storage tier that can keep pace, the disk cost of the [bigger-than-RAM](j-bigger-than-ram.html) runs hides behind the arithmetic it feeds. Deeper pipelining and explicit readahead are the next steps. The ring, its buffers, and the worker are documented on the [predictive prefetch](prefetch.html) architecture page.
