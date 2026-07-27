# Streaming layer loader

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Diagrams → Future.** The loader is the seam that makes the [memory-bounded thesis](../journal/streaming-layers.md) true. For the reference of every function named here, see the [C++ engine API](api-cpp.html).

## Problem

An LLM's weights dwarf its activations. Loading them all resident makes peak RSS scale with *model size* — 1.1&nbsp;GB for a 1.1&nbsp;B Q8 model, tens of GB for a 70&nbsp;B one, which simply does not fit on a phone. But a decoder only touches **one transformer block at a time**: block `L`'s weights are needed only while block `L` runs. If we could make exactly one block resident and reuse that memory for the next, peak weight RSS would track *layer* size (model width), not *model* size (depth). The loader is what makes that reuse safe, fast, and invisible to the math.

## Design

The transformer never opens a file. It talks to a **`WeightSource`** (`include/llm/weight_source.h`) that exposes three things: a tensor directory (`tensors()`, `find()`), typed metadata (`meta_int/meta_float/meta_str`), and positional reads (`read_raw()` / `read_raw_at()`, a `pread` that never loads the whole file). Both the real GGUF parser (`GgufFile`) and the toy `.llmw` reader (`ModelFile`) implement it, so **swapping loaders or residency strategy touches no line of math.**

`LayerLoader` (`include/llm/loader.h`, `src/loader.cpp`) drives blocks through that seam. Each block asks for its weights by **`Role`** (28 roles; the core nine are `AttnNorm, AttnQ, AttnK, AttnV, AttnOut, FfnNorm, FfnGate, FfnUp, FfnDown`), which the loader maps to GGUF tensor names like `blk.<L>.attn_q.weight`. The forward pass loops:

```text
embed_token(token)              -> x   (one row streamed, or resident if tied)
for layer L in 0 .. n_layers-1:
    loadLayer(L)                -> make block L's weights resident (may block on prefetch)
    block(L, pos)               -> RMSNorm -> QKV -> RoPE -> GQA attn -> proj -> RMSNorm -> SwiGLU
    unloadLayer()               -> release block L for reuse
final RMSNorm + project_output  -> logits
```

Two mechanisms keep only a layer's worth of weights live:

1. **Per-layer residency.** `loadLayer(L)` reads block `L` into a slot buffer; the slot is recycled for a later layer. With the synchronous single-buffer path exactly one block is resident.
2. **Weights stay quantized; dequant is per-row.** In `Residency::Quantized` the loader keeps each weight's raw on-disk bytes resident and never bulk-expands them. `matmul_quant` dequantizes *one output row* into a tiny scratch buffer, dots it with the activation, and moves on — so a whole layer stays ~4 bits/weight in RAM instead of 32. `Residency::FP32` (dequant on load) exists mainly as the numeric oracle. Norm weights are tiny and always fp32.

<div class="diagram"><svg viewBox="0 0 760 250" role="img" aria-label="One block is streamed off disk into a bounded resident buffer, then reused"><defs><linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d9488"/><stop offset="1" stop-color="#0f766e"/></linearGradient><marker id="ar" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#0d9488"/></marker></defs><text x="12" y="22" font-family="ui-monospace,monospace" font-size="13" fill="#57606a">on disk — GGUF, quantized (may be tens of GB)</text><g font-family="ui-monospace,monospace" font-size="12"><rect x="12" y="32" width="84" height="32" rx="6" fill="#eef1f5" stroke="#d0d7de"/><text x="54" y="53" text-anchor="middle" fill="#57606a">block 0</text><rect x="104" y="32" width="84" height="32" rx="6" fill="#eef1f5" stroke="#d0d7de"/><text x="146" y="53" text-anchor="middle" fill="#57606a">block 1</text><rect x="196" y="32" width="84" height="32" rx="6" fill="#eef1f5" stroke="#d0d7de"/><text x="238" y="53" text-anchor="middle" fill="#57606a">block 2</text><text x="294" y="53" fill="#8b949e">…</text><rect x="326" y="32" width="96" height="32" rx="6" fill="#eef1f5" stroke="#d0d7de"/><text x="374" y="53" text-anchor="middle" fill="#57606a">block N-1</text><rect x="430" y="32" width="92" height="32" rx="6" fill="#eef1f5" stroke="#d0d7de"/><text x="476" y="53" text-anchor="middle" fill="#57606a">out head</text></g><path d="M238 70 L238 104" stroke="#0d9488" stroke-width="2" fill="none" marker-end="url(#ar)"/><text x="248" y="94" font-family="ui-monospace,monospace" font-size="11" fill="#0f766e">pread one block</text><rect x="150" y="108" width="470" height="94" rx="10" fill="none" stroke="#0d9488" stroke-dasharray="5 4"/><text x="162" y="126" font-family="ui-monospace,monospace" font-size="12" fill="#0f766e">resident in RAM — bounded</text><rect x="168" y="136" width="150" height="50" rx="8" fill="url(#g1)"/><text x="243" y="159" text-anchor="middle" font-family="ui-monospace,monospace" font-size="12" fill="#fff">1–2 layer buffers</text><text x="243" y="175" text-anchor="middle" font-family="ui-monospace,monospace" font-size="11" fill="#d6f5f1">attn + FFN</text><rect x="340" y="136" width="120" height="50" rx="8" fill="#2563eb"/><text x="400" y="165" text-anchor="middle" font-family="ui-monospace,monospace" font-size="12" fill="#fff">KV cache</text><path d="M318 161 L340 161" stroke="#57606a" stroke-width="1.5" marker-end="url(#ar)"/><text x="486" y="156" font-family="ui-monospace,monospace" font-size="12" fill="#1c2128">peak RAM ≈</text><text x="486" y="174" font-family="ui-monospace,monospace" font-size="12" fill="#1c2128">one layer + KV</text><path d="M243 188 C243 218, 168 210, 168 188" stroke="#8b949e" stroke-width="1.3" fill="none" stroke-dasharray="3 3" marker-end="url(#ar)"/><text x="150" y="238" font-family="ui-monospace,monospace" font-size="11" fill="#8b949e">free &amp; reuse the buffer for the next block</text></svg></div>

There are three interchangeable backends, all producing **identical output**:

- **Synchronous single-buffer `pread`** — one slot, one block resident (`--no-async`).
- **Async double-buffered prefetch ring** (default) — a background worker materializes block `L+1` while the compute thread runs block `L`. See [Predictive prefetch](prefetch.html).
- **`mmap`** — `read_raw_at` copies from mapped pages instead of `pread`, letting the OS page cache buffer (`--mmap`).

## Alternatives considered

| Approach | Peak RSS | Why not the default |
|:---------|:---------|:--------------------|
| Load everything resident (llama.cpp default) | ∝ model size | Cannot run a model larger than RAM at all — the entire point. |
| `mmap` the whole file | ∝ working set, uncontrolled | No hard ceiling; the OS decides residency and can still fault the whole model in. SipLLM offers mmap as one *backend*, but pairs it with an explicit budget. |
| Dequantize the layer to fp32 on load | 8× the quantized layer | Wastes the dominant memory term; kept only as `Residency::FP32` for the numeric oracle. |
| Keep every layer, evict LRU | tunable but complex | The [memory planner](memory-planner.html) + [layer residency](layer-residency.html) achieve the same RAM↔speed control with a simpler contiguous-pin model. |

## Tradeoffs

Streaming trades **disk bandwidth for memory**. In the bounded-memory extreme (budget 0) decode is disk-bound — arithmetic intensity is Θ(1), so an off-cache model runs well under 1 tok/s. That is the price of running a model that otherwise would not run at all. The [`--ram-budget`](memory-planner.html) dial buys the speed back linearly: pin more hot layers, re-stream fewer per token. Correctness is never traded — every backend and every budget yields bit-identical logits, guarded by `tests/test_e2e.cpp` and `tests/test_ram_budget.cpp`.

## Source files

| File | Role |
|:-----|:-----|
| `include/llm/weight_source.h` | the `WeightSource` seam (directory + metadata + positional reads) |
| `include/llm/loader.h` / `src/loader.cpp` | `LayerLoader`: slots, roles, residency, the prefetch worker |
| `include/llm/file_backing.h` | POSIX `pread_exact` / optional `mmap` shared by every source |
| `src/quant.cpp` | `matmul_quant` — the per-row dequantize-then-dot |
| `src/transformer.cpp` | the forward pass that drives `loadLayer`/`unloadLayer` |

## Future work

- **Deeper prefetch pipelining** and explicit `posix_fadvise`/`madvise(WILLNEED)` readahead (`FileBacking::prefetch()` exists but is currently unused).
- **Non-contiguous / hotness-aware pinning** — today pinning always covers the leading run `[0, n_pinned)`; there is no hotness heuristic.
- **Speculative streaming** to amortize weight movement in the exceeds-RAM regime (the long-term throughput moat).
