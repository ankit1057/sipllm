# Streaming the layers

**This is the change everything else rests on.** A transformer's weights dwarf its activations, so the standard runtime — load the whole file resident — makes peak memory scale with *model size*: 1.1&nbsp;GB for a 1.1&nbsp;B Q8 model, tens of GB for a 70&nbsp;B one, which never fits on a phone. But a decoder only ever touches **one transformer block at a time**. We built the loader so exactly one block is resident and its memory is reused for the next, which makes peak weight RSS track a *layer's width*, not the *model's depth*.

## The problem: memory scaled with the wrong number

When we started, "run a bigger model" meant "buy more RAM." Every weight was resident for the whole run even though block `L`'s weights are needed only while block `L` executes. The math never needed them all at once — the *loader* was forcing it. If we could make block `L`'s memory reusable for block `L+1`, the dominant term (weights) would stop scaling with the number of layers entirely.

## The fix: a seam the math can't see through

The transformer never opens a file. It talks to a **`WeightSource`** (`include/llm/weight_source.h`) that exposes only three things: a tensor directory (`tensors()`, `find()`), typed metadata, and positional reads (`read_raw()` / `read_raw_at()`, a `pread` that never loads the whole file). Both the real GGUF parser (`GgufFile`) and the toy `.llmw` reader implement it, so the entire forward pass is oblivious to *where* bytes come from. That seam is what lets us change residency strategy without touching a line of arithmetic.

`LayerLoader` (`include/llm/loader.h`, `src/loader.cpp`) drives blocks through the seam. `Transformer::forward` (`src/transformer.cpp`) loops:

```text
embed_token(token)              -> x   (one row streamed, or resident if tied)
for layer L in 0 .. n_layers-1:
    loadLayer(L)                -> make block L's weights resident
    block(L, pos)               -> RMSNorm -> QKV -> RoPE -> GQA attn -> proj -> RMSNorm -> SwiGLU
    unloadLayer()               -> release block L for reuse
final RMSNorm + project_output  -> logits
```

Each block asks for weights by **`Role`** (`AttnNorm, AttnQ, AttnK, AttnV, AttnOut, FfnNorm, FfnGate, FfnUp, FfnDown` at the core), which the loader maps to GGUF names like `blk.<L>.attn_q.weight`. Two mechanisms keep only a layer's worth of weights live:

1. **Per-layer residency.** `loadLayer(L)` reads block `L` into a slot buffer; `unloadLayer()` releases it for reuse. With the synchronous single-buffer path exactly one block is resident.
2. **Weights stay quantized; dequant is per-row.** In `Residency::Quantized` the loader keeps each weight's raw on-disk bytes resident and never bulk-expands them. `matmul_quant` (`src/quant.cpp`) dequantizes *one output row* into a tiny scratch buffer, dots it with the activation, and moves on — so a whole layer stays ~4 bits/weight in RAM instead of 32. `Residency::FP32` (dequant on load) survives only as the numeric oracle for the equivalence tests. This is the subject of [Flat resident weights](j-resident-weights.html).

> [!KEY] The invariant
> Peak weight RSS is bounded by *one layer's* width, so a deeper model of the same width has *flat* peak RSS. Depth became free.

## Measured: the same model, dialed from resident to streamed

TinyLlama-1.1B Q8_0, `--ctx 512`, greedy, 4 threads, warm cache — one model, one dial:

| Runtime | Peak RSS | Decode |
|:--|--:|--:|
| llama.cpp (CPU) | 2326 MB | ~57 tok/s |
| SipLLM `--fast --ram-budget 1200M` (resident) | 1113 MB | ~50 tok/s |
| SipLLM `--fast` (streaming) | 175 MB | 6.8 tok/s |

Resident, we use **2.1× less RAM** at ~88% of llama.cpp's decode; fully streamed, **13× less RAM** for the same output. The streamed path is disk-bound — that tradeoff is the whole point of [Running bigger than RAM](j-bigger-than-ram.html) — and `--ram-budget` buys the speed back linearly.

> [!MEASURED] Depth-independence, three ways
> On a 16&nbsp;GB Mac with ~3&nbsp;GB free, peak RSS grew with layer width and never with depth: TinyLlama-1.1B (1.17&nbsp;GB) in 61&nbsp;MB, Llama-3.1-8B (4.92&nbsp;GB) in 204&nbsp;MB, Llama-2-13B (7.87&nbsp;GB) in 317&nbsp;MB — 19–25× smaller than the file.

> [!NOTE] Honest limits
> The `--fast` int8 SDOT kernel is Q8_0-only, ARM-only (`__ARM_FEATURE_DOTPROD`), and an *approximation* (the activation is quantized); the fp32-dequant path stays the correctness oracle. Every backend and every budget yields bit-identical logits from the fp32 path, guarded by `tests/test_e2e.cpp` and `tests/test_ram_budget.cpp`.

## What it unlocked

The seam is why every later article is possible: streaming is switchable per-backend, the [async prefetch ring](j-prefetch.html) overlaps I/O with compute, [single-pass prefill](j-prefill.html) stops re-streaming, and [streaming the LM head](j-lm-head.html) removes the last resident floor. The reference for the seam and its three interchangeable backends is the [streaming layer loader](streaming-loader.html) architecture page.
