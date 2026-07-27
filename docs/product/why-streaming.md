# Why streaming?

<div class="hero"><p class="lead"><strong>Because an LLM's weights dwarf its activations, and a decoder only touches one transformer block at a time.</strong> If exactly one block is made resident and its memory reused for the next, peak weight RSS tracks <em>layer width</em>, not <em>model size</em> — and a model many times larger than RAM still runs.</p></div>

## The core argument

Loading a model resident makes peak RSS scale with *model size*: ~1.1&nbsp;GB for a 1.1&nbsp;B Q8 model, tens of GB for a 70&nbsp;B one. But block `L`'s weights are needed **only while block `L` runs**. Reuse one block's worth of memory across the whole depth and the dominant memory term collapses:

```text
peak RSS  ≈  one transformer layer  +  KV cache
          (flat across the whole model depth)
```

> [!KEY] Peak RSS = layer width, not model size
> A deeper model of the same width has *flat* peak RSS. The `toy_scaling` benchmark holds resident weights **flat at ~1.5 MB** across 4-, 16-, and 32-layer toy models — depth changed, footprint did not.

This is the whole thesis, and it is measured, not asserted. On a 16&nbsp;GB Mac with ~3&nbsp;GB free (where loading these resident is impossible), `--stream-lm-head --no-async --ctx 512`, greedy:

| Model | Weights on disk | Peak RSS | Model ÷ RSS |
|:------|----------------:|---------:|------------:|
| TinyLlama-1.1B (Q8_0) | 1.17 GB | 61 MB | 19× |
| Llama-3.1-8B (Q4_K_M) | 4.92 GB | 204 MB | 24× |
| Llama-2-13B (Q4_K_M) | 7.87 GB | 317 MB | 25× |

A 13&nbsp;B model runs in **317 MB — 25× smaller than its own weights** — on a machine that cannot hold it. See [Bigger than RAM](j-bigger-than-ram.html) for the full story.

## The seam that makes it safe

The transformer never opens a file. It talks to a **`WeightSource`** (`include/llm/weight_source.h`): a tensor directory plus positional reads (`read_raw_at`, a `pread` that never loads the whole file). `LayerLoader` (`src/loader.cpp`) reads block `L` into a slot buffer and recycles that slot for the next layer, so **swapping the loader or the residency strategy touches no line of math.** Two mechanisms keep only a layer live:

<div class="card-grid"><div class="card"><h3>Per-layer residency</h3><p><code>loadLayer(L)</code> reads block <code>L</code> into a recycled slot; the synchronous single-buffer path keeps exactly one block resident. See <a href="layer-residency.html">Layer residency</a>.</p></div><div class="card"><h3>Weights stay quantized</h3><p><code>matmul_quant</code> dequantizes <em>one output row</em> into a tiny scratch buffer, dots it, and moves on — a whole layer stays ~4 bits/weight in RAM instead of 32. See <a href="quantization.html">Quantization</a>.</p></div></div>

Three interchangeable backends — synchronous `pread` (`--no-async`), an async double-buffered [prefetch](prefetch.html) ring (default), and [`mmap`](why-not-mmap.html) — all produce identical logits. See [Streaming layer loader](streaming-loader.html).

## The RAM↔speed dial

Streaming trades **disk bandwidth for memory**. In the bounded-memory extreme (budget 0) decode is disk-bound — arithmetic intensity is Θ(1), so an off-cache model runs well under 1 tok/s. `--ram-budget N` buys the speed back linearly: it pins as many contiguous hot layers `[0, n_pinned)` resident as fit under a hard ceiling and re-streams fewer per token.

Measured (Apple M3, warm cache, `--ctx 512`, median-of-3):

| Model | Budget | Pinned | Decode tok/s | Streamed | Peak RSS |
|:------|-------:|-------:|-------------:|---------:|---------:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | 22.1 | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | 65.9 | 113 MB | 161 MB |

> [!MEASURED] Bit-identical at any budget
> Pinning is a pure cache: `tests/test_ram_budget.cpp` proves logits + KV are bit-for-bit identical across the whole budget sweep and that the hard ceiling holds. The dial changes *where* the bytes live, never *what* the math computes.

## When streaming wins

- **Edge devices with lots of storage, little RAM/VRAM** — phones, SBCs, embedded Linux. The exact target SipLLM was built for.
- **Running a model that otherwise would not fit at all** — the bounded-memory extreme is slow, but the alternative is "cannot run."
- **A hard peak-RSS ceiling matters** — when the OS must not fault the whole model in. This is the key difference from [mmap](why-not-mmap.html), which has no hard ceiling.

Where the model comfortably fits in RAM and you want maximum throughput, a mature resident runtime is the better tool — SipLLM sits at ~88% of llama.cpp's decode at 2.1× less RAM (TinyLlama Q8, resident). See [Why another runtime?](why-another-runtime.html) for the honest positioning.
