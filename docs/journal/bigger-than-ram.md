# Running bigger than RAM

**The headline demo: a 7.87&nbsp;GB model generating coherent text in 317&nbsp;MB of RAM.** Once [layer streaming](j-streaming-layers.html) made peak memory track a single layer's width instead of the whole model, the obvious test was to run a model that physically cannot be loaded resident on the machine — and watch it work anyway. It does.

## The problem: some models simply do not fit

Loading a model resident makes peak RSS scale with the file. On a 16&nbsp;GB Mac with only ~3&nbsp;GB free, a 7.87&nbsp;GB Q4 file is a non-starter — the standard runtime can't even allocate it, let alone leave room for the KV cache and OS. The question was never "how fast" but "at all": can a model many times larger than available RAM produce correct output?

## The fix: stream everything with a fixed working set

Nothing new was needed beyond the founding design — that is the point. The loader keeps one (or two, double-buffered) transformer blocks resident, weights stay quantized, and the [LM head streams](j-lm-head.html) in row blocks so the last resident floor drops too. The run below uses `--stream-lm-head --no-async --ctx 512`, greedy, on that same 16&nbsp;GB Mac with ~3&nbsp;GB free:

<div class="stat-grid"><div class="stat"><div class="stat-value">317 MB</div><div class="stat-label">peak RSS, Llama-2-13B</div><div class="stat-sub">7.87 GB Q4 weights &rarr; 25&times; smaller than the file</div></div><div class="stat"><div class="stat-value">25&times;</div><div class="stat-label">model &divide; RSS</div><div class="stat-sub">the largest model we've run</div></div><div class="stat"><div class="stat-value">~3 GB</div><div class="stat-label">free RAM on the host</div><div class="stat-sub">where loading it resident is impossible</div></div></div>

## Measured: the bigger-than-RAM table

| Model | Weights on disk | Peak RSS | Model ÷ RSS |
|:--|--:|--:|--:|
| TinyLlama-1.1B Q8_0 | 1.17 GB | 61 MB | 19× |
| Llama-3.1-8B Q4_K_M | 4.92 GB | 204 MB | 24× |
| Llama-2-13B Q4_K_M | 7.87 GB | 317 MB | 25× |

All three produced coherent output. Peak RSS grows with layer *width*, so the 13&nbsp;B model costs only ~1.5× the 8&nbsp;B's RSS despite being 60% larger on disk — depth is nearly free.

## The tradeoff: disk bandwidth for memory

This is not free speed — it is a *different budget*. In the bounded-memory extreme (budget 0) every weight is re-read from storage each token, so decode is disk-bound: arithmetic intensity is Θ(1) and an off-cache model runs well under 1 tok/s. That is the price of running a model that otherwise would not run at all.

> [!KEY] The `--ram-budget` dial
> `--ram-budget` buys the speed back linearly. It pins as many contiguous hot layers as fit under a hard weight-RSS ceiling and streams the rest, so peak RSS never exceeds the ceiling. Pin more → re-stream fewer layers per token → faster; pin nothing → minimum RAM. Output is bit-identical at every budget.

> [!WARNING] What the dial is *not*
> The auto-tuner does **not** choose the RAM budget — it only picks thread count and scheduler policy from matmul micro-benchmarks. RAM budgeting is the separate contract planner in `mem_plan.cpp`; you set the ceiling. The tradeoff is real disk I/O on the cold-cache phone target, not the warm-cache numbers above.

## What it unlocked

This is the capability the whole project exists to demonstrate: model size is decoupled from RAM. The mechanics of pinning and the RAM↔speed contract live in [layer residency](layer-residency.html) and the [memory planner](memory-planner.html); the founding seam is [Streaming the layers](j-streaming-layers.html).
