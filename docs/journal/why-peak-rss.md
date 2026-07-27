# Why peak RSS matters

**On an edge device the number that decides whether your model runs is not average memory, not model size on disk — it is *peak resident set size*. The OS out-of-memory killer fires at the high-water mark, so a runtime that averages 100 MB but spikes to 2 GB is a runtime that dies on a 1 GB phone. SipLLM treats peak RSS as its North Star metric, measures it externally with `/usr/bin/time -l`, and is architected so that peak tracks a single layer's width — not the model's depth or total size. This is why.**

## Why the peak, not the average

Memory pressure is not billed by the average. The kernel's OOM killer watches the *instantaneous* resident set, and it acts at the worst moment — a prefill spike, an LM-head projection, a KV growth event. A runtime whose mean footprint is tiny but whose transient peak blows past physical RAM gets killed at that peak, and the average never enters into it. So the only honest memory number for "can this device run this model" is the maximum RSS observed across the entire run. Everything SipLLM reports is a peak.

## Why not model size

Model size is a property of the *file*, not the *process*. A 7.87 GB Q4 checkpoint is 7.87 GB on disk whether you run it or not; what matters is how much of it is resident at once. Loading everything makes peak RSS scale with model size — the conventional approach, and the reason an 8 B model "does not fit on a phone." Streaming decouples the two:

| Model | Weights on disk | Peak RSS | Model ÷ RSS |
|:--|--:|--:|--:|
| TinyLlama-1.1B Q8_0 | 1.17 GB | 61 MB | 19× |
| Llama-3.1-8B Q4_K_M | 4.92 GB | 204 MB | 24× |
| Llama-2-13B Q4_K_M | 7.87 GB | 317 MB | 25× |

A 13 B model runs in **317 MB** — 25× smaller than the file — on a 16 GB Mac with only ~3 GB free, where loading it resident is simply impossible. Peak RSS, not model size, is what the device experiences.

## Peak tracks layer width, not model depth

Here is the structural claim, and it is the thesis: because the decoder touches exactly one transformer block at a time, streaming makes peak weight RSS track *layer size* (model width) rather than *model size* (depth). The evidence is that **resident weights stay flat at ~1.5 MB across toy models of 4, 16, and 32 layers** — adding depth adds no resident weight at all. On real models the resident working set is one layer's worth: ~37.6 MB for a smollm2 layer, ~106.5 MB for a tinyllama layer.

> [!KEY] The consequence
> A deeper model of the *same width* has the *same* peak RSS. Depth is free; only width costs memory. That is the inversion streaming buys — and it is why the largest-runnable-model number is bounded by a layer, not the whole network.

## `/usr/bin/time -l` is the authoritative number

Peak RSS could be self-reported, but self-reported memory is easy to fool — internal allocators lie, arenas hide, and every runtime accounts differently. So SipLLM's headline figures come from **`/usr/bin/time -l`**: the OS measures the process's maximum resident set from the outside, after the fact, in a way the program cannot game. Crucially, it is *cross-runtime* — the exact same tool measures llama.cpp, vLLM, or anything else with a PID. That is what makes the comparisons fair:

<div class="stat-grid"><div class="stat"><div class="stat-value">121 MB</div><div class="stat-label">SipLLM tinyllama Q8 (stream)</div><div class="stat-sub">vs llama.cpp 1356 MB — 11× smaller</div></div><div class="stat"><div class="stat-value">54 MB</div><div class="stat-label">SipLLM smollm2 (stream)</div><div class="stat-sub">vs llama.cpp 546 MB — 10× smaller</div></div></div>

All figures are Apple M3, warm cache, median-of-3, and committed as JSON under `bench/results/`.

## Contrast with mmap: a working set with no ceiling

The obvious objection is "just `mmap` the file and let the OS page it." But `mmap` gives you a working set with **no hard ceiling**: the OS decides residency, and under a favorable access pattern it can fault the entire model into RAM — exactly the spike the OOM killer punishes. SipLLM offers `mmap` as one *backend*, but pairs it with an explicit budget so the peak stays bounded. The [`--ram-budget`](memory-planner.html) dial makes this a guarantee, not a hope — the loader pins as many hot layers as fit under the ceiling and streams the rest, and the measured peak never exceeds the budget:

| Model | Budget | Pinned | Decode tok/s | Peak RSS |
|:--|--:|--:|--:|--:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 480 MB |
| tinyllama | 768M | 22/22 | 22.1 | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 54 MB |
| smollm2 | 256M | 30/30 | 65.9 | 161 MB |

Peak RSS ≤ budget at every point — a hard ceiling `mmap` cannot offer.

That controllable, externally-measured peak is the whole product. The mechanism that enforces it is the [memory planner](memory-planner.html); the full measured record — accuracy, speed, and peak across every model and budget — is in [Benchmarks](benchmarks.html); and the argument against paging the whole file is in [Why not mmap everything?](why-not-mmap.html).
