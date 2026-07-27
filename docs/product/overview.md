# SipLLM

<div class="hero"><p class="lead"><strong>Run GGUF models larger than available RAM through bounded-memory transformer&nbsp;layer&nbsp;streaming.</strong> SipLLM is a dependency-free, CPU-first LLM inference engine in C++17 that <em>sips</em> weights off disk one transformer block at a time, so peak memory tracks a single resident layer — not the whole model. A Flutter/Android SDK turns that runtime into an offline AI computer you carry in your pocket.</p><div class="hero-cta"><a class="btn primary" href="why-streaming.html">Why streaming? &#8594;</a><a class="btn ghost" href="architecture.html">Architecture book</a><a class="btn ghost" href="benchmarks.html">Measured numbers</a></div></div>

Start anywhere:

<div class="flow"><a href="architecture.html">Architecture</a><span class="arrow">&#8595;</span><a href="j-streaming-layers.html">Streaming thesis</a><span class="arrow">&#8595;</span><a href="benchmarks.html">Benchmarks</a><span class="arrow">&#8595;</span><a href="api.html">API</a><span class="arrow">&#8595;</span><a href="flutter-runtime.html">Flutter</a><span class="arrow">&#8595;</span><a href="j-android.html">Android</a><span class="arrow">&#8595;</span><a href="vision.html">Playground</a><span class="arrow">&#8595;</span><a href="contributing.html">Contributing</a></div>

## The one idea

The usual way to run an LLM loads the entire model into memory. A 1.1&nbsp;B model in Q8_0 is ~1.1&nbsp;GB resident; an 8&nbsp;B model will not fit on a phone at all. SipLLM instead streams one transformer block at a time: `pread` the block's weights from disk, run attention + FFN, free it, move on. Only a single layer's weights (plus the KV cache) are ever resident, so a model many times larger than RAM still runs — **memory is bounded by _layer_ size, not _model_ size.**

> [!KEY] Central hypothesis
> *"LLM inference should be bounded by a configurable working set rather than total model size."* SipLLM is not trying to replace llama.cpp — it exists to prove a different execution model, and every number here is measured against llama.cpp on the same hardware.

## Measured, not claimed

Apple M3, CPU-only, warm cache, median-of-3; peak RSS from `/usr/bin/time -l` (the authoritative cross-runtime figure). Reproducible and committed as JSON under `bench/results/`.

<div class="stat-grid"><div class="stat"><div class="stat-value">317 MB</div><div class="stat-label">peak RSS to run Llama-2-13B</div><div class="stat-sub">7.87 GB Q4 weights &rarr; 25&times; smaller than the model</div></div><div class="stat"><div class="stat-value">2.1&times;</div><div class="stat-label">less RAM than llama.cpp</div><div class="stat-sub">at ~88% of its decode (TinyLlama Q8, resident)</div></div><div class="stat"><div class="stat-value">13&times;</div><div class="stat-label">less RAM, streaming</div><div class="stat-sub">same TinyLlama in 175 MB vs 2326 MB</div></div><div class="stat"><div class="stat-value">0</div><div class="stat-label">runtime dependencies</div><div class="stat-sub">standard C++17 + pthreads only</div></div></div>

### Bigger than RAM — the defining capability

Measured on a 16&nbsp;GB Mac with only ~3&nbsp;GB free, where loading these models resident is impossible (`--stream-lm-head --no-async --ctx 512`, greedy):

| Model | Weights on disk | Peak RSS | Model ÷ RSS | Output |
|:------|----------------:|---------:|------------:|:-------|
| TinyLlama-1.1B (Q8_0) | 1.17 GB | 61 MB | 19× | coherent |
| Llama-3.1-8B (Q4_K_M) | 4.92 GB | 204 MB | 24× | coherent |
| Llama-2-13B (Q4_K_M) | 7.87 GB | 317 MB | 25× | coherent |

Peak RSS grows with layer *width*, never model depth/total size — a deeper model of the same width has flat peak RSS.

### Half the RAM, comparable speed

TinyLlama-1.1B Q8_0, `--ctx 512`, greedy, 4 threads, warm cache:

| Runtime | Peak RSS | Decode | vs llama.cpp |
|:--------|---------:|-------:|:-------------|
| llama.cpp (CPU, `-ngl 0 -t 4`) | 2326 MB | ~57 tok/s | baseline |
| **SipLLM** `--fast --ram-budget 1200M` (resident) | 1113 MB | ~50 tok/s | **2.1× less RAM**, ~12% slower |
| **SipLLM** `--fast` (streaming) | 175 MB | 6.8 tok/s | **13.3× less RAM** |

It is a smooth RAM↔speed dial, not a fixed point. Output is numerically equivalent — see [Benchmarks & validation](benchmarks.html) for the layer-by-layer diff against llama.cpp.

## What is inside

<div class="card-grid"><div class="card"><h3>Real GGUF parser</h3><p>Loads unmodified GGUF v2/v3 files from Hugging Face — metadata, tensor directory, and the common quantizations.</p></div><div class="card"><h3>Streaming layer loader</h3><p>Synchronous <code>pread</code>, an async double-buffered prefetcher, or an <code>mmap</code> backend — switchable and benchmarked side by side.</p></div><div class="card"><h3>RAM↔speed dial</h3><p><code>--ram-budget</code> pins as many hot layers as fit under a hard peak-RSS ceiling and streams the rest. Output is bit-identical at any budget.</p></div><div class="card"><h3>Nine architectures</h3><p>Llama, Mistral, Qwen2/2.5, Gemma&nbsp;2, Gemma&nbsp;3 text, Phi-3, Phi-2, GPT-2, and Mixtral/MoE — dispatched on <code>general.architecture</code>.</p></div><div class="card"><h3>ARM64 NEON kernels</h3><p><code>sdot</code>-accelerated int8 matmul with an x86 AVX2/FMA path and scalar fallbacks everywhere else.</p></div><div class="card"><h3>Flutter / Android SDK</h3><p>A stable C ABI wraps the engine; a Dart isolate streams tokens to a phone &amp; Wear OS UI with downloads, embeddings, and benchmarks — all offline.</p></div></div>

## Quick start

```bash
# one-liner installer (prebuilt release, or builds from source)
curl -fsSL https://raw.githubusercontent.com/ankit1057/sipllm/main/install.sh | sh
sipllm run tinyllama -p "The capital of France is"

# from source — the entire toolchain is make + a C++17 compiler
git clone https://github.com/ankit1057/sipllm.git && cd sipllm
make            # -> build/llm, build/bench, build/inspect_gguf, ...
make test       # dependency-free unit suite, all green

# run a model bigger than your RAM
./build/llm model.gguf -p "prompt" -n 40 --stream-lm-head --no-async
./build/llm model.gguf -p "prompt" --ram-budget 512M   # cap peak RSS
```

> [!NOTE] Where to go next
> New here? Read [What is SipLLM?](what-is-sipllm.html) and [Why streaming?](why-streaming.html). Want the mechanics? The [Architecture book](architecture.html) is a module-by-module tour, and the [Engineering journal](journal.html) tells the story behind each optimization. Building on it? Jump to the [API reference](api.html).
