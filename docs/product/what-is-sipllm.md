# What is SipLLM?

<div class="hero"><p class="lead"><strong>SipLLM is a dependency-free, CPU-first LLM inference engine in C++17 that runs GGUF models larger than available RAM at a bounded, configurable peak memory ceiling.</strong> It <em>sips</em> weights off disk one transformer block at a time, so peak RSS tracks a single resident layer — not the whole model. It is not a llama.cpp replacement; it is a different execution model.</p><div class="hero-cta"><a class="btn primary" href="why-streaming.html">Why streaming? &#8594;</a><a class="btn ghost" href="benchmarks.html">Measured numbers</a><a class="btn ghost" href="architecture.html">Architecture book</a></div></div>

## The one idea

Almost every runtime loads the whole model into memory before it can generate a token. A 1.1&nbsp;B model in Q8_0 is ~1.1&nbsp;GB resident; an 8&nbsp;B model will not fit on a phone at all. SipLLM inverts that: it reads one transformer block's weights from disk with `pread`, runs attention + FFN, releases the block, and reuses that memory for the next one. Only a single layer's weights (plus the KV cache) are ever resident.

> [!KEY] Central hypothesis
> *"LLM inference should be bounded by a configurable working set rather than total model size."* Peak memory is bounded by **layer width**, not **model depth** — a deeper model of the same width has flat peak RSS.

Measured on a 16&nbsp;GB Mac with only ~3&nbsp;GB free (where loading these resident is impossible), `--stream-lm-head --no-async --ctx 512`, greedy:

<div class="stat-grid"><div class="stat"><div class="stat-value">317 MB</div><div class="stat-label">peak RSS to run Llama-2-13B</div><div class="stat-sub">7.87 GB Q4 weights &rarr; 25&times; smaller than the model</div></div><div class="stat"><div class="stat-value">0</div><div class="stat-label">runtime dependencies</div><div class="stat-sub">standard C++17 + pthreads only</div></div><div class="stat"><div class="stat-value">C++17</div><div class="stat-label">the entire toolchain</div><div class="stat-sub">no PyTorch, ONNX, ggml, or BLAS</div></div></div>

## What it is

- **A streaming execution model.** The transformer never opens a file; it talks to a `WeightSource` seam (`include/llm/weight_source.h`) that exposes a tensor directory plus positional reads. `LayerLoader` (`src/loader.cpp`) drives blocks through that seam with a synchronous `pread` path, an async double-buffered prefetcher, or an `mmap` backend — all producing identical output. See [Streaming layer loader](streaming-loader.html).
- **A hard RAM↔speed dial.** `--ram-budget N` pins as many hot layers resident as fit under a byte ceiling and streams the rest, so peak weight RSS never exceeds the budget. Output is bit-for-bit identical at any budget — the dial only changes *where* the bytes live. See [Memory planner](memory-planner.html).
- **A real GGUF engine.** It loads unmodified GGUF v2/v3 files from Hugging Face and dispatches on `general.architecture`: Llama, Mistral, Qwen2/2.5, Gemma&nbsp;2, Gemma&nbsp;3 text, Phi-3, Phi-2, GPT-2, and Mixtral/MoE. Dequant covers F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0 and the K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, plus IQ4_NL.
- **Correct.** For the same model and prompt, SipLLM dumps every block's residual stream and the final logits and diffs them numerically against llama.cpp. F16 is cosine `1.000000`; quantization error grows monotonically as the format coarsens — the signature of a faithful implementation. See [Benchmarks &amp; validation](benchmarks.html).
- **Edge-first, so CPU-first.** It targets phones, SBCs, and other hardware with lots of storage but little RAM/VRAM. Hand-written ARM64 NEON kernels, scalar fallbacks elsewhere.

## What it is not

<div class="card-grid"><div class="card"><h3>Not a llama.cpp replacement</h3><p>It exists to prove a different execution model, and every number is measured <em>against</em> llama.cpp on the same hardware. Where the model fits in RAM, mature resident runtimes are faster.</p></div><div class="card"><h3>Not a GPU runtime</h3><p>The Vulkan backend is <span class="badge exp">experimental</span> — <code>make VULKAN=1</code> enables device <em>detection</em> only; <code>vulkan_matmul</code> always falls back to CPU. Never treat it as working GPU acceleration.</p></div><div class="card"><h3>Not a chat framework</h3><p>The engine runs inference over raw text and adds BOS only on a fresh sequence — it applies <strong>no</strong> chat/prompt template. The Flutter app formats conversations itself (chatml/llama3/zephyr/raw).</p></div><div class="card"><h3>Not fast in the extreme</h3><p>Streaming an off-cache model is disk-bound (well under 1 tok/s at budget 0). That is the price of running a model that otherwise would not run at all; the RAM budget buys the speed back.</p></div></div>

## Where to go next

> [!NOTE] Reading order
> The argument is in [Why streaming?](why-streaming.html); the honest positioning against other runtimes is in [Why another runtime?](why-another-runtime.html); the mmap comparison is in [Why not just mmap?](why-not-mmap.html). For the mechanics, the [Architecture book](architecture.html) is a module-by-module tour. For the roadmap and the offline-AI-computer vision, see [Vision](vision.html) and [Roadmap](roadmap.html).
