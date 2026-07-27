# FAQ

<div class="hero"><p class="lead">Honest answers, grounded in the code and the measured <a href="benchmarks.html">scorecard</a>. Where something is experimental, stubbed, or unmeasured, this page says so.</p></div>

## Does it really run models bigger than RAM?

Yes, and it is measured. On a 16&nbsp;GB Mac with only ~3&nbsp;GB free — where loading these resident is impossible — SipLLM streams **Llama-2-13B (7.87 GB Q4) at 317 MB peak RSS** (25× smaller than its weights) and Llama-3.1-8B (4.92 GB Q4) at 204 MB, producing coherent output. Peak RSS tracks a single transformer layer's width, not the model's size or depth. See [Bigger than RAM](j-bigger-than-ram.html).

## Is the output correct?

Yes. SipLLM diffs every transformer block's residual stream and the final logits against llama.cpp. **F16 is numerically identical** (cosine `1.000000`); as quantization coarsens, error grows *monotonically* (F16 ≪ Q8_0 < Q5_K_M < Q4_K_M) — pure quantization error, the signature of a faithful implementation. Cross-engine results are never bit-exact (summation order and rounding differ), so validation is numerical. Every quantization format in the [golden matrix](benchmarks.html) predicts `" Paris"` for `"The capital of France is"`.

## Why is streaming slow?

Because streaming trades **disk bandwidth for memory**. At the bounded-memory extreme (`--ram-budget 0`), decode is disk-bound — arithmetic intensity is Θ(1), so an off-cache model runs well under 1 tok/s. That is the price of running a model that otherwise would not run at all. Raise `--ram-budget` to pin hot layers and climb the [RAM↔speed dial](why-streaming.html): tinyllama goes 11.4 → 22.1 tok/s, smollm2 53.1 → 65.9, as the budget rises.

## So is it faster than llama.cpp?

No — not in general. Where the model fits in RAM, llama.cpp is faster; SipLLM's `--fast` Q8 decode sits at ~88% of it (2.1× less RAM). TTFT and prefill throughput trail llama.cpp substantially. SipLLM's win is **footprint and the ability to run models that do not fit at all** — see [Why another runtime?](why-another-runtime.html).

## Does it use the GPU?

Not really — treat it as CPU-only today. The Vulkan backend is **experimental and detection-only**: `make VULKAN=1` enables device detection (and deliberately rejects CPU software rasterizers like llvmpipe), but `vulkan_matmul` **always falls back to CPU and returns false** — the compute-shader dispatch is omitted even when the SPIR-V is embedded. Do not treat SipLLM as GPU-accelerated. A working matmul→full-layer offload path is on the [roadmap](roadmap.html).

## Why not just mmap the model?

`mmap` has **no hard ceiling** — the OS decides residency and can fault the whole model in, which is fatal on a memory-constrained phone. SipLLM offers `mmap` as one backend (its win is page-cache buffering; note it is *not* zero-copy — `read_raw_at` memcpys from mapped pages) but pairs it with an explicit budget contract that enforces a per-layer RAM ceiling. See [Why not just mmap?](why-not-mmap.html).

## Which models and architectures are supported?

The engine dispatches on `general.architecture` and implements **Llama, Mistral, Qwen2/2.5, Gemma&nbsp;2, Gemma&nbsp;3 text, Phi-3, Phi-2, GPT-2, and Mixtral/MoE**. A few caveats: sliding-window attention is **not** modeled for Mistral or Gemma3 (they attend the full causal range); "nemo"/Mistral-Nemo is not a recognized arch string and falls back through the Unknown→Llama path (which works because Nemo is Llama-like). The cross-engine golden matrix currently covers the Llama path.

## Which quantization formats can it read?

Dequant is supported for `F32`, `F16`, `BF16`, `Q4_0/1`, `Q5_0/1`, `Q8_0`, and the K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, plus `IQ4_NL`. **`Q8_1` and `Q8_K` are not dequantizable** (they throw). Only Q4_K and Q6_K have NEON dequant fast paths, and the `--fast` int8 SDOT kernel is **Q8_0-only and ARM-only** (`__ARM_FEATURE_DOTPROD`) — it is an approximation, so the fp32-dequant path stays the correctness oracle.

## What does `--fast` actually do?

It routes **Q8_0** projections through an int8 SDOT kernel (the same activation-quantization technique llama.cpp uses) instead of fp32-dequant-then-dot, closing most of the decode gap (tinyllama ~50 vs llama.cpp 57 tok/s). It is opt-in and ARM-only; the exact fp32 path remains the default and the numeric oracle, so all correctness tests are unchanged. Extending int-dot to K-quants (Q4_K) is the next step on the [roadmap](roadmap.html).

## Which platforms does it run on?

Linux, macOS, and Android (NDK) on ARM64 or x86 — CPU-first, with hand-written ARM64 NEON kernels and scalar fallbacks elsewhere. It was built and tested on a phone via Termux (Dimensity 8300). The toolchain is just `make` + a C++17 compiler; no PyTorch, ONNX, ggml, or BLAS. CI publishes Linux `x86_64`/`aarch64` bundles; macOS bundles are released manually from a local Mac.

## Is there an Android app?

There is a Flutter/FFI layer (Wave 8): a stable C ABI wrapping the C++ engine, a Dart runtime that streams tokens on a worker isolate and cancels mid-generate, on-device embeddings backed by a SQLite vector store, a resumable Hugging Face downloader, and phone→Wear OS transfer. **But it is host-verified on Apple M3 only — the Android APK and on-device inference are not yet runtime-verified.** No phone or watch performance number is quoted anywhere in these docs. See [Vision](vision.html), [Flutter runtime](flutter-runtime.html), and the [Android journal](j-android.html).

## Does it apply a chat template?

No. The engine runs inference over **raw text** and adds a BOS token only on a fresh sequence — it applies no chat/prompt template. The Flutter app formats conversations itself (chatml/llama3/zephyr/raw). If you use the raw engine, format the prompt yourself.

## Can I get embeddings?

Yes — via the final-layer hidden-state hook (L2-normalized), used by the Flutter app's SQLite vector store with cosine top-k search. One caveat: **`embed()` clears the KV state**, so use a dedicated runtime for embeddings, not one mid-conversation.

## Does it need the internet?

No — inference is fully offline. The only network use is *downloading* models: `sipllm pull`/`run` resolve names to public GGUF files and cache them under `~/.sipllm/models`. You can also pass any GGUF URL or local path directly. Once a model is local, generation needs no connection.

## Is there a built-in web UI?

Not bundled. `sipllm serve` exposes JSON + SSE APIs (`/api/model`, `/api/selftest`, `/api/generate`); **`server/index.html` is not shipped in the repo**, so the server serves a fallback page unless you supply your own `index.html`.

## Does the auto-tuner pick my RAM budget?

No. The auto-tuner picks **thread count** (cap 16) and **scheduler policy** via 4096×4096 matmul micro-benchmarks, cached at `~/.sipllm/runtime/<hw_id>.json` (only 4 of 7 policies are auto-benchmarked; no big.LITTLE detection). The RAM budget is a separate, explicit contract set by `--ram-budget` and planned by `src/mem_plan.cpp`. See [Auto-tuning](auto-tuning.html) and the [memory planner](memory-planner.html).
