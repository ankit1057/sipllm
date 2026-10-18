# Roadmap

<div class="hero"><p class="lead"><strong>Everything on this page is planned, not measured.</strong> The shipped, measured state lives in the North Star scorecard (<a href="benchmarks.html">Benchmarks</a>) and the <a href="performance-history.html">performance history</a>. This page is the ranked forward look — engine throughput moats first, then productization.</p></div>

> [!WARNING] Planned ≠ measured
> No number below is a benchmark. Per the engineering charter, speculation and future ideas are kept strictly separate from measured facts. Items here earn their place only if they reduce peak RSS, raise throughput/expansion factor, improve correctness, or improve portability.

## v0.5 — the documentation portal

The current milestone is **this portal**: a static, self-contained docs site (product pages, the architecture book, the API reference, the engineering journal, and project meta) rendered by a stdlib generator (`docs/build.py`) with every number sourced from the committed North Star scorecard. Goal: make the streaming thesis and its evidence navigable, so the *why* is as legible as the *what*.

History to date: tags v0.1.0 → v0.1.1 → v0.4.0 across 35 commits; the project began as a "Streaming GGUF LLM inference engine (C++17)" and reached a Developer Preview at v0.4.0.

## Engine roadmap (ranked)

<div class="card-grid"><div class="card"><h3>1 · Numbers on the target hardware</h3><p><strong>Shipped 2026-10-19:</strong> Q4_K and Q6_K int8 SDOT kernels under <code>--fast</code> (resident 4-bit decode 21 → 60 tok/s on M3, 62–67% of llama.cpp in half the RAM). <strong>Still missing:</strong> any run on a phone (Snapdragon / Dimensity) or an AMD host — the AVX2 path is cross-compiled only. The standing goal (beat LiteRT / MediaPipe on ARM and AMD CPUs) cannot be claimed or refuted without them. This is item 1 because it decides everything after it.</p></div><div class="card"><h3>2 · Remaining resident gap, then prefill</h3><p>67% → parity on M3: the fp32 attention/KV path and the per-token overheads now dominate; prefill is 73 vs llama.cpp's 266 tok/s. Speculative decoding (<code>--spec N</code>, shipped and proven token-identical) already gives 3.2× on copy-heavy output in the streaming regime; a batched lm_head projection in the verify pass and a draft source better than prompt lookup are the next steps there.</p></div><div class="card"><h3>3 · Wider prefetch + NEON K-quant</h3><p>Deeper prefetch pipelining (the unused <code>FileBacking::prefetch()</code> <code>fadvise</code>/<code>madvise</code> readahead hook) and NEON dequant fast paths beyond the current Q4_K/Q6_K coverage.</p></div><div class="card"><h3>4 · Vulkan matmul &rarr; full-layer offload</h3><p>The GPU path is a stub today (see below). The plan: land a working <code>vulkan_matmul</code> compute-shader dispatch, then a full-layer offload for the rare edge device with a usable GPU. CPU stays the primary, correctness-defining target.</p></div><div class="card"><h3>5 · Android on-device verification</h3><p>Wave 8 (Flutter FFI, downloader, embedding store, Wear transfer) is host-verified on Apple M3 but <strong>not</strong> yet runtime-verified on a phone. Verifying the Android APK / on-device inference (POCO X6 Pro) and Wear transfer (OnePlus Watch 2) is the remainder of milestone M6.5.</p></div></div>

## The GPU caveat, up front

> [!CAUTION] Vulkan is detection-only today
> `make VULKAN=1` enables device *detection* and deliberately rejects CPU software rasterizers (llvmpipe). But `vulkan_matmul` **always falls back to CPU and returns false** — the compute-shader dispatch is omitted even when `matmul.comp`/SPIR-V is embedded. The "Vulkan matmul → full-layer offload" item above is precisely the work of making that path real. Until it lands, do not treat SipLLM as GPU-accelerated.

## Not on the roadmap

Per the [vision guardrail](vision.html), Studio will **not** grow new inference architectures or model families to chase parity. The engine's architecture set is deliberate; forward work sharpens bounded-memory streaming, not breadth. Correctness is never traded for speed — every backend and budget must keep the [golden matrix](benchmarks.html) green.

> [!NOTE] Where the ranking comes from
> The order follows the charter's Rule 0 (measure → rank by impact × leverage × differentiation) and the current North Star "largest bottleneck": the kernels now exist; what is missing is evidence on the hardware the standing goal names (ARM phones, AMD laptops), then the last third of resident decode and prefill on M3. See [Benchmarks](benchmarks.html) for the measured baseline these plans build on.
