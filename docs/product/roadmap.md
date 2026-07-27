# Roadmap

<div class="hero"><p class="lead"><strong>Everything on this page is planned, not measured.</strong> The shipped, measured state lives in the North Star scorecard (<a href="benchmarks.html">Benchmarks</a>) and the <a href="performance-history.html">performance history</a>. This page is the ranked forward look — engine throughput moats first, then productization.</p></div>

> [!WARNING] Planned ≠ measured
> No number below is a benchmark. Per the engineering charter, speculation and future ideas are kept strictly separate from measured facts. Items here earn their place only if they reduce peak RSS, raise throughput/expansion factor, improve correctness, or improve portability.

## v0.5 — the documentation portal

The current milestone is **this portal**: a static, self-contained docs site (product pages, the architecture book, the API reference, the engineering journal, and project meta) rendered by a stdlib generator (`docs/build.py`) with every number sourced from the committed North Star scorecard. Goal: make the streaming thesis and its evidence navigable, so the *why* is as legible as the *what*.

History to date: tags v0.1.0 → v0.1.1 → v0.4.0 across 35 commits; the project began as a "Streaming GGUF LLM inference engine (C++17)" and reached a Developer Preview at v0.4.0.

## Engine roadmap (ranked)

<div class="card-grid"><div class="card"><h3>1 · K-quant int-dot</h3><p>Today the <code>--fast</code> int8 SDOT kernel is <strong>Q8_0-only</strong> and ARM-only; K-quant (Q4_K) decode still runs the fp32-dequant path. Extending integer dot to K-quants brings Q8-class speed to 4-bit models — the biggest RAM headline and the strongest next demo.</p></div><div class="card"><h3>2 · Speculative / pipelined streaming</h3><p>In the exceeds-RAM regime decode is disk-bandwidth-bound (arithmetic intensity Θ(1)). Speculative streaming amortizes weight movement across tokens — the long-term throughput moat for models that do not fit.</p></div><div class="card"><h3>3 · Wider prefetch + NEON K-quant</h3><p>Deeper prefetch pipelining (the unused <code>FileBacking::prefetch()</code> <code>fadvise</code>/<code>madvise</code> readahead hook) and NEON dequant fast paths beyond the current Q4_K/Q6_K coverage.</p></div><div class="card"><h3>4 · Vulkan matmul &rarr; full-layer offload</h3><p>The GPU path is a stub today (see below). The plan: land a working <code>vulkan_matmul</code> compute-shader dispatch, then a full-layer offload for the rare edge device with a usable GPU. CPU stays the primary, correctness-defining target.</p></div><div class="card"><h3>5 · Android on-device verification</h3><p>Wave 8 (Flutter FFI, downloader, embedding store, Wear transfer) is host-verified on Apple M3 but <strong>not</strong> yet runtime-verified on a phone. Verifying the Android APK / on-device inference (POCO X6 Pro) and Wear transfer (OnePlus Watch 2) is the remainder of milestone M6.5.</p></div></div>

## The GPU caveat, up front

> [!CAUTION] Vulkan is detection-only today
> `make VULKAN=1` enables device *detection* and deliberately rejects CPU software rasterizers (llvmpipe). But `vulkan_matmul` **always falls back to CPU and returns false** — the compute-shader dispatch is omitted even when `matmul.comp`/SPIR-V is embedded. The "Vulkan matmul → full-layer offload" item above is precisely the work of making that path real. Until it lands, do not treat SipLLM as GPU-accelerated.

## Not on the roadmap

Per the [vision guardrail](vision.html), Studio will **not** grow new inference architectures or model families to chase parity. The engine's architecture set is deliberate; forward work sharpens bounded-memory streaming, not breadth. Correctness is never traded for speed — every backend and budget must keep the [golden matrix](benchmarks.html) green.

> [!NOTE] Where the ranking comes from
> The order follows the charter's Rule 0 (measure → rank by impact × leverage × differentiation) and the current North Star "largest bottleneck": Q8 `--fast` is within ~12% of llama.cpp, so the two open fronts are 4-bit (Q4_K) decode speed and streaming-regime throughput. See [Benchmarks](benchmarks.html) for the measured baseline these plans build on.
