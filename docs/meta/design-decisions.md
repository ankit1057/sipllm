# Design decisions

The load-bearing choices behind SipLLM, each stated as a question and answered
from the [engineering charter](contributing.html) (`CLAUDE.md`) and the measured
record. The through-line is one hypothesis: *"LLM inference should be bounded by
a configurable working set rather than total model size."* Every decision below
either serves that thesis or the constraints (edge-first, CPU-first, zero
dependencies) that make it deployable.

> [!KEY] The charter test
> `CLAUDE.md` demands every change earn its place against eight levers — reduce
> peak RSS, raise throughput, cut latency, grow the largest runnable model,
> improve correctness/portability, or simplify. These decisions are the ones
> that reduce peak RSS or enlarge the runnable model without trading correctness.

## Why streaming, not resident?

Loading all weights resident makes peak RSS scale with *model size*: a 1.1&nbsp;B
Q8_0 model is ~1.1&nbsp;GB resident, and an 8&nbsp;B model does not fit on a
phone at all. But a decoder touches exactly **one transformer block at a time** —
block `L`'s weights are live only while block `L` runs. SipLLM makes one block
resident, runs it, and reuses that memory for the next, so peak weight RSS tracks
*layer* size (model width), not *model* size (depth). That is the entire reason
the runtime exists: it is what lets Llama-2-13B (7.87&nbsp;GB Q4) run in
**317&nbsp;MB** on a machine that cannot hold it resident. See
[Streaming layer loader](streaming-loader.html) and
[Why streaming?](why-streaming.html).

## Why offer mmap but not rely on it?

`mmap` is the obvious "don't load it all" trick, and SipLLM ships it as one
interchangeable backend (`--mmap`). But it is *not* the thesis, for two reasons.
First, it gives **no hard ceiling**: the OS decides residency and can still fault
the whole model into the page cache under memory pressure — you get "usually
small" instead of "provably bounded." Second, in this loader mmap is **not
zero-copy** anyway: `read_raw_at` memcpys from mapped pages into a resident
buffer, so the win is OS page-cache buffering, not free residency
(`FileBacking::prefetch()` / `madvise` readahead exists but is currently unused).
SipLLM's contribution is the explicit `--ram-budget` ceiling, not the mapping
primitive; mmap is a benchmarkable alternative, not the design. See
[Why not mmap?](why-not-mmap.html).

## Why not cache all layers — why a budget *contract*?

Because "keep everything resident" is the problem, and "evict LRU" trades one
uncontrolled memory curve for another. SipLLM instead exposes a **contract**:
`--ram-budget N` is a hard peak-RSS ceiling. The [memory planner](memory-planner.html)
pins as many contiguous hot layers `[0, n_pinned)` as fit under the budget
(after reserving the KV cache and a scratch allowance) and streams the rest, with
a per-layer guard that keeps `resident_bytes() ≤ budget`. This turns
*bounded-RSS XOR speed* into a tunable continuum instead of a fixed point.

<div class="stat-grid"><div class="stat"><div class="stat-value">11.4&rarr;22.1</div><div class="stat-label">tinyllama decode tok/s</div><div class="stat-sub">budget 0 &rarr; 768M (0/22 &rarr; 22/22 pinned)</div></div><div class="stat"><div class="stat-value">bit-identical</div><div class="stat-label">logits at any budget</div><div class="stat-sub">pinning is a pure cache — <code>--ram-budget 0</code> == prior behavior</div></div><div class="stat"><div class="stat-value">&le; budget</div><div class="stat-label">peak RSS, every point</div><div class="stat-sub">hard ceiling, proven by a fuzzed budget sweep</div></div></div>

The alternative — an LRU eviction cache — reaches the same RAM↔speed control but
with more moving parts and no provable ceiling. A contiguous-pin model is simpler
to reason about, cheaper to guard, and correct by construction (see
[Layer residency](layer-residency.html)).

## Why C++17 and zero dependencies?

The charter names it a **hard rule**: pure C++17 + `pthread`, no PyTorch, no ONNX,
no ggml, no BLAS, no CMake. The reasons are portability and honesty. Portability:
the target set is Linux · macOS · Android (NDK) · embedded Linux, and the only
toolchain is `make` + a C++17 compiler — the same source cross-compiles per-ABI
for Android without dragging a runtime stack onto the device. Honesty: a
from-scratch engine means the streaming/quant/attention code is auditable end to
end, and the [test harness](contributing.html) itself has zero deps (no
gtest/catch2). A PR that adds a third-party library to the inference path is not
merged. See [Contributing](contributing.html).

## Why GGUF (plus a `.llmw` toy) only?

GGUF is the de-facto edge quantization container: unmodified files pull straight
from Hugging Face, and reading it directly means SipLLM validates against
llama.cpp on *identical bytes*. The engine parses GGUF v2/v3 metadata, the tensor
directory, and the common quantizations behind a single `WeightSource` seam
(`include/llm/weight_source.h`). The toy `.llmw` reader (`ModelFile`) implements
the same seam purely so tests and `toy_scaling` can synthesize models of
arbitrary depth to prove residency is flat vs depth — it is a test fixture, not a
second production format. One real format keeps the parser honest and the golden
comparison meaningful. See [GGUF parser](gguf-parser.html).

## Why CPU-first, no CUDA?

Because the deployment target dictates it. Edge hardware — phones, SBCs — is
**storage-rich but RAM/VRAM-poor**: there is plenty of flash to stream from and
almost no usable GPU memory. A CUDA-first design optimizes for exactly the
resource the target lacks. So the CPU is the primary compute target: everything
runs correctly and is optimized on CPU (hand-written ARM64 NEON kernels, scalar
fallbacks elsewhere), and the runtime is built and tested on a phone.

> [!WARNING] Vulkan is experimental — detection only
> The Vulkan backend is a secondary GPU-offload path, and today it is
> **stubbed**: `vulkan_matmul` always falls back to CPU and returns false, and
> the compute-shader dispatch is omitted even when the SPIR-V is embedded.
> `make VULKAN=1` enables device *detection* only, and `vulkan_available()`
> deliberately rejects CPU software rasterizers (llvmpipe). SipLLM claims no
> working GPU acceleration.

## Why is the int8 SDOT `--fast` path opt-in?

Because **the fp32-dequant path is the correctness oracle** and must stay the
default. `--fast` routes Q8_0 projections through an int8 SDOT kernel
(`matmul_q8_0_i8`, ARM-only, `__ARM_FEATURE_DOTPROD`) — the same int8-activation
technique llama.cpp uses — and closes most of the decode gap: Q8 `--fast`
resident hits smollm2 **62→171** and tinyllama **50** tok/s (vs llama.cpp 57).
But it quantizes the activation, so it is an *approximation*, not bit-identical.
Making it opt-in means every `1e-3`/bit-identical correctness test still runs
against the exact fp32 path, and a numerical regression cannot hide behind the
fast kernel. It is a demo dial you turn on knowingly, not a silent default. See
[Quantization](quantization.html).

## Related

<div class="card-grid"><div class="card"><h3>Where the numbers live</h3><p>Every figure here is committed measured data — see the <a href="performance-history.html">performance history</a> and <a href="benchmarks.html">benchmarks</a>.</p></div><div class="card"><h3>How it is bounded</h3><p>The mechanism behind the budget contract is in the <a href="memory-planner.html">memory planner</a> and <a href="layer-residency.html">layer residency</a>.</p></div><div class="card"><h3>The positioning</h3><p>Why "bounded > RAM" is a capability others do not target: <a href="competitive-analysis.html">competitive analysis</a>.</p></div></div>
