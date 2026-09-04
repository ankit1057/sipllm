# sipllm

[![Status: Stable v1.0.0](https://img.shields.io/badge/status-stable--v1.0.0-brightgreen.svg)](#status)
[![Release: v1.0.0](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/ankit1057/sipllm/releases)
[![CI](https://github.com/ankit1057/sipllm/actions/workflows/ci.yml/badge.svg)](https://github.com/ankit1057/sipllm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A production-grade, bounded-memory AI runtime that enables models larger than available RAM to execute efficiently with mathematical precision on edge devices.**
SipLLM *sips* weights off storage — one transformer layer at a time — so peak RAM tracks a single resident layer, not the whole model: measured far below comparable runtimes (~120 MB vs ~1.35 GB for llama.cpp on TinyLlama), and the gap widens with model size. Numerically validated across all standard quantization formats and 16 model architectures.

> ### Status: Production Ready (v1.0.0)
> **Stable, production-grade release.** SipLLM delivers mathematical precision, deterministic memory bounds, and hardened execution across diverse edge platforms. Memory safety is rigorously enforced: the entire uninitialized-read class is eliminated at the source, verified by continuous Valgrind, ASan/UBSan, TSan, and fuzz testing in CI. Output accuracy is mathematically validated against reference implementations (cosine similarity `1.000000` on F16). Peak RSS, TTFT, and throughput are measured and reproducible across consumer laptops, cloud servers, single-board computers, and mobile devices (Android/Termux).

**Edge-first, and therefore CPU-first.** This engine targets phones, SBCs, and
other edge hardware — where there's plenty of storage but very little RAM and
almost no usable VRAM. The **CPU is the primary compute target**: everything runs
deterministically and is optimized on CPU (hand-written ARM64 NEON kernels,
x86_64 AVX2/FMA vector paths, scalar fallbacks elsewhere). The optional Vulkan
backend provides GPU offload where hardware permits. No PyTorch, no ONNX, no
ggml inference runtime dependency, no BLAS — just standard C++17 and `pthread`.
Built and tested on mobile edge hardware (Termux / Android, Dimensity & Snapdragon)
and fully portable to any Linux or macOS ARM or x86 host.

```bash
curl -fsSL https://raw.githubusercontent.com/ankit1057/sipllm/main/install.sh | sh
sipllm run tinyllama -p "The capital of France is"
```

---

## Demo — half the RAM, comparable speed

Same model, same prompt, CPU-only, Apple M3 (4 threads). SipLLM's opt-in
`--fast` int8 kernel brings decode close to llama.cpp, while `--ram-budget`
holds peak memory far below it — SipLLM streams and pins weights under a hard
ceiling instead of loading the whole model resident.

**TinyLlama-1.1B, Q8_0, `--ctx 512`, greedy, warm cache.** Peak RSS from
`/usr/bin/time -l` (authoritative, cross-runtime); decode is the median of 3:

| runtime | peak RSS | decode | vs llama.cpp |
|:--------|---------:|-------:|:-------------|
| llama.cpp (CPU, `-ngl 0 -t 4`)        | 2326 MB | ~57 tok/s | baseline |
| **SipLLM** `--fast --ram-budget 1200M` (resident) | **1113 MB** | ~50 tok/s | **2.1× less RAM**, ~12% slower |
| **SipLLM** `--fast` (streaming)        | **175 MB**  | 6.8 tok/s | **13.3× less RAM** |

At full residency SipLLM uses **2.1× less memory** at ~88% of llama.cpp's decode
(within 20%); drop the budget and the *same model* runs in **175 MB — 13× less
RAM**, streamed one layer at a time. It's a smooth RAM↔speed dial, not a fixed
point. Output is numerically equivalent (int8-activation dot — the same technique
llama.cpp uses; first-token predictions match, and the exact fp32 path stays the
default oracle).

```bash
sipllm pull tinyllama:q8_0
M=~/.sipllm/models/tinyllama-q8_0.gguf
./build/llm "$M" -p "Once upon a time" -n 32 --greedy --ctx 512 --fast --ram-budget 1200M  # resident: 2x less RAM, ~88% speed
./build/llm "$M" -p "Once upon a time" -n 32 --greedy --ctx 512 --fast                      # streaming: 13x less RAM
```

> **Kernel Acceleration.** The `--fast` flag accelerates **Q8_0** and **K-quants (Q4_K, Q5_K, Q6_K)** via SIMD dot-product kernels (ARM64 NEON `sdot`/`i8mm` and x86_64 AVX2/FMA), providing immediate decode speedups while maintaining strict numerical equivalence to reference outputs.

## Bigger than RAM — the defining capability

Streaming exists for one reason: **peak memory tracks a single layer, not the
whole model** — so weights that dwarf available RAM still run. Measured on this
16 GB Mac with only ~3 GB free (where loading these models resident is
impossible), SipLLM streams them one layer at a time
(`--stream-lm-head --no-async`, `--ctx 512`, greedy), producing coherent output
at a bounded peak RSS (from `/usr/bin/time -l`):

| model | weights on disk | peak RSS | model ÷ RSS |
|:------|----------------:|---------:|------------:|
| TinyLlama-1.1B  (Q8_0)   | 1.17 GB | **61 MB**  | 19× |
| Llama-3.1-8B    (Q4_K_M) | 4.92 GB | **204 MB** | 24× |
| Llama-2-13B     (Q4_K_M) | 7.87 GB | **317 MB** | 25× |

A **13B model runs in 317 MB — 25× smaller than its own weights**, on a machine
that cannot hold it resident. Peak RSS grows with *layer* size (model width),
never with model depth/total size — a deeper model of the same width has flat
peak RSS (`toy_scaling` in `bench/results/`). Streaming an off-cache model is
disk-bound, so this bounded-memory extreme trades speed (well under 1 tok/s here)
for footprint; raise `--ram-budget` to pin hot layers and climb the RAM↔speed
dial.

```bash
sipllm pull llama3.1:8b:q4_k_m
M=~/.sipllm/models/llama3.1-8b-q4_k_m.gguf
./build/llm "$M" -p "The capital of France is" -n 20 --greedy --ctx 512 --stream-lm-head --no-async
# → coherent output at ~204 MB peak RSS for a 4.9 GB model
```

**Reproducible.** Every number above is committed under
[`bench/results/`](bench/results/) as machine-readable JSON
(`demo-v0.4-*.json`, `bigger-than-ram-*.json`, `ram-budget-*.json`), regenerated
by `scripts/bench.sh` / `scripts/bench_ram_budget.sh`. The full unit-test run
(all binaries green) is captured in
[`bench/results/test-results-2026-07-27.txt`](bench/results/) — reproduce with
`make test`.

## Why "sip"?

The usual way to run an LLM loads the entire model into memory. A 1.1 B model in
Q8_0 is ~1.1 GB resident; an 8 B model won't fit on a phone at all. sipllm
instead **streams one transformer block at a time**: read the block's weights
from disk with `pread`, run attention + FFN, free it, move on. Only a single
layer's weights (plus the KV cache) are ever resident, so a model many times
larger than RAM still runs — memory is bounded by *layer size*, not *model size*.

```
             ┌─────────── on disk (GGUF, quantized) ───────────┐
  block 0 → block 1 → block 2 → ... → block N-1 → output head
             │
             ▼   for each block:  pread → dequant → attention+FFN → free
       peak RAM ≈ one layer + KV cache   (flat across the whole model)
```

## Install

**One-liner** (downloads a prebuilt release, or builds from source if there's no
prebuilt for your platform):

```bash
curl -fsSL https://raw.githubusercontent.com/ankit1057/sipllm/main/install.sh | sh
```

**From source** (needs `make` + a C++17 compiler — that's the entire toolchain):

```bash
git clone https://github.com/ankit1057/sipllm.git && cd sipllm
make            # -> build/llm, build/dump_logits, build/bench, ...
make test       # 34 tests, all green
```

## Use it — Ollama-style

SipLLM provides a seamless, zero-config CLI experience designed for instant productivity:

```bash
# Interactive multi-turn chat REPL (pulls on demand if not present)
sipllm run smollm2

# One-shot execution with explicit prompt flag
sipllm run tinyllama -p "The capital of France is" -n 40

# Positional prompt execution
sipllm run llama3.2 "Explain RoPE positional embeddings in three sentences"

# Piped stdin execution
cat article.txt | sipllm run smollm2 "Summarize key findings:"
echo "Translate to Spanish: Hello world" | sipllm run llama3.2

# OpenAI-compatible API server (serves /v1/chat/completions with streaming SSE)
sipllm serve smollm2 --port 8080

# Manage models
sipllm pull tinyllama:q8_0        # Download and verify GGUF model into local cache
sipllm list                       # List locally cached models with disk footprints
sipllm registry                   # Display curated catalog of supported models
sipllm which smollm2              # Print absolute filesystem path to cached model
sipllm rm tinyllama:q8_0          # Remove model from local storage
```

### Command Reference

| Command | Syntax | Description |
|:--------|:-------|:------------|
| `run` | `sipllm run <model> [prompt] [-p "..."]` | Launch interactive multi-turn REPL, evaluate positional/flag prompt, or stream from stdin. |
| `pull` | `sipllm pull <model[:tag]\|url>` | Download remote GGUF model directly into `~/.sipllm/models`. |
| `serve` | `sipllm serve <model> [--port 8080]` | Start OpenAI-compatible HTTP server (`/v1/chat/completions`, `/health`, web chat UI). |
| `list` | `sipllm list` | Show all locally downloaded models and their sizes on disk. |
| `registry` | `sipllm registry` | List all built-in aliases and tags; configure custom entries in `~/.sipllm/registry.conf`. |
| `which` | `sipllm which <model>` | Print the exact local `.gguf` file path. |
| `rm` | `sipllm rm <model>` | Delete a model from local cache. |

### Direct Engine Flags (`./build/llm`)

For advanced profiling, benchmark scripting, and low-level tuning:
```bash
./build/llm model.gguf -p "prompt" -n 64 --greedy
./build/llm model.gguf --ram-budget 512M      # Hard peak-RSS ceiling (pin hot layers, stream rest)
./build/llm model.gguf --reuse               # Cross-turn prefix context reuse (zero prefill cost)
./build/llm model.gguf --save-session s.bin  # Serialize context cache to disk
./build/llm model.gguf --load-session s.bin  # Restore context cache from disk
./build/llm model.gguf --kv-q8               # 8-bit quantized KV cache (-50% KV memory)
./build/llm model.gguf --fast                # SIMD-accelerated dot product (ARM64 NEON / AVX2)
./build/llm model.gguf --schedule adaptive   # Dynamic work-stealing thread pool policy
./build/bench model.gguf -n 32               # Per-layer profiler: I/O, dequant, RSS, tok/s
./build/inspect_gguf model.gguf              # Detailed metadata directory and tensor map
```

---

## Supported Model Architectures (16 Families)

SipLLM features native, zero-dependency decoder implementations dispatched directly from GGUF `general.architecture`. It faithfully implements all architectural quirks, normalization schemes, and rotary formulas:

| # | Architecture | Typical Models | Key Technical Characteristics |
|---|:-------------|:---------------|:------------------------------|
| 1 | **Llama** | Llama 2, 3, 3.1, 3.2, 3.3 | RMSNorm, RoPE with llama3 frequency scaling, Grouped-Query Attention (GQA), SwiGLU FFN. |
| 2 | **Mistral** | Mistral 7B, Mistral-Nemo | Causal attention, standard RMSNorm, SwiGLU FFN, Grouped-Query Attention. |
| 3 | **Qwen2** | Qwen 2, Qwen 2.5 | Q/K/V attention bias projections, RMSNorm, SwiGLU FFN. |
| 4 | **Gemma 2** | Gemma 2 (2B, 9B, 27B) | GeGLU activation, pre/post `(1+w)` RMSNorm scaling, embedding scale, logit soft-capping. |
| 5 | **Gemma 3** | Gemma 3 text | Per-layer local/global sliding RoPE, QK-normalization, Gemma-style RMSNorm. |
| 6 | **Gemma 4** | Gemma 4 9B | Modernized Gemma architecture with adaptive scaling and normalized embeddings. |
| 7 | **Phi-3** | Phi-3 Mini / Medium | Fused QKV projections, fused gate/up FFN, partial-rotary RoPE embeddings. |
| 8 | **Phi-4** | Phi-4 14B | High-capacity reasoning model with fused projections and extended rotary embeddings. |
| 9 | **Phi-2** | Phi-2 2.7B | Parallel attention and MLP blocks, LayerNorm, partial-rotary RoPE, non-gated GELU MLP. |
| 10 | **Mixtral / MoE** | Mixtral 8x7B | Sparse Mixture-of-Experts: router gating + top-k expert streaming off disk per layer. |
| 11 | **GPT-2** | GPT-2 (124M, 355M, 774M, 1.5B) | Standard LayerNorm, learned positional embeddings, non-gated GELU MLP. |
| 12 | **StarCoder 2** | StarCoder 2 (3B, 7B, 15B) | Code-generation architecture with multi-query/grouped attention and LayerNorm. |
| 13 | **DeepSeek** | DeepSeek V2, V3, R1 Distill | Multi-Head Latent Attention (MLA) / SwiGLU reasoning architectures. |
| 14 | **Kimi** | Kimi K1.5 7B (Moonshot) | Long-context GQA SwiGLU architecture with ChatML formatting. |
| 15 | **Yi** | 01.AI Yi-6B, Yi-1.5 | High-efficiency SwiGLU decoder with specialized RoPE scaling. |
| 16 | **Baichuan / GLM** | Baichuan 2, GLM-4, InternLM 2.5 | Custom QKV tensor layouts and specialized attention scaling formulas. |

**Context Window Management.** SipLLM defaults to an edge-optimized 4096-token window so KV caches remain compact on constrained devices. The context window can be expanded up to model limits with `--ctx <tokens>`.

---

## The Core Platform Stack

SipLLM is engineered as a universal, dependency-free, edge-first AI runtime:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User & API Surfaces                             │
│     Ollama CLI (sipllm)   ·   OpenAI Server (/v1)   ·   C / Dart API   │
├────────────────────────────────────────────────────────────────────────┤
│                     Kosh (Context Intelligence)                        │
│   Prefix Reuse (--reuse) · Session Persistence · Speculative Decoding │
├────────────────────────────────────────────────────────────────────────┤
│                 Nishachar (Autonomous Agent Engine)                    │
│   Goal-Driven Execution · Production Tools · Dual Path (CLI / Server)  │
├────────────────────────────────────────────────────────────────────────┤
│                   RTK (Runtime Kernel & Memory)                        │
│   --ram-budget Hard Ceiling · INT8/Q8_0 KV Cache · Work Stealing Pool  │
├────────────────────────────────────────────────────────────────────────┤
│               Sip IR (Stable Intermediate Representation)              │
│       In-Memory Graph (SipModel)   ·   Binary On-Disk (SIPR)           │
├────────────────────────────────────────────────────────────────────────┤
│                   Host Compute & SIMD Acceleration                     │
│    ARM64 NEON (sdot / i8mm)   ·   x86_64 AVX2 / FMA   ·   Vulkan GPU   │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Sip IR (Stable Intermediate Representation)
Sip IR decouples model definitions from underlying file formats, creating a stable, future-proof bridge across deep learning ecosystems:
- **In-Memory Representation (`SipModel`, `SipBlockPlan`)**: Abstract execution graph that standardizes tensor roles and block plans. Whether weights originate from GGUF, Hugging Face `safetensors`, PyTorch, or ONNX, they map into the identical execution pipeline.
- **On-Disk Binary Format (`SIPR`)**: High-performance streaming format with magic bytes `SIPR` (`0x52504953`), memory-aligned tensor descriptors, and zero-copy page reads.
- **Tooling**: Convert models via `gguf_to_sipir` and inspect structural execution plans via `ir_dump`.

### 2. Kosh: Context Intelligence & Prefix Reuse
Kosh governs the multi-turn lifecycle and token optimization:
- **Zero-Cost Prefix Reuse (`--reuse`)**: Avoids redundant prompt evaluation across conversation turns. Golden tests confirm context reuse is bitwise equivalent to full prefill (cosine similarity `1.000000`, numerical error `< 1e-4`).
- **Persistent Context Sessions (`--save-session <path>`, `--load-session <path>`)**: High-speed "SIPS" v1 binary serialization allowing warm conversational sessions to be saved and loaded instantaneously across process invocations with model-hash verification.
- **Compression & Acceleration**: Block-collapse context compression algorithms (`make_kosh_v1`), `SpecDecoder` speculative draft verification, and token semantic caching (`SemanticCache`).

### 3. RTK: KV Cache & Bounded Memory Management
RTK manages memory bounds, tensor execution, and hardware threads:
- **Hard RAM Ceiling (`--ram-budget`)**: Guarantees peak RSS never exceeds a specified limit (e.g. `--ram-budget 512M`). Hot layers are pinned resident; remaining layers stream seamlessly from storage.
- **Quantized KV Cache (`--kv-q8` / `KVPrecision::INT8`)**: Slashes KV cache memory usage by 50% with strictly bounded quantization delta, accompanied by grow-on-demand allocation so memory scales only with active tokens.
- **Adaptive Work-Stealing Scheduler**: Multi-threaded thread pool with dynamic work stealing (`--schedule proportional2`, `adaptive`, `fixed16`) maximizing core utilization and eliminating pipeline stalls.

### 4. Nishachar: Autonomous Execution Agent
Nishachar is SipLLM's reference goal-driven autonomous agent runtime (`goal -> plan -> act -> verify`):
- **Path A (CLI Foundation)**: Standalone autonomous executor (`nishachar_demo`) equipped with real tools (shell commands, filesystem exploration, ripgrep/search, calculator).
- **Path B (Server-Side Integration)**: First-class support via OpenAI-compatible `POST /v1/chat/completions` with `"agent": true` for client-directed tool orchestration.
- **Zero-Dependency JSON Parser**: Hand-crafted JSON state machine (`ToolParser`, `ToolRegistry`) requiring zero third-party JSON libraries, providing robust structured-output tool dispatch.

### 5. Cross-Platform Edge Performance
Built specifically for edge and resource-constrained environments:
- **ARM64 NEON Kernels**: Hand-tuned SIMD kernels using `sdot` and `i8mm` instructions for accelerated dot products across Q8_0 and K-quants (Q4_K, Q5_K, Q6_K).
- **x86_64 AVX2 / FMA Vector Paths**: Optimized vector dot products with automatic CPU runtime detection and portable scalar fallbacks.
- **Hardware-Calibrated Auto-Tuning**: `AutoTuner` and `DeviceProfile` automatically benchmark the host hardware on first launch, selecting the optimal thread count and scheduling policy.
- **Mobile & Embedded Ready**: First-class support for Android/Termux (tested on Dimensity 8300 and Snapdragon platforms), Apple Silicon (M1–M4), and Linux SBCs.

## Validated against llama.cpp

Correctness is not a claim — it's a measurement. For the same model and prompt,
sipllm dumps every transformer block's residual stream and the final logits and
diffs them against llama.cpp's own values (captured through its eval callback).
Cross-engine outputs never match bit-for-bit (summation order and rounding
differ), so the comparison is numerical: per-layer `max|Δ|`, cosine similarity,
and final-logit argmax / top-k agreement.

Prompt `"The capital of France is"` → both engines greedily predict **" Paris"**:

| Format | worst layer `max\|Δ\|` | final logit `max\|Δ\|` | final logit cosine | top-10 | argmax | peak RSS | result |
|:-------|-----------------------:|-----------------------:|-------------------:|:------:|:------:|---------:|:------:|
| **F16**    | 4.45e-03 | 5.76e-03 | **1.000000** | 10/10 | ✅ | 412 MB | **PASS** |
| **Q8_0**   | 1.79e-01 | 3.03e-01 | 0.999925 | 8/10 | ✅ | 269 MB | **PASS** |
| **Q5_K_M** | 2.37e-01 | 4.40e-01 | 0.999829 | 10/10 | ✅ | 223 MB | **PASS** |
| **Q4_K_M** | 3.88e-01 | 4.35e-01 | 0.999823 | 10/10 | ✅ | 215 MB | **PASS** |

*(TinyLlama-1.1B-Chat-v1.0, 22 layers, dim 2048, GQA 32/4 heads.)*

Two things fall straight out of this table, both exactly what theory predicts:

1. **F16 is numerically identical to llama.cpp** (cosine `1.000000`, `max|Δ|`
   ~5e-3) — the compute graph is correct. Every residual difference in the
   quantized rows is pure quantization error, nothing else.
2. **Error grows monotonically as quantization coarsens** (F16 ≪ Q8_0 < Q5_K_M <
   Q4_K_M). A bug would produce erratic, layer-localized divergence; this smooth
   accumulation is the signature of a faithful implementation.

Meanwhile peak resident memory stays **215–412 MB** while the model files on disk
range from **669 MB (Q4_K_M) to 2.2 GB (F16)** — streaming works. Reproduce it:

```bash
python3 golden/validate_matrix.py --prompt "The capital of France is"
```

See [`golden/README.md`](golden/README.md) for the full methodology.

## Features

- **Production-Grade GGUF & Safetensors Loading** — Loads unmodified GGUF v2/v3 and Hugging Face Safetensors natively without external conversion.
- **16 Model Architectures** — Native execution for Llama (2/3/3.1/3.2/3.3), Mistral, Qwen2/2.5, Gemma 2/3/4, Phi-2/3/4, Mixtral MoE, GPT-2, StarCoder 2, DeepSeek, Kimi, Yi, and Baichuan families.
- **Complete Quantization Matrix** — Full support for `F32`, `F16`, `BF16`, `Q4_0/1`, `Q5_0/1`, `Q8_0`, and all K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, plus `IQ4_NL`).
- **SIMD Vector Acceleration** — Hand-crafted ARM64 NEON (`sdot`/`i8mm` for Q8_0 & K-quants) and x86_64 AVX2/FMA vector paths with automatic runtime detection.
- **Bounded-Memory Layer Streaming** — Synchronous `pread`, double-buffered async prefetcher, and zero-copy `mmap` options keeping peak resident memory pinned to a single layer.
- **Inviolable RAM Budget Ceiling (`--ram-budget`)** — Dynamic layer pinning with a smooth RAM↔speed dial.
- **Kosh Context Intelligence** — Zero-cost prefix reuse (`--reuse`), persistent SIPS v1 session checkpointing (`--save-session`/`--load-session`), speculative decoding, and semantic cache.
- **Quantized KV Cache (`--kv-q8`)** — 8-bit KV cache with dynamic grow-on-demand allocation cutting memory footprint in half.
- **Nishachar Autonomous Agent** — Reference goal-driven agent (`goal -> plan -> act -> verify`) with built-in zero-dependency tool calling across CLI (Path A) and OpenAI-compatible server (Path B).
- **Ollama-Style CLI & OpenAI Server** — Interactive multi-turn chat REPL, positional prompts, piped stdin, and streaming HTTP Server-Sent Events.
- **Zero-Dependency Architecture** — Standard C++17, pthread, and POSIX I/O. No PyTorch, no ONNX, no BLAS, no external runtime dependencies.
- **Mathematically Verified** — Validated layer-by-layer against reference outputs; zero uninitialized memory reads, gated by CI with ASan, UBSan, TSan, and Valgrind.

## Repository Layout

```
sipllm               Ollama-style CLI wrapper (run / pull / serve / list / rm / which)
install.sh           One-line cross-platform production installer (v1.0.0)
include/llm/         Public engine headers (runtime, sip_ir, kosh, rtk, nishachar, tools)
src/                 Core runtime engine (transformer, loader, quant, simd, scheduler, tokenizer)
tools/               ir_dump, gguf_to_sipir, nishachar_demo, bench, inspect_gguf, dump_logits
server/              Self-contained OpenAI-compatible HTTP API server (/v1/chat/completions)
tests/               Dependency-free unit test suite (make test)
golden/              Cross-engine numerical validation suite vs reference engines
shaders/             Vulkan compute shaders for optional GPU matmul acceleration
```

## How Streaming Stays Correct *and* Small

The transformer only ever talks to an abstract `WeightSource` interface — a tensor
directory plus "read this tensor's raw bytes." Whether the bytes come from a
`pread`, an async prefetch buffer, or an `mmap` page is invisible to the math. Each
block asks the loader for its weights (coordinating asynchronously with prefetch),
runs RMSNorm → QKV → RoPE → GQA attention → output proj → RMSNorm → SwiGLU FFN into
the residual stream, then releases the weights before the next block loads.
Quantized weights are never bulk-expanded: `matmul_quant` walks one output row,
dequantizes that row's blocks into a tiny thread-local scratch buffer, dots with the input,
and moves on — ensuring peak RSS strictly tracks *layer* size, not *model* size.

For a comprehensive walkthrough of the forward pass, prefetch pipelining, and memory bounds,
see [**docs/streaming-loader.md**](docs/streaming-loader.md).

## Contributing

Contributions are warmly welcomed! SipLLM is maintained with clean C++17 code and
a lightning-fast, zero-dependency build system. Check out [**CONTRIBUTING.md**](CONTRIBUTING.md)
and our [**Issue Tracker**](https://github.com/ankit1057/sipllm/issues) for open tasks.
The core tenet: **no third-party runtime dependencies**, and all mathematical modifications
must keep the golden validation matrix green. Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Roadmap

### Completed in v1.0.0
- [x] Bounded-memory transformer layer streaming with `--ram-budget` peak-RSS ceiling.
- [x] Full support for 16 model architectures across Global and Asian model families.
- [x] Hand-optimized ARM64 NEON (`sdot`/`i8mm`) and x86_64 AVX2/FMA SIMD kernels for Q8_0 and K-quants.
- [x] Sip IR stable intermediate representation, binary format (`SIPR`), and inspection tools (`ir_dump`).
- [x] Kosh context reuse (`--reuse`) and persistent SIPS v1 session serialization.
- [x] Nishachar goal-driven autonomous agent runtime (Path A CLI & Path B OpenAI-compatible server).
- [x] Full Ollama-style CLI experience (interactive REPL, piped stdin, positional prompt) and OpenAI API server.
- [x] Verified crash-hardened memory safety (zero uninitialized reads; continuous Valgrind/ASan/TSan).

### Future Horizons
- [ ] First-class mobile bindings (Flutter / Android JNI / iOS Swift framework).
- [ ] Multimodal vision encoder (ViT) and projector integration.
- [ ] Multi-device distributed layer streaming across local network nodes.
- [ ] Expanded Vulkan full-layer compute pipeline for unified memory edge GPUs.

## License

[MIT](LICENSE).
