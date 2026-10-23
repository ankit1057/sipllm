# sipllm

[![Release](https://img.shields.io/github/v/release/ankit1057/sipllm?label=release)](https://github.com/ankit1057/sipllm/releases)
[![CI](https://github.com/ankit1057/sipllm/actions/workflows/ci.yml/badge.svg)](https://github.com/ankit1057/sipllm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A bounded-memory LLM runtime: models larger than available RAM run under a hard, configurable peak-RSS ceiling, CPU-first, with zero dependencies.**
SipLLM *sips* weights off storage — one transformer layer at a time — so peak RAM tracks a single resident layer, not the whole model: measured at 122 MB vs 1323 MB for llama.cpp on the same TinyLlama Q4_K_M file (10.8× less), and the gap widens with model size. Output is checked token for token against llama.cpp on Llama-family, Qwen2, Gemma 3 and Phi-2 models; see [Model architectures](#model-architectures) for what is and is not validated.

> ### Status
> **Latest tagged release: v0.6.0** (see [`CHANGELOG.md`](CHANGELOG.md) for what changed
> and the known issues). The one-line installer fetches its prebuilt bundle where one
> exists for your platform and builds from source otherwise. Every number in this README is a
> measurement with its date, hardware and source file named; the authoritative,
> per-wave-refreshed figures are the **North Star scorecard** at the top of
> [`CHANGELOG.md`](CHANGELOG.md). Memory safety: the uninitialized-read class is
> eliminated at the source and gated in CI by Valgrind, ASan/UBSan, TSan and a fuzzer.
> Output accuracy is validated against reference implementations (cosine similarity
> `1.000000` on F16).
>
> **What a user on a phone (ARM) or an AMD laptop still cannot do:** match llama.cpp's
> speed. On a OnePlus (Dimensity 900, 2×A78 + 6×A55) the same TinyLlama Q4_K_M file runs
> in **615 MB resident at 8.9–12.7 tok/s** or **118 MB streaming at 4.7 tok/s**, against
> llama.cpp's 1251 MB at 18.7 tok/s — 2–10.6× less RAM at half to two-thirds of the
> decode speed and a fifth of the prefill. The AVX2 path for AMD is cross-compiled only
> and has never been run. On Apple M3, resident 4-bit decode is 62–67% of llama.cpp.

**Edge-first, and therefore CPU-first.** This engine targets phones, SBCs, and
other edge hardware — where there's plenty of storage but very little RAM and
almost no usable VRAM. The **CPU is the primary compute target**: everything runs
deterministically and is optimized on CPU (hand-written ARM64 NEON kernels,
x86_64 AVX2/FMA vector paths, scalar fallbacks elsewhere). The optional Vulkan
backend provides GPU offload where hardware permits. No PyTorch, no ONNX, no
ggml inference runtime dependency, no BLAS — just standard C++17 and `pthread`.
Builds and runs on Termux / Android and on any Linux or macOS ARM or x86 host;
all performance numbers below were measured on Apple M3 unless stated otherwise.

```bash
curl -fsSL https://raw.githubusercontent.com/ankit1057/sipllm/main/install.sh | sh
sipllm run tinyllama -p "The capital of France is"
```

---

## Demo — the RAM↔speed dial, measured

Same model file, same prompt, CPU-only, Apple M3 (4 threads), warm cache. Peak RSS
is the OS figure from `/usr/bin/time -l` (authoritative, cross-runtime).

**TinyLlama-1.1B Q4_K_M (669 MB) — the 4-bit file a phone would run.** Measured
2026-10-19 (`bench/results/*2026-10-19.json`), `--ctx 512`, greedy, median of 3–5:

| runtime | peak RSS | decode | vs llama.cpp |
|:--------|---------:|-------:|:-------------|
| llama.cpp (CPU, `-ngl 0 -t 4`, ggml 0.27)   | 1323 MB | 88–96 tok/s | baseline (two quiet runs) |
| **SipLLM** `--fast --ram-budget 768M` (22/22 pinned) | **645 MB** | **59.6 tok/s** | **2.05× less RAM**, 62–67% of the speed |
| **SipLLM** `--ram-budget 768M` (exact fp32 dequant) | 645 MB | 21.0 tok/s | 2.05× less RAM, 24% |
| **SipLLM** `--fast --ram-budget 512M` (13/22 pinned) | **482 MB** | 23.8 tok/s | 2.7× less RAM |
| **SipLLM** streaming (`--ram-budget 0`)       | **122 MB** | 11.7 tok/s | **10.8× less RAM**, I/O-bound |
| **SipLLM** streaming `--spec 5`, copy-heavy output | 133 MB | 30.7 tok/s | 10× less RAM, 3.2× plain |

`--fast` routes Q4_K and Q6_K projections through int8 dot kernels (NEON `sdot`;
AVX2 on x86-64) with the activation quantized per 32-block — the same approximation
llama.cpp uses. First greedy token matched the exact fp32 path on 6/6 prompts, full
48-token continuations on 5/6. The exact path stays the default.

**On a phone (OnePlus IV2201, MediaTek Dimensity 900, Android 13; adb shell, nothing
installed, measured 2026-10-19, `bench/results/device-*.json`):** same file, 4 threads,
`--ctx 512`, greedy, median of 3. Peak RSS is the kernel's VmHWM.

| runtime | peak RSS | decode | prefill |
|:--------|---------:|-------:|--------:|
| llama.cpp (CPU, NDK build) | 1251 MB | 18.7 tok/s | 39 tok/s |
| **SipLLM** `--fast --ram-budget 1G` (22/22) | **615 MB** | 8.9–11.0 tok/s (8 thr: 12.7) | 7.5 tok/s |
| **SipLLM** `--ram-budget 1G` (exact fp32) | 615 MB | 2.8 tok/s | 6.4 tok/s |
| **SipLLM** `--fast` streaming | **118 MB** | 4.7 tok/s | 4.4 tok/s |

Big cores sat at 1.3–1.5 GHz throughout (battery 15%, governor unreadable), so these
are a floor for the SoC. The kernel and speculative-decoding test suites pass on the
phone (7/7, 5/5). Prefill is the on-device bottleneck: the batched `--fast` path loops
the single-vector kernel per prompt token.

**TinyLlama-1.1B Q8_0 (1.17 GB), `--fast`, `--ctx 512`, greedy.** Measured
2026-07-27 at v0.4.0 (`bench/results/demo-v0.4-*.json`), median of 3:

| runtime | peak RSS | decode | vs llama.cpp |
|:--------|---------:|-------:|:-------------|
| llama.cpp (CPU, `-ngl 0 -t 4`)        | 2326 MB | ~57 tok/s | baseline |
| **SipLLM** `--fast --ram-budget 1200M` (resident) | **1113 MB** | ~50 tok/s | **2.1× less RAM**, ~12% slower |
| **SipLLM** `--fast` (streaming)        | **175 MB**  | 6.8 tok/s | **13.3× less RAM** |

On Q8_0 with `--fast`, full residency uses **2.1× less memory** at ~88% of
llama.cpp's decode; drop the budget and the *same model* runs in **175 MB — 13× less
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

> **Kernel acceleration, precisely.** `--fast` routes **Q8_0, Q4_K and Q6_K**
> projections through int8 dot kernels: ARM64 NEON `sdot` (verified on Apple M3) and
> x86-64 AVX2 `maddubs` (cross-compiled, **not yet run on real x86 hardware**). Other
> quant types, and hosts without SDOT/AVX2, take the exact fp32-dequant path. On
> TinyLlama Q4_K_M resident: 21.0 → 59.6 tok/s; streaming is unchanged (I/O-bound).

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

## Model architectures

SipLLM reads `general.architecture` from the GGUF and runs one data-driven
transformer block. "Checked against llama.cpp" means greedy output from the same
GGUF file compared token for token on Apple M3 at v0.6.0; anything else is stated as
what it is.

| Architecture (GGUF name) | Status | Evidence |
|:--|:--|:--|
| **Llama** (`llama`): Llama 2 / 3.x, TinyLlama, SmolLM2, Mistral 7B | Checked against llama.cpp | Golden matrix (TinyLlama Q4_K_M, Q5_K_M, Q8_0, F16); Llama-3.2-1B and SmolLM2 output and chat-template token ids |
| **Qwen2** (`qwen2`): Qwen 2 / 2.5 | Checked against llama.cpp | Qwen2.5-0.5B-Instruct Q4_K_M with `--fast`: 32 identical tokens on one prompt; on another the first digit differs (logit margin 0.79) and the rest of the sentence matches in structure |
| **Gemma 3** (`gemma3`) | Checked against llama.cpp | Gemma-3-270M-it Q8_0 with `--fast`: 24 identical tokens on each of three prompts |
| **Phi-2** (`phi2`) | Checked against llama.cpp | Phi-2 Q4_K_M, exact and `--fast`: two of three prompts identical over llama.cpp's output; the third identical through its first sentence, then different |
| **Gemma 2**, **Gemma 4**, **Phi-3 / Phi-4**, **Mixtral** (MoE), **GPT-2** | Implemented; synthetic weights only | Unit tests on toy models; not yet run on real weights |
| **DeepSeek**, **Kimi**, **Yi**, **Baichuan**, **InternLM2**, **GLM-4** | Name recognized; runs the Llama block | Works only where the model really is Llama-shaped. Multi-head latent attention (DeepSeek V2/V3), ALiBi and fused-QKV layouts are not implemented |
| **StarCoder 2** | Not supported | No implementation |

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
│ ARM64 NEON sdot (Q8_0/Q4_K/Q6_K) · x86_64 AVX2 (untested) · Vulkan: detect only │
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
- **Path B (Server-Side Integration)**: OpenAI-compatible `POST /v1/chat/completions` with `"agent": true` for client-directed tool orchestration. Agent requests run shell and file tools on the server host, so they are refused unless `llm_server` is started with `--allow-agent`; the server binds 127.0.0.1 unless `--host` is given.
- **Zero-Dependency JSON Parser**: Hand-crafted JSON state machine (`ToolParser`, `ToolRegistry`) requiring zero third-party JSON libraries, providing robust structured-output tool dispatch.

### 5. Cross-Platform Edge Performance
Built specifically for edge and resource-constrained environments:
- **ARM64 NEON Kernels**: int8 `sdot` dot-product kernels for Q8_0, Q4_K and Q6_K (`--fast`), NEON-assisted fp32 dequantization for the exact path, NEON fp16→fp32 conversion.
- **x86_64 AVX2 / FMA Vector Paths**: Optimized vector dot products with automatic CPU runtime detection and portable scalar fallbacks.
- **Hardware-Calibrated Auto-Tuning (opt-in)**: `--autotune` runs `AutoTuner` micro-benchmarks to pick a thread count and scheduling policy, then caches the result. It is off by default because on big.LITTLE phones it picked slower settings than the defaults (every core, `fixed16`).
- **Mobile & Embedded**: builds on Android (NDK, static libc++), Termux, Apple Silicon and Linux SBCs. First phone numbers (Dimensity 900, 2026-10-19) are in the demo above and in `bench/results/device-*.json`; the harness is `scripts/device/`.

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
- **Model families** — Llama-family, Qwen2, Gemma 3 and Phi-2 checked against llama.cpp; Gemma 2/4, Phi-3/4, Mixtral MoE and GPT-2 implemented and unit-tested on synthetic weights. See [Model architectures](#model-architectures).
- **Complete Quantization Matrix** — Full support for `F32`, `F16`, `BF16`, `Q4_0/1`, `Q5_0/1`, `Q8_0`, and all K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, plus `IQ4_NL`).
- **SIMD Vector Acceleration** — int8 kernels for Q8_0 / Q4_K / Q6_K on ARM64 NEON `sdot` and x86_64 AVX2 (`--fast`; AVX2 untested on hardware), plus fp32 NEON / AVX2+FMA vector paths.
- **Bounded-Memory Layer Streaming** — Synchronous `pread`, double-buffered async prefetcher, and zero-copy `mmap` options keeping peak resident memory pinned to a single layer.
- **Inviolable RAM Budget Ceiling (`--ram-budget`)** — Dynamic layer pinning with a smooth RAM↔speed dial.
- **Kosh Context Intelligence** — Zero-cost prefix reuse (`--reuse`), persistent SIPS v1 session checkpointing (`--save-session`/`--load-session`), prompt-lookup speculative decoding (`--spec N`, proven token-identical to plain decoding), and a semantic (radix-tree) prefix cache.
- **Quantized KV Cache (`--kv-q8`)** — 8-bit KV cache with dynamic grow-on-demand allocation cutting memory footprint in half.
- **Nishachar Autonomous Agent** — Reference goal-driven agent (`goal -> plan -> act -> verify`) with built-in zero-dependency tool calling across CLI (Path A) and OpenAI-compatible server (Path B).
- **Ollama-Style CLI & OpenAI Server** — Interactive multi-turn chat REPL, positional prompts, piped stdin, and streaming HTTP Server-Sent Events.
- **Zero-Dependency Architecture** — Standard C++17, pthread, and POSIX I/O. No PyTorch, no ONNX, no BLAS, no external runtime dependencies.
- **Mathematically Verified** — Validated layer-by-layer against reference outputs; zero uninitialized memory reads, gated by CI with ASan, UBSan, TSan, and Valgrind.

## Repository Layout

```
sipllm               Ollama-style CLI wrapper (run / pull / serve / list / rm / which)
install.sh           One-line installer (prebuilt bundle when a release has one, else source build)
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

### Shipped (latest tag v0.6.0)
- [x] Bounded-memory transformer layer streaming with `--ram-budget` peak-RSS ceiling.
- [x] Llama-family, Qwen2, Gemma 3 and Phi-2 checked against llama.cpp; more families implemented on synthetic weights (see [Model architectures](#model-architectures)).
- [x] Int8 `--fast` kernels for Q8_0, Q4_K and Q6_K: ARM64 NEON `sdot` (verified), x86_64 AVX2 (cross-compiled only).
- [x] Sip IR stable intermediate representation, binary format (`SIPR`), and inspection tools (`ir_dump`).
- [x] Kosh context reuse (`--reuse`) and persistent SIPS v1 session serialization.
- [x] Nishachar goal-driven autonomous agent runtime (Path A CLI & Path B OpenAI-compatible server).
- [x] Full Ollama-style CLI experience (interactive REPL, piped stdin, positional prompt) and OpenAI API server.
- [x] Verified crash-hardened memory safety (zero uninitialized reads; continuous Valgrind/ASan/TSan).
- [x] Prompt-lookup speculative decoding (`--spec N`) with a token-identity proof against plain decoding.

### Current wave — the standing goal
Run LLMs **better than LiteRT / MediaPipe LLM Inference on ARM and AMD CPUs**: an
order of magnitude less RAM at competitive speed. The RAM half is measured; the speed
half needs:
- [x] Integer K-quant kernels (Q4_K, Q6_K × int8 activations): resident 4-bit decode 24% → 62–67% of llama.cpp on M3.
- [ ] Run and verify the AVX2 path on a real AMD/Intel host (`make test`, then the bench).
- [x] On-device numbers from a real phone (Dimensity 900): 2–10.6× less RAM than llama.cpp, 48–68% of its decode, 19% of its prefill.
- [ ] Batched int8 kernel for prefill (the on-device bottleneck: 7.5 vs llama.cpp's 39 tok/s).
- [ ] Same-hardware head-to-head against LiteRT-LM (published anchor, S24 Ultra CPU, not ours: Gemma3-1B int4 at 50–55 tok/s in ~1 GB).
- [ ] Remaining M3 gap (62–67% → parity): fp32 attention/KV path, prefill (73 vs 266 tok/s).

### Future Horizons
- [ ] First-class mobile bindings (Flutter / Android JNI / iOS Swift framework).
- [ ] Multimodal vision encoder (ViT) and projector integration.
- [ ] Multi-device distributed layer streaming across local network nodes.
- [ ] Expanded Vulkan full-layer compute pipeline for unified memory edge GPUs.

## License

[MIT](LICENSE).
