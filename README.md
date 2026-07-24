# sipllm

[![Status: WIP](https://img.shields.io/badge/status-work_in_progress-orange.svg)](#status)
[![CI](https://github.com/ankit1057/sipllm/actions/workflows/ci.yml/badge.svg)](https://github.com/ankit1057/sipllm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A dependency-free, CPU-first LLM inference engine in C++17 that runs models
*larger than RAM*. It *sips* weights off disk — one transformer layer at a time —
so peak RAM tracks a single resident layer, not the whole model: measured far
below comparable runtimes (~120 MB vs ~1.35 GB for
[llama.cpp](https://github.com/ggml-org/llama.cpp) on TinyLlama), and the gap
widens with model size. Numerically validated against llama.cpp, layer by layer,
across four GGUF quantization formats.**

> ### Status
> **Actively developed.** The runtime is crash-hardened — the whole
> uninitialized-read class is eliminated at source, with valgrind, ASan/UBSan,
> TSan, and a prompt/config fuzzer gating CI — and numerically validated against
> llama.cpp (see the matrix below). Performance is measured, not guessed: a
> reproducible harness (`scripts/bench.sh`) tracks peak RSS, TTFT, and decode
> tok/s. Expect API/CLI evolution; contributions and issue reports are welcome.

**Edge-first, and therefore CPU-first.** This engine targets phones, SBCs, and
other edge hardware — where there's plenty of storage but very little RAM and,
let's be honest, almost no usable VRAM. So the **CPU is the primary compute
target**: everything runs correctly and is optimized on CPU (hand-written ARM64
NEON kernels, scalar fallbacks elsewhere). The Vulkan backend is **experimental**
and secondary — a GPU offload path for the rare edge device that has one, not a
requirement. No PyTorch, no ONNX, no ggml for inference, no BLAS — just standard
C++17 and `pthread`. Built and tested on a phone (Termux / Android, Dimensity
8300) and portable to any Linux/macOS ARM or x86 host.

```bash
curl -fsSL https://raw.githubusercontent.com/ankit1057/sipllm/main/install.sh | sh
sipllm run tinyllama -p "The capital of France is"
```

---

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

```bash
sipllm run smollm2  -p "The capital of India is" -n 40     # pulls on first use
sipllm run llama3.2 -p "Write a haiku about the sea"       # Llama-3.2-1B-Instruct
sipllm run llama3.2:3b -p "Explain RoPE briefly"           # bigger, still streams
sipllm serve smollm2 --port 8080                           # browser chat UI
sipllm pull tinyllama:q8_0                                 # just download
sipllm list                                                # local models
sipllm registry                                            # what's available
```

**Bundled models** (all public / ungated, all Llama-architecture — what the
engine implements today): `llama3.2` (Llama-3.2-1B/`:3b`-Instruct — verified
end-to-end here), `smollm2` (SmolLM2-1.7B-Instruct / `:360m` — standard Llama
arch, strong for its size), and `tinyllama` (1.1B — tiny and prone to rambling;
fine for smoke tests). Each takes quant tags, e.g. `smollm2:q8_0`,
`tinyllama:q4_k_m`. Bigger *instruct* models hallucinate far less than TinyLlama.

Model names resolve to public GGUF files and cache in `~/.sipllm/models`. You can
also pass any GGUF **URL** or **local path** directly, or add your own names to
`~/.sipllm/registry.conf` (`name<TAB>url`). Under the hood the engine streams the
file, so `sipllm run` on a model bigger than your RAM just works.

**Context window.** The engine defaults to a 4096-token window so the KV cache
stays small on edge devices — models that advertise huge windows (Llama 3.2 =
131072) would otherwise allocate multi-GB caches up front. Raise it explicitly
with `./build/llm model.gguf --ctx 32768 ...` when you have the RAM.

> **Architecture note.** The engine dispatches on `general.architecture` and now
> implements, besides the **Llama** reference (RMSNorm + RoPE + GQA + SwiGLU):
> **Mistral**, **Qwen2/2.5** (q/k/v bias), **Gemma 2** (GeGLU, pre/post
> `(1+w)` norms, embedding scale, logit soft-capping), **Gemma 3 text** (QK-norm
> + per-layer local/global RoPE), **Phi-3** (fused QKV / gate-up, partial-rotary
> RoPE), **Phi-2** and **GPT-2** (LayerNorm, biases; GPT-2 with learned position
> embeddings), and **Mixtral / MoE** (router + top-k experts, streamed). Llama
> 3.x "llama3" RoPE scaling is applied. Cross-engine golden validation
> (`golden/validate_matrix.py`) currently covers the Llama path; the newer
> architectures are unit-tested here and validated against llama.cpp as models
> are added. Vision/audio (Gemma 3n) and the GPT-NeoX/StableLM/Falcon variants
> remain [tracked](https://github.com/ankit1057/sipllm/issues).

Prefer the raw engine? It takes a model path directly:

```bash
./build/llm model.gguf -p "prompt" -n 40
./build/bench model.gguf -n 32          # per-layer profiler: I/O, dequant, RSS, tok/s
./build/inspect_gguf model.gguf         # metadata + tensor directory
```

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

- **Real GGUF v2/v3 parser** — loads unmodified files from Hugging Face.
- **Broad quantization support** — dequant for `F32`, `F16`, `BF16`, `Q4_0/1`,
  `Q5_0/1`, `Q8_0`, and K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`.
  A **fused quantized matmul** dequantizes one weight row into a small scratch
  buffer and dots it with the activation — a whole layer stays quantized in RAM.
- **Streaming layer loader** — synchronous `pread`, an **async double-buffered
  prefetcher** (compute layer *N* while layer *N+1* loads), or an **`mmap`
  backend** — switchable and benchmarked side by side.
- **Correct transformer** — RMSNorm, RoPE (ggml-compatible), Grouped-Query
  Attention, SwiGLU FFN, causal KV cache.
- **Tokenizers** — SentencePiece (Llama) and byte-level BPE (GPT-2 / Llama-3),
  decoded straight from the GGUF metadata.
- **ARM64 NEON kernels** — `sdot` / `i8mm` accelerated, with scalar fallbacks.
- **Ollama-style CLI + one-line installer**, a **terminal profiler**, a
  **web chat server** (`make server`), and an **optional Vulkan** matmul backend.
- **Zero-dependency test suite** — 34 unit tests, no gtest/catch2 (`make test`).

## Repository layout

```
sipllm         Ollama-style CLI wrapper (pull / run / serve / list)
install.sh     one-line installer
include/llm/   engine headers (public API surface)
src/           gguf parser, dequant + fused matmul, transformer, tokenizer,
               streaming loader, kv cache, sampler, neon kernels
tools/         dump_logits, bench, inspect_gguf, make_toy_model, gguf_to_f16
tests/         dependency-free unit tests  (make test)
golden/        cross-engine validation vs llama.cpp (the matrix above)
server/        self-contained HTTP chat server  (make server)
shaders/       Vulkan compute shader for the optional GPU matmul backend
```

## How streaming stays correct *and* small

The transformer only ever talks to a `WeightSource` interface — a tensor
directory plus "read this tensor's raw bytes." Whether the bytes come from a
`pread`, a prefetch buffer, or an `mmap` page is invisible to the math. Each
block asks the loader for its weights (possibly blocking on the prefetch), runs
RMSNorm → QKV → RoPE → GQA attention → output proj → RMSNorm → SwiGLU FFN into
the residual stream, then releases the weights before the next block loads.
Quantized weights are never bulk-expanded: `matmul_quant` walks one output row,
dequantizes that row's blocks into a tiny buffer, dots with the input, and moves
on — which is why peak RSS tracks *layer* size, not *model* size.

For a step-by-step walkthrough of one forward pass — the `WeightSource` seam,
the async double-buffer prefetch, and why peak RSS is flat — with a diagram, see
[**docs/streaming-loader.md**](docs/streaming-loader.md).

## Contributing

Contributions are very welcome — it's a small, readable codebase with a fast,
dependency-free build, which makes it a great project to learn on. Start with
[**CONTRIBUTING.md**](CONTRIBUTING.md) and the
[**good first issues**](https://github.com/ankit1057/sipllm/labels/good%20first%20issue)
(new quant formats, x86 SIMD, sampler features, more registry models, docs). The
one hard rule: **no third-party runtime dependencies**, and changes to the math
must keep the golden matrix green. Be kind — see the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Roadmap

- Android JNI/NDK packaging (Termux-first, already the primary target).
- Prebuilt x86_64 + macOS release binaries via CI.
- Wider prefetch pipelining and NEON coverage of the K-quant dequant paths.
- Maturing the Vulkan backend from matmul offload to full-layer offload.

## License

[MIT](LICENSE).
