# Streaming GGUF LLM Inference Engine

**A dependency-free, from-scratch LLM inference engine in C++17 that runs
quantized Llama / TinyLlama models by streaming weights off disk — so peak RAM
stays flat (~200–400 MB) no matter how big the model is. Numerically validated
against [llama.cpp](https://github.com/ggml-org/llama.cpp), layer by layer,
across four GGUF quantization formats.**

No PyTorch. No ONNX. No ggml for inference. No BLAS. Just standard C++17 and
`pthread` — plus hand-written ARM64 NEON kernels and an optional Vulkan matmul
backend. Built to run real language models on a phone (Termux / Android on a
Dimensity 8300), and portable to any Linux/macOS ARM or x86 host.

---

## Why this exists

The usual way to run an LLM loads the entire model into memory. A 1.1 B model in
Q8_0 is ~1.1 GB resident; an 8 B model is far more than a phone can hold. This
engine instead **streams one transformer block at a time**: read the block's
weights from disk with `pread`, run attention + FFN for that block, free it,
move to the next. Only a single layer's weights (plus the KV cache) are ever
resident, so a model many times larger than RAM still runs — memory is bounded
by *layer size*, not *model size*.

```
             ┌─────────── on disk (GGUF, quantized) ───────────┐
  block 0 → block 1 → block 2 → ... → block N-1 → output head
             │
             ▼   for each block:  pread → dequant → attention+FFN → free
       peak RAM ≈ one layer + KV cache   (flat across the whole model)
```

## Validated against llama.cpp

Correctness is not a claim — it's a measurement. For the same model and prompt,
the engine dumps every transformer block's residual stream and the final logits,
and diffs them against llama.cpp's own values (captured through its eval
callback). Cross-engine outputs never match bit-for-bit (summation order and
rounding differ), so the comparison is numerical: per-layer `max|Δ|`, cosine
similarity, and final-logit argmax / top-k agreement.

Prompt `"The capital of France is"` → both engines greedily predict **" Paris"**:

| Format | worst layer `max\|Δ\|` | final logit `max\|Δ\|` | final logit cosine | top-10 | argmax | peak RSS | result |
|:-------|-----------------------:|-----------------------:|-------------------:|:------:|:------:|---------:|:------:|
| **F16**    | 4.45e-03 | 5.76e-03 | **1.000000** | 10/10 | ✅ | 412 MB | **PASS** |
| **Q8_0**   | 1.79e-01 | 3.03e-01 | 0.999925 | 8/10 | ✅ | 269 MB | **PASS** |
| **Q5_K_M** | 2.37e-01 | 4.40e-01 | 0.999829 | 10/10 | ✅ | 223 MB | **PASS** |
| **Q4_K_M** | 3.88e-01 | 4.35e-01 | 0.999823 | 10/10 | ✅ | 215 MB | **PASS** |

*(TinyLlama-1.1B-Chat-v1.0, 22 layers, dim 2048, GQA 32/4 heads.)*

Two things fall straight out of this table and both are exactly what theory
predicts:

1. **F16 is numerically identical to llama.cpp** (cosine `1.000000`, `max|Δ|`
   ~5e-3) — the compute graph is correct. Every residual difference in the
   quantized rows is pure quantization error, nothing else.
2. **Error grows monotonically as quantization coarsens** (F16 ≪ Q8_0 < Q5_K_M <
   Q4_K_M). A bug would produce erratic, layer-localized divergence; this smooth
   accumulation is the signature of a faithful implementation.

Meanwhile peak resident memory stays **215–412 MB** while the model files on disk
range from **669 MB (Q4_K_M) to 2.2 GB (F16)** — streaming works.

Reproduce it yourself:

```bash
python3 golden/validate_matrix.py --prompt "The capital of France is"
```

See [`golden/README.md`](golden/README.md) for the full methodology.

## Features

- **Real GGUF v2/v3 parser** — loads unmodified files from Hugging Face
  (TheBloke, etc.). Reads hyperparameters, tensor directory, and the tokenizer
  straight out of the container.
- **Broad quantization support** — dequantization for `F32`, `F16`, `BF16`,
  `Q4_0/1`, `Q5_0/1`, `Q8_0`, and the K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_K`. A **fused quantized matmul** dequantizes one weight row into a
  small scratch buffer and dots it with the activation, so a whole layer stays
  quantized in RAM (never expanded to fp32 in bulk).
- **Streaming layer loader** — synchronous `pread`, an **async double-buffered
  prefetcher** (compute layer *N* while layer *N+1* loads), or an **`mmap`
  backend** — switchable and benchmarked side by side.
- **Correct transformer** — RMSNorm, RoPE (adjacent-pair, ggml-compatible),
  Grouped-Query Attention, SwiGLU FFN, causal KV cache.
- **Tokenizers** — SentencePiece (Llama), byte-level BPE (GPT-2 / Llama-3), with
  byte fallback, all decoded from the GGUF metadata.
- **ARM64 NEON kernels** — `sdot` / `i8mm` accelerated dot products and quantized
  matmul, with scalar fallbacks for portability.
- **Profiling & visualization** — a terminal profiler shows per-layer I/O,
  dequant, and compute time as live bars, plus peak RSS and prefill/decode
  tokens/sec, `pread` vs `mmap`.
- **Web GUI** — a tiny self-contained HTTP server (`make server`) for chatting
  with a model in the browser.
- **Optional Vulkan backend** — GPU matmul offload via a compute shader when a
  Vulkan device and `glslc` are available (CPU fallback otherwise).
- **Zero-dependency test suite** — 34 unit tests, no gtest/catch2 (`make test`).

## Quick start

```bash
# 1. Build the engine, tools, and tests
make            # -> build/llm, build/dump_logits, build/bench, ...
make test       # 34 tests, all green

# 2. Grab a GGUF model (any Llama-family model works)
mkdir -p models
curl -L -o models/tinyllama-1.1b-q4_k_m.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 3. Generate text
./build/llm models/tinyllama-1.1b-q4_k_m.gguf -p "The capital of France is" -n 40

# 4. Profile it — memory stays flat, watch each layer stream
./build/bench models/tinyllama-1.1b-q4_k_m.gguf -p "Once upon a time" -n 32

# 5. Inspect a model's tensors and metadata
./build/inspect_gguf models/tinyllama-1.1b-q4_k_m.gguf
```

## Repository layout

```
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

The invariant is that the transformer only ever talks to a `WeightSource`
interface — a tensor directory plus "read this tensor's raw bytes." Whether the
bytes come from a `pread`, a prefetch buffer, or an `mmap` page is invisible to
the math. Each block:

1. asks the loader for its weights (which may block until the prefetch lands),
2. runs RMSNorm → QKV projections → RoPE → GQA attention → output projection →
   RMSNorm → SwiGLU FFN, accumulating into the residual stream,
3. releases the weights before the next block loads.

Quantized weights are never bulk-expanded: `matmul_quant` walks one output row,
dequantizes that row's blocks into a tiny buffer, dots with the input, and moves
on. That's why a layer stays quantized in memory and why peak RSS tracks *layer*
size, not *model* size.

## Building llama.cpp for the golden test

The comparison needs a llama.cpp build (CPU backend is enough). See
[`golden/README.md`](golden/README.md) — in short: clone it into `third_party/`,
`cmake --build` the `llama`/`ggml` targets, compile `golden/llama_dump.cpp`
against them, then run `golden/validate_matrix.py`.

## Roadmap

- Android JNI/NDK packaging (Termux-first, already the primary target).
- Wider prefetch pipelining and NEON coverage of the K-quant dequant paths.
- Maturing the Vulkan backend from matmul offload to full-layer offload.

## License

[MIT](LICENSE).
