# Quantization

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** Quantization is what lets a whole streamed layer stay ~4 bits/weight in RAM instead of 32. The [streaming loader](streaming-loader.html) keeps weights in their on-disk form; this module turns them back into fp32 one row at a time, inside the matmul the [transformer](transformer.html) calls. Parallelism across output rows is the [thread pool](thread-pool.html)'s job.

## Problem

Real Q4_K_M GGUF files are a *mix* of block-quantized tensor types — most weights Q4_K, a few (output, some attention) Q6_K, norms F32, embeddings often Q6_K/F16. To run one on CPU we must dequantize each block back to fp32 for the arithmetic. The naive move — bulk-expand a layer to fp32 on load — throws away the entire memory win: an 8× blow-up of the dominant RAM term. The engine has to dequantize *without ever materializing more than a sliver of fp32 at once*, and it must read the on-disk bytes with byte-exact offsets so unmodified Hugging Face files parse.

## Design

### On-disk block geometry (`kTraits`)

`DType` codes mirror ggml's `enum ggml_type` so a GGUF type code stores directly and the dispatcher switches on the same enum. `type_traits()` (`src/dtype.cpp`) gives each type its block geometry — `block_size` logical elements packed into `type_size` bytes — matching `ggml-quants.c` byte-for-byte so offsets line up:

| DType | code | block_size | type_size | notes |
|:--|--:|--:|--:|:--|
| F32 | 0 | 1 | 4 | unquantized |
| F16 | 1 | 1 | 2 | |
| Q4_0 | 2 | 32 | 18 | half `d` + 16 B nibbles |
| Q4_1 | 3 | 32 | 20 | half `d`, half `m` + 16 B |
| Q5_0 | 6 | 32 | 22 | half `d` + 4 B `qh` + 16 B |
| Q5_1 | 7 | 32 | 24 | |
| Q8_0 | 8 | 32 | 34 | half `d` + 32 B int8 |
| Q8_1 | 9 | 32 | 36 | half `d`, half `s` + 32 B |
| Q2_K | 10 | 256 | 84 | |
| Q3_K | 11 | 256 | 110 | |
| Q4_K | 12 | 256 | 144 | half `d`,`dmin` + 12 B scales + 128 B |
| Q5_K | 13 | 256 | 176 | |
| Q6_K | 14 | 256 | 210 | 128 B `ql` + 64 B `qh` + 16 B `sc` + half `d` |
| Q8_K | 15 | 256 | 292 | |
| IQ4_NL | 20 | 32 | 18 | half `d` + 16 B nibbles → 16-entry LUT |
| BF16 | 30 | 1 | 2 | |

`type_nbytes(t, n) = n / block_size · type_size`; K-quants use `QK_K = 256`, the legacy quants 32.

### Per-format dequant

`dequantize_row(t, src, dst, n)` (`src/quant.cpp`) expands `n` logical elements of type `t` into fp32, dispatching one case per supported format: F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL. Anything else `throw`s.

### The fused matmul — bounded RSS

`matmul_quant(y, W, t, x, n_out, n_in, pool)` computes `y = W @ x` where `W` is `[n_out, n_in]` stored as type `t`, row-major, each row a whole number of blocks. The trick: **dequantize exactly one output row into a tiny per-worker scratch, dot it with `x`, discard, move on.**

```text
row_bytes = type_nbytes(t, n_in)          # one output row's on-disk size
for o in [begin, end):                    # a worker's slice of the n_out rows
    dequantize_row(t, W + o*row_bytes, scratch[n_in], n_in)   # ~4 bits -> fp32, ONE row
    y[o] = dot_f32(scratch, x, n_in)
```

Only `n_in` fp32 values are ever live per worker, so a whole layer stays quantized in RAM even while it is being multiplied — this is the mechanism behind the flat resident-weight footprint. `linear()` (`include/llm/linear.h`) is the single call the transformer makes: F32 → `matmul`, block-quantized → `matmul_quant`, Q8_0 under `--fast` → the SDOT kernel below.

### The int8 SDOT `--fast` kernel

`matmul_q8_0_i8` (`src/neon.cpp`) is an approximate speed path. Under `--fast` (`fast_quant_enabled()`) *and* for Q8_0 weights whose `n_in` is a multiple of 32, `linear()` routes here instead of `matmul_quant`. It:

1. quantizes the **activation** to per-32-block int8 once (`quantize_activation_q8`), reused across all output rows;
2. per output row, accumulates each block's int8·int8 dot in an `int32x4` via ARM `vdotq_s32` (two `SDOT`s per block, no per-block horizontal reduce), scaled by `weight_d · activation_d` through the hardware fp16 converter and folded into a float accumulator.

Because the activation is quantized, the result is an **approximation** — the fp32-dequant `matmul_quant` path stays the correctness oracle. The kernel is guarded by `__ARM_FEATURE_DOTPROD`; where SDOT is absent it falls back to `matmul_quant(…, Q8_0, …)`.

### NEON dequant fast paths

Only **Q4_K and Q6_K** have hand-vectorized NEON dequant fast paths (the tensors that dominate a Q4_K_M model). Every other format uses the portable scalar dequant.

## Alternatives considered

| Approach | Resident fp32 | Why not |
|:--|:--|:--|
| Bulk-dequant the layer on load | whole layer × 8 | Discards the memory win; kept only as `Residency::FP32`, the numeric oracle. |
| Dequant one row, dot, discard (**`matmul_quant`**) | one row (`n_in`) | Bounded RSS; the layer stays ~4 bits/weight while multiplying. |
| Int8 SDOT on Q8_0 (`--fast`) | one activation copy | Fastest on ARM, but approximate and Q8_0/ARM-only. |

## Tradeoffs

- **Accuracy holds across formats.** Against llama.cpp on TinyLlama ("The capital of France is" → " Paris"): F16 final cosine **1.000000**, Q8_0 **0.999925** (8/10 top-10), Q5_K_M **0.999829**, Q4_K_M **0.999823** — all PASS on argmax. See [Benchmarks](benchmarks.html).
- **`--fast` trades a hair of accuracy for speed, on ARM only.** Q8_0 `--fast` resident decode reaches smollm2 **62→171** and tinyllama **50** tok/s (vs llama.cpp 57); the fp32 path stays available and exact.
- **No AVX2 equivalents.** Neither the SDOT kernel nor the K-quant NEON fast paths have an x86 counterpart — x86 runs the portable scalar dequant.
- **Q8_1 and Q8_K are NOT dequantizable.** Their geometry is in `kTraits` (so files parse and offsets compute), but `dequantize_row` has no case for them and throws. They appear as activation/intermediate scratch formats in ggml, not as weights we consume.

## Source files

| File | Role |
|:--|:--|
| `include/llm/dtype.h` | `DType` enum (ggml codes), `TypeTraits`, `type_nbytes` |
| `src/dtype.cpp` | `kTraits` block geometry table (byte-exact vs ggml) |
| `include/llm/quant.h` | `dequantize_row`, `matmul_quant`, fp16/bf16 converters, reference quantizers |
| `src/quant.cpp` | per-format dequant, the fused per-row `matmul_quant`, `quantize_q8_0`/`q4_0` |
| `src/neon.cpp` | `matmul_q8_0_i8` int8 SDOT kernel, `quantize_activation_q8`, `--fast` flag, bulk fp16 convert |
| `include/llm/linear.h` | `linear()` — the one dtype dispatch the transformer uses |

## Future work

- **NEON dequant fast paths for more formats** (Q5_K, the legacy quants) beyond today's Q4_K/Q6_K.
- **An x86 SIMD path** (AVX2/AVX-512 VNNI) mirroring the ARM SDOT and K-quant kernels.
- **IQ-family beyond IQ4_NL** and the missing importance-matrix quants, as GGUF adds them.
