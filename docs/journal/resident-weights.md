# Flat resident weights

**The clearest proof that streaming works is a number that refuses to move.** In a resident runtime, weight RAM grows linearly with layer count — twice the layers, twice the memory. In SipLLM, resident *weight* RAM stays essentially **flat** as a model gets deeper, because only one layer's weights are ever live and they are held at ~4 bits/weight, not 32. We built two toy models of the same width but 4, 16, and 32 layers deep and measured: resident weights held at ~**1.5&nbsp;MB** across all three.

## The problem: depth taxes memory in a resident runtime

Load everything resident and each transformer block adds its full weight footprint to RAM for the entire run. A 32-layer model costs 8× the weight RAM of a 4-layer one of the same width, even though the compute only ever touches one block at a time. That linear-in-depth tax is exactly what makes large models un-loadable on small devices.

## The fix: quantized residency + fused per-row dequant

Two mechanisms from the [founding design](j-streaming-layers.html) combine to break the dependence on depth:

1. **Per-layer residency.** `loadLayer(L)` makes one block resident; `unloadLayer()` frees it for reuse (`src/loader.cpp`). Resident weight RAM is bounded by a single block regardless of how many blocks follow it.
2. **Weights stay quantized; dequant is fused per-row.** In `Residency::Quantized` the raw on-disk bytes stay resident and are never bulk-expanded. `matmul_quant` (`src/quant.cpp`) dequantizes *one output row* into a tiny scratch buffer, dots it with the activation, and discards it — so a layer occupies ~4 bits/weight in RAM instead of the 32 it would cost as fp32. `Residency::FP32` (dequant on load) uses 8× the RAM and survives only as the numeric-equivalence oracle. Norm weights are tiny and always fp32.

> [!KEY] Depth is free, width is the cost
> Because residency is per-*layer* and dequant is per-*row*, resident weight RAM tracks a single layer's *width*. Adding layers adds run *time* (more streamed reads) but not resident weight RAM.

## Measured: the number that doesn't move

Two toy models, same width, increasing depth — resident weights held flat:

| Model | Resident weights |
|:--|--:|
| toy, 4 layers | ~1.5 MB |
| toy, 16 layers | ~1.5 MB |
| toy, 32 layers | ~1.5 MB |

On real models a single resident layer is **37.6&nbsp;MB** (SmolLM2, 1 layer) and **106.5&nbsp;MB** (TinyLlama, 1 layer) — still one layer's width, not the whole model. Contrast the *resident* extreme, where you deliberately pin hot layers with `--ram-budget`: pinning dials weight residency up to 143&nbsp;MB (SmolLM2) or 630&nbsp;MB (TinyLlama). That is the whole RAM↔speed range — from one flat layer to a fully pinned model — and it is your choice, not a fixed cost of depth.

> [!NOTE] Quantization caveats
> The fused per-row path supports F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, and the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K plus IQ4_NL. **Q8_1 and Q8_K are not dequantizable and throw.** Only Q4_K/Q6_K have NEON dequant fast paths; the int8 SDOT `--fast` kernel is Q8_0-only, ARM-only, and an approximation — the fp32-dequant path remains the correctness oracle. See the [quantization](quantization.html) page for the kernel details.

## What it unlocked

Flat resident weights are why depth stopped mattering: the [bigger-than-RAM](j-bigger-than-ram.html) 13&nbsp;B run costs barely more resident weight RAM than the 8&nbsp;B, and a hypothetical 70&nbsp;B of the same width would too. The residency modes and the per-row dequant seam are documented on the [streaming layer loader](streaming-loader.html) architecture page; the kernels live in [quantization](quantization.html).
