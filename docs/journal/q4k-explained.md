# Q4_K explained

**Q4_K is how a 4-bit GGUF actually stores its weights: 256 numbers squeezed into a 144-byte block — roughly 4.5 bits per weight — with two levels of scaling so the quantization error stays tiny. SipLLM has to decode that block *byte-for-byte identically to ggml*, because the whole point of the runtime is to consume unmodified `Q4_K_M` files pulled straight from Hugging Face. This is the walkthrough of `dequant_q4_K` in `src/quant.cpp`: what the 144 bytes mean, how `get_scale_min_k4` unpacks the packed 6-bit scales, and why we keep two implementations of the same formula.**

## The problem: consume ggml's format exactly

A `Q4_K_M` checkpoint is a mix of `Q4_K` and `Q6_K` tensors. If SipLLM's dequantizer disagrees with `ggml-quants.c` by even one ULP in the wrong place, the logits drift and the model stops matching llama.cpp. So `src/quant.cpp` does not invent a layout — every block formula mirrors ggml, and `QK_K = 256` is the K-quant block size. The job is decode, not design: read the on-disk bytes, reproduce the exact fp32 values ggml would.

## Anatomy of a 144-byte block

A `Q4_K` block holds `QK_K = 256` weights in exactly 144 bytes. From `dequant_q4_K`, the pointers are `scales = p + 4`, `q = p + 16`, and `p += 144` per block:

| Offset | Bytes | Field | Meaning |
|:--|--:|:--|:--|
| 0 | 2 | `d` (fp16) | super-block scale |
| 2 | 2 | `dmin` (fp16) | super-block minimum |
| 4 | 12 | packed scales/mins | eight 6-bit scales + eight 6-bit mins |
| 16 | 128 | `q` | 256 × 4-bit quants (two nibbles per byte) |
| | **144** | | **256 weights ≈ 4.5 bits each** |

The 256 weights are split into **eight 32-element sub-blocks**. Each sub-block gets its own 6-bit scale and 6-bit min, and *those* are themselves scaled by the two fp16 super-block values `d` and `dmin`. Two tiers of scaling — a coarse fp16 pair for the whole block, a fine 6-bit pair per sub-block — is what buys Q4_K its accuracy at 4 bits.

## Unpacking the scales: `get_scale_min_k4`

The eight 6-bit scales and eight 6-bit mins do not fit cleanly in bytes — 16 × 6 bits = 96 bits = 12 bytes, but with the low four and high four sub-blocks packed differently. `get_scale_min_k4(j, q, d, m)` (shared with `Q5_K`) is the bit-surgery that recovers sub-block `j`'s scale `d` and min `m`:

```text
if j < 4:
    d = q[j]     & 63            // low 6 bits
    m = q[j + 4] & 63
else:                            // j in 4..7, high bits stashed in the first 8 bytes
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4)
```

The first four sub-blocks store their 6-bit scale/min directly in the low six bits of bytes `0..3` (scales) and `4..7` (mins). The upper four sub-blocks reuse the *top two bits* of those same bytes as the high nibble, with the low nibble living in bytes `8..11`. It looks arbitrary; it is exactly ggml's packing, which is the only thing that matters.

## The dequant formula

The main loop walks the 256 weights in strides of 64 (two 32-element sub-blocks per pass), pulling a fresh pair of scale/min pairs each time:

```text
for j in 0, 64, 128, 192:                 // is steps 0,2,4,6
    get_scale_min_k4(is + 0) -> sc, m ;  d1 = d*sc ;  m1 = dmin*m
    get_scale_min_k4(is + 1) -> sc, m ;  d2 = d*sc ;  m2 = dmin*m
    for l in 0..31:  y[l]      = d1 * (q[l] & 0xF) - m1     // low nibbles
    for l in 0..31:  y[l + 32] = d2 * (q[l] >> 4)  - m2     // high nibbles
    q += 32 ; is += 2
```

So the fully-expanded value of a weight is `d * sub_scale * quant - dmin * sub_min`. The low nibble of each `q` byte feeds one sub-block (`d1`, `m1`); the high nibble feeds the next (`d2`, `m2`). Note the sign: it is a `- m` (a subtracted minimum), not the `+ m` of the legacy affine `Q4_1` a few functions up — another detail that must match ggml or the numbers walk off.

## Two implementations, one answer

`dequant_q4_K` is written twice. Under `LLM_HAVE_NEON` the inner loop is hand-vectorized: `q` bytes are loaded 16 at a time, low/high nibbles masked and shifted apart, widened `u8 → u16 → u32 → f32`, and the affine `d1*x - m1` is a single fused multiply-add — `vfmaq_n_f32(nm1, x, d1)` computes `(-m1) + x*d1`. The `#else` branch is the plain scalar loop shown above.

> [!KEY] The scalar path is the oracle
> `Q4_K` and `Q6_K` are the only K-quants with a NEON fast path, and the int8 `--fast` SDOT kernel is **Q8_0-only** and ARM-only — it does not touch K-quants yet. The scalar fp32-dequant path is deliberately kept as the numeric oracle: every backend, on every architecture, must reproduce it. That is what lets the golden test assert the NEON output is correct rather than merely fast.

## Where it sits in the engine

`dequant_q4_K` is one arm of the `dequantize_row` dispatch, which also serves `matmul_quant` — the per-row *dequantize-then-dot* that keeps a whole layer at ~4 bits in RAM instead of expanding it to fp32. That fused path is what makes [streaming a Q4_K_M model](streaming-loader.html) memory-bounded. The correctness of this exact decode is measured against llama.cpp:

| Format | worst layer max\|Δ\| | final logit max\|Δ\| | final cosine | top-10 | argmax | result |
|:--|--:|--:|--:|:--:|:--:|:--:|
| Q4_K_M | 3.88e-01 | 4.35e-01 | 0.999823 | 10/10 | ✓ | PASS |

Same prompt, same argmax, cosine `0.999823` against llama.cpp on TinyLlama-1.1B. The next win is extending the int-dot `--fast` kernel to K-quants so 4-bit decode reaches Q8-class speed — today `Q4_K` decode still runs through this fp32 path. For the format's place among all the quantizations SipLLM handles, see the [Quantization](quantization.html) chapter; for the measured accuracy of every format, see [Benchmarks](benchmarks.html).
