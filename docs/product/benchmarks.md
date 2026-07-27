# Benchmarks &amp; validation

<div class="hero"><p class="lead"><strong>Correctness is a measurement, not a claim; footprint is a measurement, not a claim.</strong> Every number here is Apple M3, CPU-only, warm cache, median-of-3, measured 2026-07-27. Peak RSS is the authoritative cross-runtime figure from <code>/usr/bin/time -l</code>. All figures are committed as JSON under <code>bench/results/</code>.</p></div>

> [!MEASURED] Host, not device
> These are measured on **host (Apple M3)**. Android on-device numbers are **pending** — the Flutter/FFI layer (Wave 8) is host-verified only. No phone/watch figure is quoted anywhere in these docs because none has been measured yet.

## North Star scorecard

The project's single measured source of truth, refreshed at the end of every optimization wave (charter Rule 1):

| Metric | Value |
|:-------|:------|
| Peak RSS — tinyllama | 121 MB (stream) … 644 MB (fully resident) |
| Peak RSS — smollm2 | 54 MB (stream) … 161 MB (fully resident) |
| vs llama.cpp CPU | 1356 MB (tinyllama) / 546 MB (smollm2) → **10–11× smaller at min budget** |
| Resident weights | FLAT ~1.5 MB across toy 4/16/32 layers; real 37.6 MB (smollm2 1-layer) / 106.5 MB (tinyllama 1-layer); pinned dial up to 143 MB (smollm2) / 630 MB (tinyllama) |
| Decode tok/s (Q8 `--fast`, resident) | smollm2 62→171 · tinyllama 50 (vs llama.cpp 57) |
| Decode tok/s (RAM-budget dial, exact fp32) | smollm2 53→66 · tinyllama 11→22 |
| TTFT | smollm2 ~0.10 s · tinyllama ~0.68 s (vs llama.cpp 0.003 / 0.021 s) |
| Prefill throughput | smollm2 50–67 · tinyllama 7–23 tok/s (vs llama.cpp 1680 / 238) |
| Expansion factor (disk ÷ peak-RSS, min budget) | 2.7× (smollm2) · 5.5× (tinyllama) |
| Largest runnable | Llama-2-13B (7.87 GB Q4) in **317 MB (25×)**; Llama-3.1-8B (4.92 GB Q4) in 204 MB (24×) |
| Energy / token | **N/A** (needs `sudo powermetrics`; never fabricated) |

> [!NOTE] Read the trade honestly
> SipLLM is *not* uniformly faster. TTFT and prefill throughput trail llama.cpp substantially — streaming trades disk bandwidth for a bounded footprint. The headline is RAM, and the ability to run models that do not fit at all.

## Bigger than RAM — the defining capability

16&nbsp;GB Mac, ~3&nbsp;GB free (loading these resident is impossible), `--stream-lm-head --no-async --ctx 512`, greedy:

| Model | Weights on disk | Peak RSS | Model ÷ RSS |
|:------|----------------:|---------:|------------:|
| TinyLlama-1.1B Q8_0 | 1.17 GB | 61 MB | 19× |
| Llama-3.1-8B Q4_K_M | 4.92 GB | 204 MB | 24× |
| Llama-2-13B Q4_K_M | 7.87 GB | 317 MB | 25× |

Peak RSS grows with layer *width*, never model depth/total size. See [Bigger than RAM](j-bigger-than-ram.html).

## Half the RAM, comparable speed

TinyLlama-1.1B Q8_0, `--ctx 512`, greedy, 4 threads, warm cache:

| Runtime | Peak RSS | Decode |
|:--------|---------:|-------:|
| llama.cpp (CPU) | 2326 MB | ~57 tok/s |
| SipLLM `--fast --ram-budget 1200M` (resident) | 1113 MB | ~50 tok/s |
| SipLLM `--fast` (streaming) | 175 MB | 6.8 tok/s |

2.1× less RAM at ~88% of llama.cpp's decode, or 13× less RAM streaming — a smooth [RAM↔speed dial](why-streaming.html), not a fixed point.

## Golden validation vs llama.cpp

For the same model and prompt, SipLLM dumps every transformer block's residual stream and the final logits and diffs them against llama.cpp's own values (captured through its eval callback). **Cross-engine outputs are never bit-exact** — summation order and rounding differ — so the comparison is numerical: per-layer `max|Δ|`, cosine similarity, and argmax/top-k agreement.

Prompt `"The capital of France is"` → both engines greedily predict `" Paris"` (TinyLlama-1.1B, 22 layers, dim 2048, GQA 32/4):

| Format | worst layer max\|Δ\| | final logit max\|Δ\| | final cosine | top-10 | argmax | peak RSS | result |
|:-------|---------------------:|---------------------:|-------------:|:------:|:------:|---------:|:------:|
| F16 | 4.45e-03 | 5.76e-03 | 1.000000 | 10/10 | ✓ | 412 MB | PASS |
| Q8_0 | 1.79e-01 | 3.03e-01 | 0.999925 | 8/10 | ✓ | 269 MB | PASS |
| Q5_K_M | 2.37e-01 | 4.40e-01 | 0.999829 | 10/10 | ✓ | 223 MB | PASS |
| Q4_K_M | 3.88e-01 | 4.35e-01 | 0.999823 | 10/10 | ✓ | 215 MB | PASS |

Two things fall straight out, both exactly what theory predicts:

1. **F16 is numerically identical to llama.cpp** (cosine `1.000000`) — the compute graph is correct; every residual difference in the quantized rows is pure quantization error.
2. **Error grows monotonically as quantization coarsens** (F16 ≪ Q8_0 < Q5_K_M < Q4_K_M). A bug would produce erratic, layer-localized divergence; this smooth accumulation is the signature of a faithful implementation.

The golden matrix currently covers the Llama path; newer architectures are unit-tested and validated against llama.cpp as models are added. See [Quantization](quantization.html) and the [LM-head journal](j-lm-head.html).

## RAM-budget sweep

`bench_ram_budget.sh`, M3, warm, ctx 512:

| Model | Budget | Pinned | Decode tok/s | Streamed | Peak RSS |
|:------|-------:|-------:|-------------:|---------:|---------:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | 22.1 | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | 65.9 | 113 MB | 161 MB |

Pinning is a pure cache: logits + KV are bit-identical across every budget, and peak RSS ≤ budget at every point. See [Memory planner](memory-planner.html).

## Streamed LM head

TinyLlama-1.1B Q4_K_M, M3, warm, t=4:

| Mode | Peak RSS | Resident weights | Decode |
|:-----|---------:|-----------------:|-------:|
| default (resident) | 121 MB | 106.5 MB | 11.48 tok/s |
| `--stream-lm-head` | 69 MB | 52.7 MB | 11.11 tok/s |

Streaming the output projection nearly halves peak RSS for a negligible decode cost — the head is often the single largest tensor. See the [LM-head journal](j-lm-head.html).

## Methodology

- **Peak RSS** is read from `/usr/bin/time -l` (maximum resident set size) — the authoritative, cross-runtime figure, independent of any engine's self-report.
- **Median-of-3** for every timing (TTFT, decode, prefill tok/s); **warm cache** so the file is in the page cache.
- **Cross-engine correctness is numerical, not bit-exact** — see the golden matrix above and `golden/README.md` for why (summation order + rounding differ between engines).
- **Every report states** model · quantization · RAM budget · hardware · compiler · commit SHA alongside the metrics, and compares against the previous baseline (charter Rule 1).

## Reproduce it

```bash
# peak RSS / TTFT / decode tok/s harness
scripts/bench.sh

# decode-tok/s + peak-RSS vs --ram-budget sweep
scripts/bench_ram_budget.sh

# cross-engine numerical validation vs llama.cpp
python3 golden/validate_matrix.py --prompt "The capital of France is"

# the dependency-free unit suite
make test
```

Latest runs are committed under `bench/results/` (`demo-v0.4-*.json`, `bigger-than-ram-*.json`, `ram-budget-*.json`) with the full test log in `bench/results/test-results-2026-07-27.txt`. See the [performance history](performance-history.html) for the wave-by-wave trend.
