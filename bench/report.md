# SipLLM Performance Benchmark

**Date:** 2026-07-28 02:47:34
**Git SHA:** `64c97a1`

## Regression Status

## North Star Scorecard (Auto Config)

### smollm2-135m.gguf

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Peak RSS | 54.00 MB | 54.00 MB | 0.0% |
| TTFT | 0.06 s | 0.07 s | +21.4% |
| Decode | 58.66 tok/s | 50.58 tok/s | -13.8% |
| Prefill | 70.96 tok/s | 58.86 tok/s | -17.1% |
| Regression | FAIL | FAIL | — |

### tinyllama-q4_k_m.gguf

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Peak RSS | 122.00 MB | 122.00 MB | 0.0% |
| TTFT | 0.20 s | 0.20 s | +3.1% |
| Decode | 8.38 tok/s | 8.36 tok/s | -0.2% |
| Prefill | 20.42 tok/s | 19.79 tok/s | -3.1% |
| Regression | PASS | PASS | — |

❌ **FAIL**: Regressions detected in some configurations:

| Model | Config | Metric | Baseline | Current | Change |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | ttft_s | 0.06 | 0.07 | +21.43% |
| smollm2-135m.gguf | Auto | prefill_tok_s | 70.96 | 58.86 | -17.05% |
| smollm2-135m.gguf | Auto | decode_tok_s | 58.66 | 50.58 | -13.77% |
| smollm2-135m.gguf | RAM Budget 150M | prefill_tok_s | 52.19 | 48.60 | -6.88% |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | peak_rss_mb | 34.00 | 42.00 | +23.53% |

## Auto-Tuner vs Manual Validation (M4)

The auto-tuner selected configurations that were competitive with the best manual configurations across tested models, outperforming the manual sweep on SmolLM2 while trading some decode throughput for lower memory usage on TinyLlama.

| Model | Config | TTFT (s) | Decode (tok/s) | Peak RSS (MB) | Resident Wt (MB) |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | 0.068 | 50.58 | 54.0 | 37.6 |
| smollm2-135m.gguf | Best Manual | 0.071 | 42.84 | 54.0 | 37.6 |
| smollm2-135m.gguf | RAM Budget 150M | 0.082 | 34.17 | 54.0 | 37.6 |
| tinyllama-q4_k_m.gguf | Auto | 0.202 | 8.36 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | Best Manual | 0.196 | 8.28 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | 0.276 | 6.63 | 42.0 | 24.8 |
