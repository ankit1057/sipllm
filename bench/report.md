# SipLLM Performance Benchmark

**Date:** 2026-07-28 02:40:03
**Git SHA:** `7f75070`

## Regression Status

## North Star Scorecard (Auto Config)

### smollm2-135m.gguf

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Peak RSS | 54.00 MB | 54.00 MB | 0.0% |
| TTFT | 0.08 s | 0.17 s | +101.2% |
| Decode | 58.61 tok/s | 15.50 tok/s | -73.6% |
| Prefill | 48.04 tok/s | 23.96 tok/s | -50.1% |
| Regression | FAIL | FAIL | — |

### tinyllama-q4_k_m.gguf

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Peak RSS | 122.00 MB | 122.00 MB | 0.0% |
| TTFT | 0.24 s | 0.41 s | +69.9% |
| Decode | 11.66 tok/s | 7.40 tok/s | -36.5% |
| Prefill | 16.73 tok/s | 9.84 tok/s | -41.2% |
| Regression | FAIL | FAIL | — |

❌ **FAIL**: Regressions detected in some configurations:

| Model | Config | Metric | Baseline | Current | Change |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | ttft_s | 0.08 | 0.17 | +101.20% |
| smollm2-135m.gguf | Auto | prefill_tok_s | 48.04 | 23.96 | -50.12% |
| smollm2-135m.gguf | Auto | decode_tok_s | 58.61 | 15.50 | -73.55% |
| smollm2-135m.gguf | Best Manual | ttft_s | 0.05 | 0.17 | +221.15% |
| smollm2-135m.gguf | Best Manual | prefill_tok_s | 77.63 | 23.93 | -69.17% |
| smollm2-135m.gguf | Best Manual | decode_tok_s | 61.12 | 17.18 | -71.89% |
| smollm2-135m.gguf | RAM Budget 150M | ttft_s | 0.09 | 0.14 | +66.28% |
| smollm2-135m.gguf | RAM Budget 150M | prefill_tok_s | 46.47 | 27.89 | -39.98% |
| smollm2-135m.gguf | RAM Budget 150M | decode_tok_s | 44.43 | 18.27 | -58.88% |
| tinyllama-q4_k_m.gguf | Auto | ttft_s | 0.24 | 0.41 | +69.87% |
| tinyllama-q4_k_m.gguf | Auto | prefill_tok_s | 16.73 | 9.84 | -41.18% |
| tinyllama-q4_k_m.gguf | Auto | decode_tok_s | 11.66 | 7.40 | -36.54% |
| tinyllama-q4_k_m.gguf | Best Manual | decode_tok_s | 10.29 | 8.35 | -18.85% |

## Auto-Tuner vs Manual Validation (M4)

The auto-tuner selected configurations that were competitive with the best manual configurations across tested models, outperforming the manual sweep on SmolLM2 while trading some decode throughput for lower memory usage on TinyLlama.

| Model | Config | TTFT (s) | Decode (tok/s) | Peak RSS (MB) | Resident Wt (MB) |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | 0.167 | 15.50 | 54.0 | 37.6 |
| smollm2-135m.gguf | Best Manual | 0.167 | 17.18 | 54.0 | 37.6 |
| smollm2-135m.gguf | RAM Budget 150M | 0.143 | 18.27 | 54.0 | 37.6 |
| tinyllama-q4_k_m.gguf | Auto | 0.406 | 7.40 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | Best Manual | 0.180 | 8.35 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | 0.274 | 5.61 | 44.0 | 24.8 |
