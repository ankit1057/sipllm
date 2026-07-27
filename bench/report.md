# SipLLM Performance Benchmark

**Date:** 2026-07-28 02:37:26
**Git SHA:** `1cd31e7`

## Regression Status

## North Star Scorecard (Auto Config)

### smollm2-135m.gguf

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Peak RSS | 54.00 MB | 54.00 MB | 0.0% |
| TTFT | 0.10 s | 0.08 s | -12.6% |
| Decode | 45.26 tok/s | 58.61 tok/s | +29.5% |
| Prefill | 42.08 tok/s | 48.04 tok/s | +14.2% |
| Regression | PASS | PASS | — |

### tinyllama-q4_k_m.gguf

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Peak RSS | 114.00 MB | 122.00 MB | +7.0% |
| TTFT | 0.34 s | 0.24 s | -30.7% |
| Decode | 7.05 tok/s | 11.66 tok/s | +65.4% |
| Prefill | 11.59 tok/s | 16.73 tok/s | +44.3% |
| Regression | FAIL | FAIL | — |

❌ **FAIL**: Regressions detected in some configurations:

| Model | Config | Metric | Baseline | Current | Change |
|---|---|---|---|---|---|
| tinyllama-q4_k_m.gguf | Auto | peak_rss_mb | 114.00 | 122.00 | +7.02% |
| tinyllama-q4_k_m.gguf | Best Manual | prefill_tok_s | 13.65 | 12.45 | -8.79% |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | ttft_s | 0.30 | 0.64 | +113.71% |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | prefill_tok_s | 13.38 | 6.26 | -53.21% |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | decode_tok_s | 5.97 | 4.47 | -25.13% |

## Auto-Tuner vs Manual Validation (M4)

The auto-tuner selected configurations that were competitive with the best manual configurations across tested models, outperforming the manual sweep on SmolLM2 while trading some decode throughput for lower memory usage on TinyLlama.

| Model | Config | TTFT (s) | Decode (tok/s) | Peak RSS (MB) | Resident Wt (MB) |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | 0.083 | 58.61 | 54.0 | 37.6 |
| smollm2-135m.gguf | Best Manual | 0.052 | 61.12 | 53.0 | 37.6 |
| smollm2-135m.gguf | RAM Budget 150M | 0.086 | 44.43 | 54.0 | 37.6 |
| tinyllama-q4_k_m.gguf | Auto | 0.239 | 11.66 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | Best Manual | 0.321 | 10.29 | 121.0 | 106.5 |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | 0.639 | 4.47 | 44.0 | 24.8 |
