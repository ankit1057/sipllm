# SipLLM Performance Benchmark

**Date:** 2026-07-28 02:57:53
**Git SHA:** `ac54e1c`

## Regression Status

No baseline found. This run is the new baseline.

## Auto-Tuner vs Manual Validation (M4)

The auto-tuner selected configurations that were competitive with the best manual configurations across tested models, outperforming the manual sweep on SmolLM2 while trading some decode throughput for lower memory usage on TinyLlama.

| Model | Config | TTFT (s) | Decode (tok/s) | Peak RSS (MB) | Resident Wt (MB) |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | 0.084 | 56.93 | 54.0 | 37.6 |
| smollm2-135m.gguf | Best Manual | 0.055 | 58.12 | 54.0 | 37.6 |
| smollm2-135m.gguf | RAM Budget 150M | 0.063 | 44.05 | 54.0 | 37.6 |
| tinyllama-q4_k_m.gguf | Auto | 0.243 | 11.63 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | Best Manual | 0.160 | 11.76 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | 0.210 | 8.68 | 42.0 | 24.8 |
