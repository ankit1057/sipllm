# Changelog

All notable changes to SipLLM. The **North Star scorecard** below is refreshed at
the end of every optimization wave from measured data (see [CLAUDE.md](CLAUDE.md),
Rule 1). Peak RSS is the authoritative cross-runtime number from `/usr/bin/time -l`.

```
============================
SipLLM North Star   (measured 2026-07-24 · Apple M3 · warm cache · median-of-3)
============================
Peak RSS:              tinyllama 121 MB (stream) … 644 MB (fully resident)
                       smollm2    54 MB (stream) … 161 MB (fully resident)
                       vs llama.cpp CPU: 1356 MB / 546 MB  → 10–11× smaller at min budget
Resident Weights:      FLAT 1.5 MB across 4/16/32 toy layers (streaming thesis holds)
                       real: 37.6 MB (smollm2 1-layer) / 106.5 MB (tinyllama 1-layer)
                       pinned dial: up to 143 MB (smollm2) / 630 MB (tinyllama)
Decode tok/s:          smollm2 53 → 66  ·  tinyllama 11.4 → 22.1  (RAM-budget dial)
                       vs llama.cpp: 404 / 95  (kernel gap remains, see below)
TTFT:                  smollm2 ~0.10 s · tinyllama ~0.68 s   vs llama.cpp 0.003 / 0.021 s
Prefill Throughput:    smollm2 50–67 tok/s · tinyllama 7–23 tok/s  vs llama.cpp 1680 / 238
Expansion Factor:      2.7× (smollm2) · 5.5× (tinyllama)  = disk / peak-RSS at min budget
Largest Runnable Model:bounded by DISK not RAM — ran a 668 MB model at 121 MB peak RSS.
                       Remote-cache streaming (#43) removes the disk bound too.
Energy / Token:        N/A (needs `sudo powermetrics`; never fabricated)

Current Largest Bottleneck:  Quantized matmul KERNEL throughput. With #37 the
                       streaming tax is now a user choice; at full residency decode
                       is compute-bound — tinyllama 22 tok/s vs llama.cpp 95 (4.3×),
                       and prefill/TTFT ~32× behind (per-position dequant).
Estimated Gain if Fixed:  Batched-GEMM dequant amortization (#41) → prefill/TTFT
                       ~3–5×; a shared decode GEMV lifts decode toward parity.
Why this is the next priority:  #37 isolated streaming from compute, so the residual
                       gap is unambiguously kernel efficiency — it compounds with
                       every future feature (remote streaming, MoE, long context)
                       and is the only thing keeping SipLLM off performance parity.
                       (Long-context peak-RSS floor is dominated by the fp32 KV
                       cache — Q8_0 KV (#39) is the moat-side runner-up.)
Confidence:            High — bottleneck is measured, not assumed; re-validate at the
                       start of the next wave before committing to #41.
```

## [Unreleased]

### Wave 6 — `--ram-budget`: hard peak-RSS ceiling + partial layer residency (#37)

The headline **RAM-speed dial**. The fixed 2-buffer streaming window becomes a
byte ceiling: the loader pins as many contiguous hot layers resident as fit under
the budget and streams the rest, so peak weight RSS never exceeds the budget.
Turns *bounded-RSS XOR speed* into a tunable continuum — a capability no other
runtime offers (llama.cpp mmap has no hard ceiling; vLLM/TRT-LLM/MLX/MLC/
ExecuTorch require the model to fit in RAM/VRAM).

**Added**
- `--ram-budget BYTES|N{K,M,G}` (CLI) — total peak-RSS target. `Runtime` derives
  the loader's weight ceiling by reserving the KV cache (up to `--ctx`) and a
  scratch allowance. `0` = unlimited (today's behavior).
- `LayerLoader` residency manager: pins layers `[0, n_pinned)` once, serves them
  with zero I/O; a per-layer guard keeps `resident_bytes() ≤ budget`. Below the
  streaming floor it degrades gracefully to pure streaming.
- `tests/test_ram_budget.cpp` — proves (1) logits + KV **bit-identical** across
  budgets (pinning is a pure cache) and (2) the hard ceiling holds across a fuzzed
  budget sweep.
- `scripts/bench_ram_budget.sh` — the reproducible decode-tok/s + peak-RSS vs
  budget sweep; latest run in `bench/results/`.

**Measured** (Apple M3, warm, ctx 512, median-of-3):

| model | budget | pinned | decode tok/s | streamed | peak RSS |
|:------|-------:|-------:|-------------:|---------:|---------:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | **22.1** | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | **65.9** | 113 MB | 161 MB |

Decode up to **+95%** (tinyllama) / **+24%** (smollm2); streamed I/O **−96%**;
peak RSS ≤ budget at every point; golden matrix + all unit tests green.

**Correctness** — pinning returns byte-identical `WeightRef`s, so the forward pass
is bit-for-bit invariant to the budget; `--ram-budget 0` reproduces prior behavior
exactly.

### CI / release
- `release.yml` no longer builds macOS artifacts on GitHub — Linux x86_64/aarch64
  only. macOS bundles are built & uploaded from a local Mac via
  `scripts/release-macos.sh` (portable `ARCHFLAGS=""` build → `gh release upload`).
