# Contributing

SipLLM is a small, readable, dependency-free codebase — genuinely nice to hack
on. This page distills `CONTRIBUTING.md` and the engineering charter (`CLAUDE.md`)
into what you need to land a change without regressing the thesis.

> [!KEY] The one hard rule
> **No new runtime dependencies.** The whole point is a from-scratch engine in
> standard **C++17 + `pthread`** — no PyTorch, no ONNX, no ggml, no BLAS, no
> CMake. A PR that adds a third-party library to the inference path will not be
> merged. (Dev-only tooling is negotiable.)

## Build

```bash
git clone https://github.com/ankit1057/sipllm.git && cd sipllm
make            # -> build/llm, build/bench, build/inspect_gguf, ...
make test       # 34 unit tests, no external deps — should be all green
```

That is the entire toolchain: `make` + a C++17 compiler. If `make test` is green,
you have a working dev environment.

## Coding standards

- **C++17, zero runtime deps** — the hard rule above.
- **Match the surrounding style** — ~100-col lines; comments explain the *why*
  (the tricky invariant), not the *what*. Read a neighboring file first.
- **Additive & guarded** — a new dial's default must reproduce prior behavior
  byte-for-byte (`--ram-budget 0` == today). Trace every shared-symbol change to
  all its callers before editing.

## Measure-first / North Star discipline

`CLAUDE.md` Rule&nbsp;0: **no optimization begins without measurement.** Before
touching an issue, run the benchmark suite (`scripts/bench.sh`,
`scripts/bench_ram_budget.sh`), rank bottlenecks by impact × leverage, and state
the current bottleneck · expected benefit · risk · complexity · how success is
measured · rollback plan. Every change must earn its place against the eight
levers (reduce peak RSS, raise throughput, cut latency, grow the largest runnable
model, improve correctness/portability, simplify, improve maintainability).

Rule&nbsp;1: at the end of every optimization wave, **refresh the North Star
scorecard** at the top of `CHANGELOG.md` with freshly measured values, append a
dated wave entry, and commit the latest benchmark JSON under `bench/results/`.
The scorecard is the single measured source of truth.

## Regression policy

- **The golden matrix must stay green.** If you touch the forward pass,
  quantization, RoPE, attention, or the loader, re-run the cross-engine validation
  before opening the PR — all formats stay PASS with per-layer cosine ≈ 1.0:

```bash
python3 golden/validate_matrix.py --prompt "The capital of France is"
```

- **`make test` stays green** on every change.
- **`--ram-budget 0` is byte-identical** to prior behavior — pinning is a pure
  cache and must never alter logits or KV. Do not redesign or remove validated
  work without strong measured evidence.

## Benchmark expectations

Every benchmark report states **model · quantization · RAM budget · hardware ·
compiler · commit SHA**, alongside peak RSS · resident weights · decode tok/s ·
prefill tok/s · TTFT, and compares against the previous baseline. Numbers are
authoritative only from **`/usr/bin/time -l`** (peak RSS, the cross-runtime
figure) and the engine's own stats. **Never fabricate a metric** — mark it
**N/A** if unmeasured (e.g. energy/token needs `sudo powermetrics`).

## PR checklist

1. Fork, branch from `main` (`git checkout -b feature/my-thing`).
2. Make the change; add or adjust tests.
3. `make test` green — and the golden matrix if you touched the math.
4. If it is an optimization: attach before/after measurements (full report
   fields above) and, at wave close, update the scorecard + commit the JSON.
5. Confirm additive & guarded — defaults reproduce prior behavior byte-for-byte.
6. Open the PR with a clear *what* and *why*; reference any issue / RFC.
7. CI (build + tests on gcc and clang) must pass.

## Testing philosophy

<div class="card-grid"><div class="card"><h3>Zero-dep harness</h3><p>34 unit tests, no gtest/catch2 — the test suite honors the same no-dependency rule as the engine (<code>make test</code>).</p></div><div class="card"><h3><code>ref_forward</code> oracle</h3><p>The exact fp32-dequant path is the correctness oracle. Approximations like <code>--fast</code> (int8 SDOT) are opt-in and validated against it, never the other way round.</p></div><div class="card"><h3>Two regimes</h3><p><strong>Bit-identical</strong> where a change must not move a bit (<code>--ram-budget</code> pinning, KV grow-on-demand); <strong>tolerance</strong> (per-layer cosine ≈ 1.0, <code>1e-3</code> max|Δ|) for cross-engine quantized comparison.</p></div><div class="card"><h3>Sanitizer / valgrind gates</h3><p>CI gates on ASan / UBSan / TSan + valgrind + a prompt/config fuzzer — the whole uninitialized-read class was eliminated at source (#32) and is kept out by these gates.</p></div></div>

## Good first issues

- **New quantization format** — a dequant path in `src/quant.cpp` + a round-trip
  test in `tests/test_quant.cpp`.
- **More registry models** — extend `builtin_url()` in `sipllm` with public,
  ungated small GGUFs.
- **NEON / x86 SIMD** — port a scalar hot path in `src/neon.cpp` (guarded by
  `LLM_HAVE_NEON`) or add an AVX2 path behind `simd.h`, with an equivalence test.
- **Docs** — a diagram or a forward-pass walkthrough.

By contributing you agree your work is licensed under the project's MIT License.
For the reasoning behind these rules, see [design decisions](design-decisions.html)
and the [RFC index](rfc-index.html).
