# Contributing to sipllm

Thanks for being here. sipllm is a small, readable, dependency-free codebase —
which makes it a genuinely nice project to hack on. This guide gets you from
clone to merged PR.

## TL;DR

```bash
git clone https://github.com/ankit1057/sipllm.git
cd sipllm
make            # build engine + tools   (needs a C++17 compiler + make)
make test       # 34 unit tests, no external deps — should be all green
```

That's the whole toolchain. No PyTorch, no CMake, no package manager. If
`make test` is green you have a working dev environment.

## Ground rules

- **No new runtime dependencies.** The point of this project is a from-scratch
  engine in standard C++17 + `pthread`. A PR that adds a third-party library to
  the inference path will not be merged. (Dev-only tooling is negotiable.)
- **Every change keeps `make test` green**, and anything touching the math keeps
  the golden comparison passing (see below).
- **Match the surrounding style.** ~100-col lines, comments that explain *why*
  (the tricky invariant), not *what*. Look at a neighboring file first.

## The bar for correctness

This engine's whole credibility is that it matches llama.cpp numerically. If you
touch the forward pass, quantization, RoPE, attention, or the loader, re-run the
cross-engine validation before opening the PR:

```bash
# builds llama.cpp + the comparison harness — see golden/README.md for setup
python3 golden/validate_matrix.py --prompt "The capital of France is"
```

All formats should stay **PASS** with per-layer cosine ≈ 1.0. If a change makes a
layer diverge, that localizes the regression — include the diff in your PR.

## Good first issues

Great starting points that don't require deep context:

- **New quantization format** — add a dequant path in `src/quant.cpp` + a
  round-trip test in `tests/test_quant.cpp` (e.g. `IQ4_NL`, `Q4_K_S`).
- **More models in the registry** — extend `builtin_url()` in `sipllm` with
  other public, ungated small GGUFs.
- **NEON coverage** — port a scalar hot path in `src/neon.cpp` to a SIMD kernel,
  guarded by `LLM_HAVE_NEON`, with a `tests/test_neon.cpp` equivalence check.
- **x86 SIMD** — add an AVX2 path alongside the NEON one behind the `simd.h`
  abstraction.
- **Sampler features** — top-p / top-k / repetition penalty in `src/` +
  `include/llm/sampler.h`.
- **Docs** — a diagram of the streaming loader, or a walkthrough of one forward
  pass.

Look for issues tagged [`good first issue`](https://github.com/ankit1057/sipllm/labels/good%20first%20issue).

## Submitting a PR

1. Fork, branch from `main` (`git checkout -b feature/my-thing`).
2. Make the change; add/adjust tests.
3. `make test` (and the golden matrix if you touched the math).
4. Open the PR with a clear description of *what* and *why*. Reference any issue.
5. CI (build + tests on gcc and clang) must pass.

By contributing you agree your work is licensed under the project's
[MIT License](LICENSE).

## Reporting bugs

Open an issue with: the model/quant, the command, expected vs actual output, and
your platform (`uname -a`, compiler version). A minimal repro is worth a
thousand words.
