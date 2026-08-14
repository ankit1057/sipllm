# Changelog

All notable changes to SipLLM. The **North Star scorecard** below is refreshed at
the end of every optimization wave from measured data (see [CLAUDE.md](CLAUDE.md),
Rule 1). Peak RSS is the authoritative cross-runtime number from `/usr/bin/time -l`.

```
============================
SipLLM North Star   (measured 2026-07-27 · Apple M3 · warm cache · median-of-3)
============================
Peak RSS:              tinyllama 121 MB (stream) … 644 MB (fully resident)
                       smollm2    54 MB (stream) … 161 MB (fully resident)
                       vs llama.cpp CPU: 1356 MB / 546 MB  → 10–11× smaller at min budget
Resident Weights:      FLAT 1.5 MB across 4/16/32 toy layers (streaming thesis holds)
                       real: 37.6 MB (smollm2 1-layer) / 106.5 MB (tinyllama 1-layer)
                       pinned dial: up to 143 MB (smollm2) / 630 MB (tinyllama)
Decode tok/s:          Q8 `--fast` resident: smollm2 62→171 · tinyllama **50 vs llama.cpp 57**
                       RAM-budget dial (exact fp32): smollm2 53→66 · tinyllama 11→22
TTFT:                  smollm2 ~0.10 s · tinyllama ~0.68 s   vs llama.cpp 0.003 / 0.021 s
Prefill Throughput:    smollm2 50–67 tok/s · tinyllama 7–23 tok/s  vs llama.cpp 1680 / 238
Expansion Factor:      2.7× (smollm2) · 5.5× (tinyllama)  = disk / peak-RSS at min budget
Largest Runnable Model:MEASURED — Llama-2-13B (7.87 GB Q4) in **317 MB** peak RSS
                       (25×); Llama-3.1-8B (4.92 GB Q4) in 204 MB (24×), on a
                       16 GB Mac with ~3 GB free. Bounded by layer, not model size.
Energy / Token:        N/A (needs `sudo powermetrics`; never fabricated)

Current Largest Bottleneck:  Two fronts now that Q8 `--fast` is within ~12% of
                       llama.cpp. (1) 4-bit (Q4_K) models: the int-dot path does
                       not apply yet, so K-quant decode still uses fp32 dequant.
                       (2) Streaming (exceeds-RAM) regime: decode is disk-bandwidth
                       -bound (arithmetic intensity Θ(1)).
Estimated Gain if Fixed:  K-quant int-dot → Q8-class speed on 4-bit models = the
                       biggest RAM headline. Speculative streaming → amortizes
                       weight movement in the exceeds-RAM regime.
Why this is the next priority:  v0.4 proves the thesis (2.1× less RAM at ~88%
                       speed on a real 1.1B model). The Q4 int-dot path is the
                       strongest next demo; streaming speed is the long-term moat.
Confidence:            High — v0.4 numbers measured vs llama.cpp on tinyllama Q8;
                       re-validate at the start of the next wave.
```

## [0.5.0] — Plugin runtime + persistent context (2026-08-13)

### Wave 11 — Persistent context: session save/restore (Phase 5)

Turns the in-process cross-turn reuse (Wave 10) into CROSS-PROCESS persistence: a
Runtime serializes its committed tokens + resident KV cache to a "SIPS" v1 file
and restores them in a new process, so a session's context survives a restart —
the checkpoint primitive Nishachar needs.

- **`Runtime::save_session(path)` / `load_session(path)`** (CLI `--save-session F`
  / `--load-session F`). load restores committed tokens + KV + position; combined
  with `--reuse`, the next turn reprocesses only the changed tail.
- **`include/llm/session.{h,cpp}`** — dependency-free "SIPS" v1 serializer
  (magic / version / **model_id** / n_layers / kv_dim / seq_len / tokens / K / V),
  using only KVCache's public API. `session_read` hard-refuses on bad
  magic/version, a **model_id mismatch** (FNV-1a over arch+shape — guards against
  loading a KV saved from a different model), dim mismatch, or seq_len > ctx.
  (Residual: same-config-different-weights is the caller's responsibility — a
  session pairs with its model.)

**Verified on host (macOS / Apple M3):** new `test_session` (4 cases) — restore+reuse
logits equal a from-scratch full prefill within 1e-4; identical reprompt after
restore reuses all-but-one token; missing file refused; a same-shape different
model (differing only in `ffn_dim`, so the dims check passes) refused by model_id.
Full `make -j4 all && make test` green; every existing suite unchanged (default
byte-identical). CROSS-PROCESS smoke: process A `--save-session` → process B
`--load-session --reuse` reused 21/30 tokens. **Deferred:** compressed / partial
KV persistence, mesh-shared sessions.

### Wave 10 — Cross-turn context reuse (Kosh × RTK), opt-in

The first real Kosh × RTK cooperation and the seed of Phase 5 (persistent
context). `Runtime::set_context_reuse(true)` (CLI `--reuse`) makes a multi-turn
session reuse the KV of the longest common prefix already committed and reprocess
only the changed tail — the win a coding agent (Nishachar) needs, where the
system prompt + repo context are resent every turn.

- **Mechanism:** attention reads `kv_->k(layer,t)` for `t in [0,pos]`, so reuse is
  just `pos_ = LCP(new_tokens, committed_)` then `prefill(delta, start_pos=pos_)`
  — a natural extension of the existing multi-turn prefill; no KVCache surgery.
- **Invariant:** `reuse_ == false` (default) ⇒ `generate()` byte-identical; the
  `committed_` token list mirrors `KV[0,pos_)` (a small vector, never read on the default path). The
  HTTP server (`rt->reset()` per request) is unaffected.

**Verified on host (macOS / Apple M3):** new `test_reuse` (3 cases) — the golden
gate **reuse ≡ full-prefill** (last-position logits within 1e-4 of a from-scratch
prefill of the same context), identical-reprompt reuses all-but-one token, first
turn reuses nothing; `make -j4 all && make test` green with every existing suite
unchanged (byte-identical default). **Deferred:** Kosh/RTK plugins *governing*
reuse policy (hint/veto), cross-process / mesh KV, on-disk persistence.

### Wave 9 — Plugin seam: Kosh (context) + RTK (runtime/KV) as opt-in plugins

First seed of the Phase-2 plugin runtime (#51), proving the boundary that Kosh
(#52, context/token intelligence) and RTK (#53, runtime/KV intelligence) plug
into — designed **mesh-ready** (the request/KV contracts survive a future
distributed/persisted backing) but implemented locally first.

- **`PluginHost`** (`include/llm/plugin.h`, `src/plugin.cpp`) — in-process
  lifecycle + graceful fallback: a plugin whose `init` throws or returns false is
  disabled and logged, never fatal; a null plugin means the identity path.
- **Kosh** (`include/llm/kosh.h`, `src/kosh.cpp`) — `KoshPlugin::optimize()`
  transforms the token stream after tokenize, before prefill; POD
  `KoshRequest`/`KoshResult` are the future mesh serialization unit. v0 collapses
  over-long identical-token runs with full `tokens_in -> tokens_out` accounting.
- **RTK** (`include/llm/rtk.h`, `src/rtk.cpp`) — a runtime-state contract over an
  abstract `RtkKvView` (dims + bytes, never the concrete `KVCache`). v0 observes
  seq-len / peak-KV-bytes / step count. (Distinct from the disabled tool-calling
  `rtk_tools.cpp.bak`; that name is retired here in favor of runtime/KV per #53.)
- **CLI** — `--kosh [--kosh-max-run N]` and `--rtk` (both default OFF); a
  one-line metrics block prints when active.

**Regression oracle:** both plugins default OFF => `Runtime::generate()` is
byte-identical to the plugin-free engine. **Verified on host (macOS / Apple M3):**
`make -j4 all && make test` green (incl. new `test_plugin` — 9 cases: Kosh
collapse/identity/metrics, host lifecycle + throw/false-init fallback, RTK
accounting), existing e2e / 200-generation stress / sampler / arch suites
unchanged; CLI smoke exercised the `--kosh --rtk` path end-to-end (RTK observed
real step/seq/byte progression). **Not yet done:** dynamic loading / sandbox /
permissions (Phase 2 proper), real Kosh compression + RTK precision/persistence.

## [0.4.0] — Developer Preview (2026-07-27)

### Wave 8 — Flutter FFI bindings + on-device productization (M6.5, in progress)

Makes SipLLM a phone/watch citizen without touching the runtime's math. A stable
C ABI (`bindings/flutter/sipllm_flutter/ffi/sipllm_ffi.h`) wraps the C++ engine in
opaque handles + POD structs + a C token callback; the callback's `false` return
is the cancellation seam, and `--ram-budget` / threads / scheduler / Vulkan are
plumbed through a zero-init `sipllm_params` whose defaults reproduce the CLI.

- **Native shim + CMake** compile the *shared* `src/*.cpp` (no runtime code
  duplicated) into `libsipllm_ffi` for desktop, and per-ABI for Android
  (arm64-v8a / armeabi-v7a / x86_64) via the NDK.
- **Dart runtime** (`SipllmRuntime`) runs inference on a worker isolate, streams
  tokens over a `Stream<SipllmToken>`, and cancels mid-generate from the UI
  isolate via the thread-safe atomic in `sipllm_cancel` (no polling, no UI block).
- **On-device embeddings** via the final-layer hidden-state hook (L2-normalized),
  backed by a SQLite float32 vector store with cosine top-k search.
- **Resumable Hugging Face downloader** — multi-connection HTTP Range, sidecar
  resume across process restarts, pause/resume/cancel, sha256 verify.
- **Phone -> Wear OS transfer** over the Wearable Data Layer `ChannelClient`
  (Bluetooth for control, Wi-Fi High-Bandwidth for bulk). RFCOMM is intentionally
  avoided (Wear OS does not expose it); resumable header/ack protocol.

**Verified on host (macOS / Apple M3):** C-ABI smoke test (generate + cancel +
embed), the full Dart -> isolate -> FFI path, downloader (5 tests), embedding
store (13 tests), engine `make test` green, `flutter analyze` clean. **Not yet
runtime-verified:** the Android APK / on-device inference (POCO X6 Pro) and the
Wear transfer on the paired OnePlus Watch 2 — the remainder of M6.5.

### Wave 7 — Demo v1: `--fast` int8 SDOT kernel + near-parity Q8 decode

Closes the decode gap that stood between SipLLM and a "wow" demo. `linear()`
routed **every** quantized weight through fp32-dequant-then-dot — including Q8_0,
for which a tested int8 SDOT kernel already existed but was dead code. `--fast`
wires it in (opt-in; the exact fp32 path stays the default/oracle), and the
kernel was rebuilt for ILP (vector float accumulator, one horizontal reduce per
row) + hardware fp16 scale conversion.

**Demo — TinyLlama-1.1B Q8_0, Apple M3, `--ctx 512`, t=4, warm** (peak RSS from
`/usr/bin/time -l`):

| runtime | peak RSS | decode |
|:--------|---------:|-------:|
| llama.cpp (CPU) | 2326 MB | ~57 tok/s |
| **SipLLM** `--fast --ram-budget 1200M` (resident) | **1113 MB** | ~50 tok/s |
| **SipLLM** `--fast` (streaming) | **175 MB** | 6.8 tok/s |

**2.09× less RAM at ~88% of llama.cpp's decode** (12% slower — within 20%), or
**13× less RAM** streaming. Numerically equivalent (int8-activation dot, same
technique as llama.cpp; first-token predictions match; smollm2 `--fast` produced
byte-identical greedy output for 24 tokens).

**Added**
- `--fast` CLI flag (`LayerLoader::Options::fast_quant`) → int8 SDOT for Q8_0
  projections. Kernel: `matmul_q8_0_i8` rewritten (vector accumulate + hw fp16),
  +11–13% over the first wiring (smollm2 62→171 tok/s vs the fp32 path).
- Clean CLI summary block (peak RSS / pinned layers / decode / fast on-off).

**Fixed**
- Makefile now tracks header dependencies (`-MMD -MP` + `-include`). Editing a
  header previously left stale objects with mismatched struct layouts across
  TUs — a silent correctness hazard that produced spurious test failures.

**Verified** — full suite green on a clean build; `--fast` is opt-in so all
`1e-3`/bit-identical correctness tests are unchanged (no regressions).

## [Unreleased]

### Architecture — data-driven BlockSpec unification (PR #49 merged)

Replaces the per-architecture `Transformer::block_*()` dispatch with a single
data-driven `block()` driven by `ModelConfig::block_spec` (a `BlockSpec` recipe).
The eight architectures added on main (Kimi, DeepSeek, Yi, Baichuan, InternLM2,
GLM4, Gemma4, Phi4) were reconciled onto BlockSpec: the llama-family arches ride
the default RMSNorm+SwiGLU spec, Gemma4 → RMSNormGemma, Phi4 → fused QKV + fused
gate/up. Verified with a differential first-token-logits oracle (FNV-1a over raw
logit bytes, single-threaded): every one of 19 architectures produces a
BIT-IDENTICAL checksum before and after the refactor — reordering the same ops
into a BlockSpec changes no number in the forward pass.

### Nishachar Path A — autonomous goal→plan→act→verify loop (Phase 9, #58)

The first slice of Nishachar (the runtime's reference autonomous consumer): a
model-agnostic C++ agent loop over the tool-calling core (`tools.h`).
- `include/llm/nishachar.h` / `src/nishachar.cpp`: `Nishachar` registers tools
  (advertised schema + C++ handler), then `run(goal, generator)` renders a chat,
  asks the generator to act, parses a `<tool_call>` via `ToolParser`, dispatches
  to the handler, feeds the result back, and repeats until the model answers
  without a tool call or a step bound is hit. Returns a structured `AgentResult`
  (per-step trace + stop reason + `report()`). The generator is an abstraction
  (`AgentGenerator`) — bound to `Runtime::generate` in production, driven by a
  scripted generator in tests, so the loop links without the engine.
- `tests/test_nishachar.cpp` (9 tests): the loop state machine — immediate final,
  single/multi tool steps, max-steps bound, unknown-tool→final (the parser rejects
  unregistered tools), tool-error continue vs. stop, empty generation, report.
- `tools/nishachar_demo.cpp`: `nishachar_demo <model> "<goal>" [--max-steps N]` —
  loads a SipLLM model, registers demo tools (calc/echo/upper), drives the loop
  with greedy sampling, prints the structured report.

Scope: Path A is the loop mechanism + tool dispatch + structured report. The
compile/test/benchmark verification and the fs/shell/git/compiler capability
layer are the Plugin phase (#51) and later.

### Bigger-than-RAM demonstration (real models, measured)

The defining-capability proof: SipLLM streams models whose weights far exceed
available RAM at a peak RSS that tracks a single layer — not the model. On this
16 GB Mac with ~3 GB free (loading these resident is impossible), with
`--stream-lm-head --no-async`, `--ctx 512`, greedy:

| model | weights | peak RSS | model ÷ RSS | output |
|:------|--------:|---------:|------------:|:-------|
| TinyLlama-1.1B Q8 | 1.17 GB | 61 MB  | 19× | coherent |
| Llama-3.1-8B Q4   | 4.92 GB | 204 MB | 24× | "…a city of grandeur and beauty…" |
| Llama-2-13B Q4    | 7.87 GB | 317 MB | 25× | "…Paris. The currency of France is the Euro." |

Peak RSS grows with layer *width*, never model depth/total size (`toy_scaling`
stays flat vs depth). Streaming an off-cache model is disk-bound (<1 tok/s at
this bounded-memory extreme); `--ram-budget` trades RAM for speed from here.

**Fixed** — hardened `sipllm` model download (curl over HTTP/1.1 with
`--retry-all-errors`); a flaky HTTP/2 stream cancel had truncated a pull and the
partial `.part` was renamed to the final name.

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
