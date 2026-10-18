# Changelog

All notable changes to SipLLM. The **North Star scorecard** below is refreshed at
the end of every optimization wave from measured data (see [CLAUDE.md](CLAUDE.md),
Rule 1). Peak RSS is the authoritative cross-runtime number from `/usr/bin/time -l`.

```
============================
SipLLM North Star   (measured 2026-10-19 · Apple M3 · 4 thr · ctx 512 · warm · load-guarded)
============================
Peak RSS:              tinyllama Q4_K_M (669 MB file): 122 MB (stream) … 645 MB (fully resident)
                       smollm2-135m Q8_0: 54 MB (stream) … 162 MB (resident)
                       vs llama.cpp CPU (ggml 0.27.0): 1323 MB / 515 MB → 10.8× / 9.5× smaller at min budget
Resident Weights:      FLAT 1.5 MB across 4/16/32 toy layers (streaming thesis holds)
                       real: 37.6 MB (smollm2 1-layer) / 106.5 MB (tinyllama 1-layer)
                       pinned dial: up to 143 MB (smollm2) / 630 MB (tinyllama)
Decode tok/s:          tinyllama Q4_K_M resident: fp32 21.0 → `--fast` int8 **59.6** (llama.cpp 88.3 → 67%)
                       tinyllama Q4_K_M stream (122 MB): 11.7 (I/O-bound); `--spec 5` on copy-heavy output: 30.7
                       smollm2 Q8_0 stream 58.9 / resident 67.6 (llama.cpp 366)
TTFT:                  tinyllama 123 ms stream · 68 ms resident `--fast`   vs llama.cpp 28 ms
Prefill Throughput:    tinyllama 37–51 (stream) · 73 (resident `--fast`) tok/s   vs llama.cpp 266
Expansion Factor:      5.5× (tinyllama Q4_K_M) · 2.7× (smollm2)  = disk / peak-RSS at min budget
Largest Runnable Model:MEASURED 2026-07-27 — Llama-2-13B (7.87 GB Q4) in 317 MB peak RSS (25×);
                       Llama-3.1-8B (4.92 GB Q4) in 204 MB (24×). Bounded by layer, not model size.
Energy / Token:        N/A (needs `sudo powermetrics`; never fabricated)

Current Largest Bottleneck:  Two measured gaps against the standing goal (beat LiteRT /
                       MediaPipe on ARM + AMD CPUs). (1) No on-device number exists:
                       nothing has run on a phone or an AMD host; the AVX2 K-quant
                       path is cross-compiled only. (2) Resident 4-bit decode is 67% of
                       llama.cpp: remaining cost is the fp32 attention/KV path and
                       per-token lm_head (Q6_K, now int8) plus prefill at 73 vs 266.
Estimated Gain if Fixed:  (1) turns the RAM thesis into a competitor claim or kills
                       it — the only step that can. (2) ~1.3× decode (to ~parity), 3× prefill.
Why this is the next priority:  The runtime now has the kernel it lacked; what it
                       lacks is evidence on the hardware the goal names. A number on a
                       Snapdragon/Dimensity phone and on a Zen laptop decides the
                       roadmap more than any further M3 tuning.
Confidence:            High on every number above (quiet CPU, load recorded, 3–5 reps,
                       JSON committed). Zero on phone/AMD — not measured.
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

### Wave 12 — edge truth: K-quant integer kernels, `--spec`, planner fix, honest claims (2026-10-19)

Standing goal set by the maintainer this session: SipLLM is complete when it is a
*scientific utility* that runs LLMs better than LiteRT / MediaPipe LLM Inference on
ARM and AMD CPUs — an order of magnitude less RAM at competitive speed, on the bet
that storage is cheap and RAM/VRAM are not. Every number below is Apple M3,
4 threads, `--ctx 512`, greedy, warm cache, peak RSS from `/usr/bin/time -l`,
commit named in the JSON. Model: **TinyLlama-1.1B Q4_K_M (669 MB)** — the 4-bit
file a phone runs — unless stated.

**Measured first (pre-change baseline, `bench/results/*2026-10-19.json`):**

| runtime | peak RSS | decode | prefill | TTFT |
|:--|--:|--:|--:|--:|
| llama.cpp CPU (`-ngl 0 -t 4`, ggml 0.27.0) | 1323 MB | 88.3 tok/s | 266 tok/s | 28 ms |
| SipLLM streaming (`--ram-budget 0`) | 122 MB | 11.7 tok/s | 40.5 tok/s | 123 ms |
| SipLLM resident (`--ram-budget 768M`, 22/22) | 645 MB | 21.5 tok/s | — | — |

RAM: 10.8× less streaming, 2.05× less resident. Speed at equal residency: 24% of
llama.cpp — the 4-bit path dequantized to fp32. That gap is the named hard part.

**Added**
- **Q4_K and Q6_K × int8-activation kernels** (`matmul_q4_K_i8`, `matmul_q6_K_i8`,
  `src/neon.cpp`) under the existing opt-in `--fast` gate. Activation quantized once per
  token to per-32 int8 (shared with the Q8_0 kernel); each sub-block is an SDOT on the
  raw nibbles plus two scalar FMAs (`xs·(d·sc·Σq·xq − dmin·m·Σxq)`); Q6_K takes the
  dot per 16-wide scale group with the −32 offset folded into a scalar term. ARM NEON
  `vdotq_s32`; x86-64 AVX2 via `_mm256_maddubs_epi16` (its (u8,s8) contract is exactly
  nibbles × int8; pair sums ≤ 16002, no saturation). Every quantized projection in a
  Q4_K_M file (135 Q4_K + 21 Q6_K tensors) now runs an integer dot with `--fast`.
  **Default path byte-for-byte unchanged**; fp32 dequant stays the oracle.
  - Correctness: `test_neon` (7 cases) — synthetic well-formed K-quant rows vs the exact
    fp32 dequant of the same bytes within the int8-activation bound; threaded ≡ single
    bit-identical; 6-bit scale/min packing round-trips. Real model, greedy 48 tokens,
    `--fast` vs fp32: first token identical 6/6 prompts, full continuation identical
    5/6, the sixth diverging at a near-tie 56 chars in (same class as Q8_0 `--fast`).
  - **AVX2 path is cross-compiled only (clang `-target x86_64`), NOT run — no x86 host
    here.** The AMD half of the goal is unverified until someone runs `make test` on one.
  - Measured, 48 tokens, median of 3:

    | `--ram-budget` | pinned | fp32 dequant | **`--fast` int8** | gain | peak RSS | prefill (fp32 → fast) |
    |:--|--:|--:|--:|--:|--:|--:|
    | 768M (resident) | 22/22 | 21.0 tok/s | **59.6 tok/s** | **2.83×** | 645 MB | 44 → 73 tok/s |
    | 512M | 13/22 | 16.5 tok/s | 23.8 tok/s | 1.45× | 482 MB | 37 → 58 tok/s |
    | 0 (streaming) | 0/22 | 11.7 tok/s | 11.9 tok/s | I/O-bound | 122 MB | 37 → 51 tok/s |

    Resident Q4_K_M now decodes at **67% of llama.cpp (59.6 vs 88.3 tok/s) in 2.05× less
    RAM**, up from 24%. TTFT 113 → 68 ms. Peak RSS unchanged (the kernel adds one int8
    copy of the activation, ~2–6 KB). `.claude-artifacts/fast-q4km-2026-10-19.tsv`,
    median of 3, load₁ₘ 3.6–5.0 at each point.
- **`--spec N` — prompt-lookup speculative decoding reachable from the CLI** (the
  Runtime API from 77cf035 had no flag). Verify loop hardened: EOS inside an accepted
  draft now stops; `max_new` honoured inside the run; one `++pos_` per consumed batch
  element with the KV truncated on first rejection; `GenStats` reports drafted /
  accepted / verify passes. `test_spec_equiv` (5 cases) proves greedy output is
  token-for-token identical to plain decoding for K=1/3/5 with drafts actually
  verified, `max_new` respected, a seeded sampling run identical (one RNG draw per
  verify step), stop callback honoured. Real model: greedy output identical with and
  without `--spec 5`.
  - Measured, streaming regime (`--ram-budget 0`, 122–133 MB, copy-continuation
    prompt, 48 tok, quiet CPU): plain 9.7 → `--spec 3` **27.8 tok/s**, 33/36 drafts
    accepted in 12 verify passes → `--spec 5` **30.7 tok/s (3.2×)**, 37/42 accepted in
    9 passes. Free text ("Once upon a time"): 1/7 accepted, 12.1 vs 11.7 — no gain, no
    loss. Each verify pass streams every layer **once** for K+1 positions — the
    "speculative streaming" item from the roadmap, now real for copy-heavy output
    (code edits, summaries, retrieval-grounded answers).
    Resident regime (fp32 dequant path, `--ram-budget 768M`), same run, copy-heavy
    prompt: plain 12.4 → `--spec 3` 21.9 → `--spec 5` **32.5 tok/s (2.6× within-run)**.
    Absolute numbers in that block are depressed — the fp32 resident baseline read
    21.0 tok/s twenty minutes earlier and 12.4–15.6 here (thermal drift on a fanless
    M3; the load guard catches foreign load, not heat) — so only the within-run ratio
    is claimed. The gain is real because the verify pass goes through the batched
    matmul, which dequantizes each weight row **once** for all K+1 positions; on the
    fp32 path that dequant is the dominant per-token cost. Free text: no change.

**Fixed**
- **`--ram-budget` dial was non-monotonic below ~512M** (planner `src/mem_plan.cpp`):
  it scored "one more pinned layer" above "two stream buffers", so every partial
  budget traded the async double-buffer for a pinned layer and lost: 128M → 8.1,
  256M → 9.2, 384M → 10.9, 512M → 13.7 tok/s, **all slower than pure streaming at
  12.4 tok/s in 122 MB**. Model fitted from the sweep: 1 buffer ≈ 5.8 ms/layer
  (I/O serial with compute), 2 buffers ≈ 3.7 ms (overlapped), pinned ≈ 2.1 ms, so one
  buffer only wins when ≤ 2 layers remain streamed. Scoring is now strict 64-bit
  tiers (ctx > all-pinned > resident head > 2 buffers > pinned count); `--ram-budget
  0` never consults the planner and is unchanged. Three new planner tests.
  Re-sweep after the fix (quiet CPU, same run, `bench/results/ram-budget-*2026-10-19.json`
  at b262811), tinyllama Q4_K_M decode vs the unbounded stream in the same run:

  | `--ram-budget` | before: pinned / buffers / tok/s / vs stream | after: pinned / buffers / tok/s / vs stream |
  |:--|:--|:--|
  | 128M | 1 / 1 / 8.1 / −34% | 0 / 1 / 9.7 / −10% (a 2-deep ring does not fit in 128M; ceiling honoured) |
  | 256M | 4 / 1 / 9.2 / −26% | 3 / 2 / 10.6 / −1% |
  | 384M | 9 / 1 / 10.9 / −12% | 8 / 2 / 12.9 / **+20%** |
  | 512M | 14 / 1 / 13.7 / +11% | 13 / 2 / 14.7 / **+36%** |
  | 768M | 22 / all / 21.5 | 22 / all / 21.6 |

  The dial is monotonic from 256M up; smollm2 at 128M went 54.1 → 60.8 tok/s (now
  above its own stream rate of 58.9). Peak RSS stays under every budget.
- `test_spec_decoder` had been **red since 77cf035** (expected a 2-token draft where
  standard prompt lookup returns 3); the tree is green again with four more cases.
- `scripts/bench.sh` toy-scaling rows at 32 layers recorded **0 generated tokens /
  0 tok/s** as if measured (greedy on random weights hits EOS at step one); toy rows
  now use seeded sampling.
- `install.sh` claimed `VERSION=1.0.0`, a release that does not exist, so every
  install silently fell back to a source build; it now says `0.6.0-dev` and why.

**Measurement validity (new harness guard)**
- `bench.sh` and `bench_ram_budget.sh` record the 1-minute load average and the
  hungriest foreign process in the JSON `meta`, and **refuse to run when load exceeds
  the thread count** (`LOAD_OVERRIDE=1` tags the run `contended`). Mid-session a
  foreign JVM and then Spotlight owned the CPU and produced "resident decode 6.4
  tok/s" — a 3× regression that did not exist. Discarded; numbers above are from quiet
  runs.

**Claims brought back to evidence**
- README: no more "Production Ready v1.0.0" (latest tag is v0.5.0; `main` is the 1.0
  candidate); release badge is now GitHub's live one; demo table rebuilt from today's
  Q4_K_M measurements with date + JSON named; the Q8_0 `--fast` table kept with its
  2026-07-27 provenance; "K-quant SIMD `sdot`/`i8mm`" claims rewritten to what exists;
  Vulkan marked detect-only; no phone numbers claimed (none exist); a standing "what a
  phone / AMD user still cannot do" line.
- Published competitor anchor, **not our hardware** (hf.co/litert-community/Gemma3-1B-IT,
  Samsung S24 Ultra CPU, XNNPACK 4 threads): Gemma3-1B dynamic_int4 — MediaPipe 138 pp /
  50 tg tok/s in 982 MB; LiteRT-LM int4-QAT 379 pp / 55 tg in 1009 MB; model files
  529–657 MB. They hold 1.5–1.9× the model in RAM; SipLLM streaming holds 0.18×.

**Still cannot (one line, per the standing goal):** a phone user has no SipLLM number
at all yet (no on-device run); an AMD user has an untested AVX2 path; on M3 the
resident 4-bit decode is 67% of llama.cpp, prefill 27%, TTFT 2.4× longer, and no
same-hardware LiteRT-LM head-to-head has been run — so "better than LiteRT /
MediaPipe" is a RAM claim with evidence and a speed claim without it.

**Deferred:** same-hardware head-to-head vs LiteRT-LM (its macOS CLI exists; not run);
Q5_K / Q8_K integer paths; VNNI (Zen4+) fast path; batched lm_head projection in
`spec_forward` (streams the head K+1× per pass today).

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

### Nishachar Path B — OpenAI-compatible /v1/chat/completions (Phase 9, #58)

Exposes the runtime over an OpenAI-compatible chat endpoint on the built-in
dependency-free HTTP server (`make server` → `llm_server`), in two modes:
- **passthrough (default):** one model turn per request; if the model calls a
  client-provided tool, the response returns OpenAI-format `tool_calls` for the
  client to execute (standard OpenAI tool protocol). Plain chat returns content.
- **agent (`"agent": true`):** the server runs the Nishachar loop with its own
  tools (calc/echo/upper) and returns the finished answer.

`include/llm/openai_api.h` / `src/openai_api.cpp`: a zero-dependency mapping
between the OpenAI wire format and the internal types — `parse_chat_request`
(messages, tools, params, agent flag) and `build_chat_response` (content or
`tool_calls`, usage). `tests/test_openai_api.cpp` (10 tests) covers request
parsing (roles, tool typing, malformed/empty → error, strings with braces) and
response building (content + tool_calls shapes, escaping). The route is `POST
/v1/chat/completions` in `server/server.cpp`, guarded by the existing
single-generation mutex.

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
