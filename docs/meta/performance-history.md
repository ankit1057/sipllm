# Performance history

SipLLM's trajectory, told through tags and optimization waves. The discipline is
`CLAUDE.md` Rule&nbsp;1: at the end of every wave the **North Star scorecard** in
`CHANGELOG.md` is refreshed from freshly measured data and the benchmark JSON is
committed under `bench/results/`. Peak RSS is always the authoritative
cross-runtime number from `/usr/bin/time -l`.

> [!MEASURED] The baseline is the present
> The scorecard below is the *current* measured state (2026-07-27, Apple M3,
> warm cache, median-of-3) — the baseline every future wave is compared against.
> Where older per-release RSS/TTFT/tok/s values were not captured at the time,
> this page says so rather than drawing a graph of numbers that were never
> measured.

## Current North Star scorecard (baseline)

<div class="stat-grid"><div class="stat"><div class="stat-value">317 MB</div><div class="stat-label">largest runnable</div><div class="stat-sub">Llama-2-13B (7.87 GB Q4) &mdash; 25&times; smaller than its weights</div></div><div class="stat"><div class="stat-value">121 MB</div><div class="stat-label">tinyllama peak RSS (stream)</div><div class="stat-sub">&hellip; 644 MB fully resident &middot; vs llama.cpp 1356 MB</div></div><div class="stat"><div class="stat-value">50 tok/s</div><div class="stat-label">tinyllama Q8 <code>--fast</code> resident</div><div class="stat-sub">vs llama.cpp 57 &mdash; ~88% at 2.1&times; less RAM</div></div><div class="stat"><div class="stat-value">N/A</div><div class="stat-label">energy / token</div><div class="stat-sub">needs <code>sudo powermetrics</code> &mdash; never fabricated</div></div></div>

Full scorecard as committed to `CHANGELOG.md`:

| Metric | Measured value |
|:--|:--|
| Peak RSS | tinyllama 121 MB (stream) … 644 MB (resident); smollm2 54 MB … 161 MB; vs llama.cpp CPU 1356 / 546 MB → 10–11× smaller at min budget |
| Resident weights | FLAT ~1.5 MB across toy 4/16/32 layers; real 37.6 MB (smollm2 1-layer) / 106.5 MB (tinyllama 1-layer); pinned dial up to 143 MB / 630 MB |
| Decode tok/s | Q8 `--fast` resident smollm2 62→171 · tinyllama 50 (vs llama.cpp 57); RAM-budget dial (exact fp32) smollm2 53→66 · tinyllama 11→22 |
| TTFT | smollm2 ~0.10 s · tinyllama ~0.68 s (vs llama.cpp 0.003 / 0.021 s) |
| Prefill throughput | smollm2 50–67 · tinyllama 7–23 tok/s (vs llama.cpp 1680 / 238) |
| Expansion factor | 2.7× (smollm2) · 5.5× (tinyllama) — disk ÷ peak-RSS at min budget |
| Largest runnable | Llama-2-13B (7.87 GB Q4) in 317 MB (25×); Llama-3.1-8B (4.92 GB Q4) in 204 MB (24×) |
| Energy / token | N/A (needs `sudo powermetrics`) |

## Release trajectory

```text
v0.1.0  ──▶  v0.1.1  ──▶  v0.4.0 (Developer Preview, 2026-07-27)
 first        patch        streaming thesis proven + Flutter FFI
 tag                       (Waves 6–8)
```

Three tags mark the line: **v0.1.0** (first tag), **v0.1.1** (patch), and
**v0.4.0** — the Developer Preview that carries the bounded-memory waves and the
on-device productization work.

> [!NOTE] Per-release metrics are being backfilled
> The committed measured record (`bench/results/`, North Star scorecard) captures
> the *current* state comprehensively, but the project did not snapshot peak-RSS /
> TTFT / decode numbers at each early tag — the reproducible harness itself only
> landed at #33 (v0.2 baseline). So there is no honest per-tag time series for
> v0.1.0 → v0.1.1; those historical values are being backfilled by re-running
> `scripts/bench.sh` against the tagged commits. This page will not fabricate a
> trend line from numbers that were never taken.

## Wave milestones (v0.4.0)

The measured progress lives in the waves, not the tags. Each entry is grounded in
`CHANGELOG.md`.

### Wave 6 — `--ram-budget`, the RAM↔speed dial (#37)

Turned the fixed 2-buffer streaming window into a hard byte ceiling: pin as many
contiguous hot layers as fit, stream the rest, `resident_bytes() ≤ budget`
always. This is the capability no other runtime offers — a *tunable* continuum
between bounded RSS and speed. Measured (M3, warm, ctx 512, median-of-3):

| Model | Budget | Pinned | Decode tok/s | Streamed | Peak RSS |
|:--|--:|--:|--:|--:|--:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | 22.1 | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | 65.9 | 113 MB | 161 MB |

Decode up to **+95%** (tinyllama) / **+24%** (smollm2); streamed I/O **−96%**;
logits bit-identical across every budget (pinning is a pure cache). See the
[memory planner](memory-planner.html).

### Wave 7 — `--fast` int8 SDOT kernel, near-parity Q8 decode

`linear()` had routed *every* quantized weight through fp32-dequant-then-dot,
including Q8_0 — for which a tested int8 SDOT kernel existed but was dead code.
`--fast` wires it in (opt-in; the exact fp32 path stays the oracle) and the
kernel was rebuilt for ILP (vector accumulator, one horizontal reduce per row) +
hardware fp16 scale conversion. Demo — TinyLlama-1.1B Q8_0, M3, ctx 512, t=4, warm:

| Runtime | Peak RSS | Decode |
|:--|--:|--:|
| llama.cpp (CPU) | 2326 MB | ~57 tok/s |
| SipLLM `--fast --ram-budget 1200M` (resident) | 1113 MB | ~50 tok/s |
| SipLLM `--fast` (streaming) | 175 MB | 6.8 tok/s |

**2.09× less RAM at ~88% of llama.cpp's decode**, or **13× less RAM** streaming.
Numerically equivalent — smollm2 `--fast` produced byte-identical greedy output
for 24 tokens; first-token predictions match. See [Quantization](quantization.html).

### Wave 8 — Flutter FFI + on-device productization (M6.5, in progress)

Makes SipLLM a phone/watch citizen without touching the runtime's math: a stable
C ABI (`sipllm_ffi.h`) wraps the C++ engine in opaque handles + POD structs + a C
token callback (whose `false` return is the cancellation seam), a Dart
`SipllmRuntime` that runs inference on a worker isolate and streams
`Stream<SipllmToken>`, on-device embeddings backed by a SQLite float32 vector
store, a resumable Hugging Face downloader, and phone→Wear OS transfer.

> [!WARNING] Host-verified, not yet on-device
> Wave 8 is **verified on host** (macOS / Apple M3): C-ABI smoke test, the full
> Dart → isolate → FFI path, downloader (5 tests), embedding store (13 tests),
> engine `make test` green, `flutter analyze` clean. It is **not yet
> runtime-verified** on device — the Android APK / on-device inference (POCO X6
> Pro) and Wear transfer (OnePlus Watch 2) are the remainder of M6.5. No
> Android/phone performance numbers are quoted anywhere; they do not exist yet.

## What comes next

Per the scorecard's own "current largest bottleneck": (1) **K-quant int-dot** —
the `--fast` path is Q8_0-only, so 4-bit (Q4_K) decode still uses fp32 dequant;
extending int-dot there is the biggest pending RAM headline. (2) **Speculative
streaming** to amortize weight movement in the exceeds-RAM regime, where decode
is disk-bandwidth-bound (arithmetic intensity Θ(1)). See the
[roadmap](roadmap.html).
