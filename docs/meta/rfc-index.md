# RFC / issue index

A chronological ledger of the numbered issues and RFCs that shaped SipLLM,
reconstructed from git history (`git log`, 35 commits, tags v0.1.0 → v0.1.1 →
v0.4.0). Each row is a landed change; every entry below is **Implemented**
(present in `main`). Two carry formal design-doc ids referenced elsewhere in the
codebase — **RFC-003** (KV grow-on-demand, #35) and **RFC-007** (single-pass
prefill, #36).

> [!NOTE] What this index is (and is not)
> This is a record of *shipped* work, grounded one-to-one in commits. The project
> uses the standard lifecycle framework — **accepted → implemented**, with
> **rejected** and **superseded** reserved for proposals that do not land. No
> rejected or superseded entries are listed here because none are sourced in the
> current history; this ledger will not invent them.

## Landed RFCs / issues

| Issue / RFC | Title | Status |
|:--|:--|:--:|
| #17 | Dispatch on `general.architecture` (enabling refactor) | Implemented |
| #18 | Apply llama3 RoPE frequency scaling (Llama-3.x) | Implemented |
| #19 | Repetition penalty + wire top-k / top-p / repeat CLI flags | Implemented |
| #20 | Make async-prefetch e2e test deterministic | Implemented |
| #21 | More public / ungated Llama-arch models in the registry | Implemented |
| #22 | Streaming layer loader walkthrough + diagram (docs) | Implemented |
| #23 | x86-64 AVX2 / FMA path in the `simd.h` kernels | Implemented |
| #24 | CI: publish prebuilt linux / macOS x86_64 + aarch64 releases | Implemented |
| #25 | Architecture: Mistral / Mistral-Nemo | Implemented |
| #26 | Architecture: Qwen2 / Qwen2.5 (attention QKV bias) | Implemented |
| #27 | Architecture: Gemma 2 (GeGLU, soft-capping, embedding scale) | Implemented |
| #28 | Architecture: Gemma 3 text (QK-norm + per-layer local/global RoPE) | Implemented |
| #29 | Architecture: Phi-3 (fused QKV / gate-up, partial-rotary RoPE) | Implemented |
| #30 | Architecture: Mixtral / MoE (sparse expert routing, streamed) | Implemented |
| #31 | Architecture: GPT-2 and Phi-2 (LayerNorm long tail) | Implemented |
| #32 | Eliminate intermittent uninitialized-stack SIGSEGV + detection gates | Implemented |
| #33 | Reproducible cross-runtime benchmark harness (v0.2 baseline) | Implemented |
| #34 | `current_rss_bytes()` works on macOS via mach `task_info` | Implemented |
| #35 | KV cache grow-on-demand allocation (**RFC-003**) | Implemented |
| #36 | Single-pass batched prefill (stream model once, not P×) (**RFC-007**) | Implemented |
| #37 | `--ram-budget`: hard peak-RSS ceiling + partial layer residency | Implemented |
| #46 | NEON Q4_K / Q6_K dequant (byte-identical, decode +57%) | Implemented |
| #47 | `--stream-lm-head`: opt-in streamed LM head (−43% peak RSS) | Implemented |

## Reading the ledger

Three arcs run through the numbers, and they map to the project's phases:

<div class="card-grid"><div class="card"><h3>Correctness &amp; breadth (#17&ndash;#31)</h3><p>Architecture dispatch (#17) unlocked nine model families — Mistral, Qwen2/2.5, Gemma&nbsp;2/3, Phi-2/3, GPT-2, Mixtral/MoE — plus llama3 RoPE (#18) and sampler features (#19). The engine learned to read many models before it learned to run them lean.</p></div><div class="card"><h3>Hardening &amp; measurement (#20, #32&ndash;#34)</h3><p>The SIGSEGV fix (#32) eliminated the uninitialized-read class at source; the bench harness (#33) and macOS RSS (#34) made "measured, not guessed" real by giving every claim a cross-runtime peak-RSS number.</p></div><div class="card"><h3>Bounded memory (#35&ndash;#37, #46&ndash;#47)</h3><p>KV grow-on-demand (#35 / RFC-003), single-pass prefill (#36 / RFC-007), the <code>--ram-budget</code> dial (#37), K-quant NEON (#46), and the streamed LM head (#47) are the streaming-thesis wave — each a measured peak-RSS or throughput win.</p></div></div>

> [!TIP] Cross-references
> The KV and prefill RFCs each have a full design page — see
> [KV cache](kv-cache.html) and the [single-pass prefill journal](j-prefill.html).
> The `--ram-budget` dial is documented in the [memory planner](memory-planner.html);
> the streamed LM head in the [LM-head journal](j-lm-head.html). For the measured
> before/after of each landed change, see the
> [performance history](performance-history.html).
