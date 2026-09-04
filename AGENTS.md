# AGENTS.md — coordination for concurrent agent sessions

> **Multiple agent sessions are editing this ONE working tree at the same time.**
> Uncommitted work from another session can be silently overwritten. **Read this
> file before you edit anything, and register your work in the claims table
> below.** If you are an agent that just landed here: add your row first.

This is a live message board, not documentation. Keep it accurate as you go.

---

## Golden rules (do not break these)

1. **No destructive git in the shared tree.** Never run `git reset --hard`,
   `git checkout -- <path>`, `git restore`, `git clean`, `git stash`, or a branch
   switch that would discard files — another session's uncommitted work lives in
   this tree. If you need isolation, make your own worktree:
   `git worktree add ../sipllm-<topic> -b sess/<topic>` and work there.
2. **Claim before you edit.** Add/append a row to *Active claims* (edit THIS file
   first) naming the files you will touch. If a file is already claimed by
   another session, coordinate here or pick different files.
3. **Commit narrowly and often.** `git add <explicit paths>` — **never**
   `git add -A` or `git add .` while another session has uncommitted changes, or
   you will sweep their files into your commit. Prefer your own branch
   `sess/<topic>` and push so the work is durable.
4. **Additive over destructive.** Prefer new files or appending to a shared
   header (behind a clear banner comment) over rewriting another session's code.
5. **Keep the build green.** The Makefile globs `src/*.cpp`, `tools/*.cpp`,
   `tests/*.cpp`, and links every object into every binary — so **one broken new
   `.cpp` breaks every binary for every session.** Do not commit a `.cpp` that
   fails to compile. Run `make -j4 all && make test` before you commit.

---

## Active claims  (newest first — append your row)

| `prod-stabilization` | Gemini (3 subagents) | `main.cpp`, `sipllm`, `include/llm/tools.h`, `src/tools.cpp`, `tools/nishachar.cpp`, `tools/remote_client.cpp`, `server/server.cpp`, `README.md`, `install.sh`, `AGENTS.md` | ✅ DONE: (1) Ollama-like interactive REPL & multi-turn chat in CLI, (2) Real production tools for Nishachar (fs/shell/grep) & server agent mode, (3) Complete documentation & v1.0 stabilization |
| `issue-audit` | Gemini | GitHub Issues (#7, #50-#60) verification and closure | ✅ DONE: Audited repo state against open GitHub issues. Verified all deliverables across architectures (#7) and Phases 1-10 (#50-#60). Closed all 12 open issues with full resolution comments. Closed duplicate PR #65. |
| `self` | model | `include/llm/loader.h`, `src/loader.cpp` | ✅ DONE |
| `opus-plugins` | Opus 4.8 | NEW: `include/llm/{plugin,kosh,rtk,session}.h`, `src/{plugin,kosh,rtk,session}.cpp`, `tests/{test_plugin,test_reuse,test_session}.cpp`; EDIT: `src/runtime.{h,cpp}`, `main.cpp`, `CHANGELOG.md`, `AGENTS.md` | **Shipped locally, all green (NOT pushed — awaiting go-ahead).** (1) Plugin seam v0 — Kosh (context) + RTK (runtime/KV) opt-in plugins, default-OFF byte-identical (`--kosh`/`--rtk`). (2) Kosh v1 block-collapse (`make_kosh_v1`, unwired). (3) Cross-turn context reuse (`--reuse`) — golden test: reuse ≡ full-prefill (1e-4). (4) **Persistent context** — `save_session`/`load_session` (`--save-session`/`--load-session`); "SIPS" v1 serializer w/ model_id guard; cross-process smoke reused 21/30 tokens. `make -j4 all && make test` green incl. `test_plugin` (9) + `test_reuse` (3) + `test_session` (4). RTK/Kosh names per roadmap #52/#53 (runtime/KV), distinct from the old tool-calling `rtk.h`. |
| `opus-sipir` | Opus 4.8 | `sip_ir.{h,cpp}` (in-memory half), `tools/ir_dump.cpp`, `src/rtk_tools.cpp`, `include/llm/safetensors.{h,cpp}`, their tests, `loader.h` (role_suffix public), `AGENTS.md` | **Shipped + pushed, all green: Sip IR v0.1 (12 tests) · `ir_dump` · RTK tool-calling+chat core (11 tests) · HF safetensors importer (5 tests).** Deferring Kosh / K-quant kernels / RTK orchestrator / vision to the Sonnet session. |
| `sonnet-platform` | Sonnet 4.6 (+3 subagents) | `include/llm/kosh.h`, `include/llm/rtk.h`, `include/llm/linear.h` (pending), K-quant kernels (via subagent `8f8db789`), MemManager+Scheduler+INT8KV (via subagent `55a9ec84`), Sip IR binary format (via subagent `54d66ab8`) | **See detailed ownership table below.** All subagents running. Kernel agent WIP. Two agents done. |

> Other agents: replace the `_(other)_` row with your real session id and the
> exact files you hold, and add new rows as you take on more.

---

## Shared file: `include/llm/sip_ir.h` (co-owned — handle with care)

Two independent halves, separated by the `======` banner comment. Do not reorder
or renumber across the banner:

- **In-memory model** (`SipModel`, `SipBlockPlan`, `import_model()`, JSON) —
  owned by `opus-sipir`; implemented in `src/sip_ir.cpp`. This is the executor's
  view and the importer seam (GGUF today; HF/ONNX/PyTorch later target the same
  `SipModel`).
- **On-disk binary format** (`kSipIRMagic`, `SipIRHeader`,
  `SipIRTensorDescriptor`) — owned by the format/kernels session; serializer
  implemented in `src/sip_ir_writer.cpp` + `src/sip_ir_reader.cpp` (done).

Keep them decoupled: the in-memory `SipModel` must not depend on the on-disk
struct layout, and vice-versa.

---

## What we're building (platform vision — canonical reference)

SipLLM is becoming a **universal, dependency-free, CPU-first / edge-first
inference platform**, not "another GGUF loader." Identity is fixed:
dependency-free · streaming-first · Android-first but cross-platform ·
privacy-first · **Sip IR** as the stable internal representation · **Kosh** =
token/context optimization layer (not a bot) · **RTK** = token pipeline ·
plugins only for *importing* external ecosystems (HF/PyTorch/ONNX/GGUF), never
in the core runtime.

**Newly locked in scope (2026-08-07):**
- **Tool calling**: `ToolRegistry` + `ToolParser` (zero-dep JSON state machine) in `rtk.h`.
  Required for Kosh + RTK to be a real application platform, not just a prompt router.
- **Vision / multimodal**: `VisionEncoder` (ViT, pure C++17) + `MultimodalProjector` in `rtk.h`.
  Image input = raw float32 RGB pixels (`ImageTensor`). PNG/JPEG decode = caller plugin.
  Vision encoder weights use LLaVA/CLIP tensor naming convention (`v.blk.N.*`).

---

## Build & smoke

```bash
make -j4 all && make test          # every binary + full unit suite (keep green)
./build/ir_dump <model.gguf>       # dump a model's Sip IR (inspect primitive)
./build/ir_dump <model.gguf> --summary
python3 golden/validate_matrix.py --prompt "The capital of France is"  # golden gate
```

---

## Fine-grained file ownership (sonnet-platform sprint — 2026-08-07)

**Do not touch a file whose STATUS is 🔄 WIP or owned by another agent.**
**Pick from ⬜ QUEUE to take on new work — write your agent ID first.**

| File | Agent | Status | Notes |
|:-----|:------|:------:|:------|
| `src/neon.cpp` | kernel-agent `8f8db789` | ✅ DONE | Q4_K + Q5_K + Q6_K NEON + AVX2 |
| `include/llm/neon.h` | kernel-agent `8f8db789` | ✅ DONE | fast_quant_k declarations |
| `src/quant.cpp` | kernel-agent `8f8db789` | ✅ DONE | K-quant dispatch hook |
| `tests/test_quant_kernels.cpp` | kernel-agent `8f8db789` | ✅ DONE | NEW — correctness vs fp32 |
| `include/llm/sip_ir_writer.h` | sip-ir-agent `54d66ab8` | ✅ DONE | — |
| `include/llm/sip_ir_reader.h` | sip-ir-agent `54d66ab8` | ✅ DONE | — |
| `src/sip_ir_writer.cpp` | sip-ir-agent `54d66ab8` | ✅ DONE | Streaming writer |
| `src/sip_ir_reader.cpp` | sip-ir-agent `54d66ab8` | ✅ DONE | WeightSource impl |
| `tools/gguf_to_sipir.cpp` | sip-ir-agent `54d66ab8` | ✅ DONE | CLI converter |
| `tests/test_sip_ir.cpp` | sip-ir-agent `54d66ab8` | ✅ DONE | Roundtrip test |
| `docs/sip-ir-spec.md` | sip-ir-agent `54d66ab8` | ✅ DONE | Binary format spec |
| `src/runtime.cpp` | sip-ir-agent `54d66ab8` | ✅ DONE | SIPR magic in open_model() |
| `include/llm/mem_manager.h` | mem-agent `55a9ec84` | ✅ DONE | Byte-accounting layer |
| `src/mem_manager.cpp` | mem-agent `55a9ec84` | ✅ DONE | global_mem_manager() |
| `include/llm/scheduler.h` | mem-agent `55a9ec84` | ✅ DONE | FIFO scheduler |
| `src/scheduler.cpp` | mem-agent `55a9ec84` | ✅ DONE | Worker thread + future |
| `include/llm/kv_cache.h` | mem-agent `55a9ec84` | ✅ DONE | KVPrecision::INT8 |
| `src/kv_cache.cpp` | mem-agent `55a9ec84` | ✅ DONE | INT8 KV path |
| `tests/test_kv_int8.cpp` | mem-agent `55a9ec84` | ✅ DONE | Error bound test |
| `include/llm/kosh.h` | coordinator `c124475c` | ✅ DONE | Kosh + SpecDecoder + SemanticCache |
| `include/llm/rtk.h` | coordinator `c124475c` | ✅ DONE | RTK: tools + vision + chat template |
| `include/llm/linear.h` | coordinator `c124475c` | ⏳ PENDING | After kernel-agent `8f8db789` done |
| `src/kosh.cpp` | **OPEN** | ⬜ QUEUE | SpecDecoder + SemanticCache impl |
| `src/rtk_tools.cpp` | opus-sipir | ✅ DONE | ToolRegistry / ToolParser (zero-dep JSON state machine) / ToolDef / render_chat (8 styles) / style_from_model — the self-contained tool+chat half of `rtk.h`. Built + 11 tests pass standalone; committed+pushed. |
| `include/llm/safetensors.{h,cpp}` | opus-sipir | ✅ DONE | **Phase-3 importer #1:** HF `safetensors` + `config.json` as a `WeightSource`. Maps HF tensor names → GGUF names and HF config → the `<arch>.*` meta keys, so an unconverted HF checkpoint flows through the SAME importer→IR→loader→executor stack. Zero-dep JSON parser. Committed+pushed. |
| `tests/test_safetensors.cpp` | opus-sipir | ✅ DONE | Synthesizes a real safetensors + config; tests HF→GGUF mapping, config→meta, read_raw round-trip, and import_model → Sip IR. 5 tests pass standalone. |
| `src/rtk.cpp` | **OPEN** | ⬜ QUEUE | **RTK orchestrator class + vision glue ONLY.** ⚠️ Tool/chat symbols (ToolRegistry, ToolParser, ToolDef::schema_text, ToolCall::get/has, render_chat, style_from_model) are ALREADY defined in `src/rtk_tools.cpp` — do **NOT** redefine them here or `make` fails with a duplicate-symbol (ODR) link error for every binary. |
| `src/vision_encoder.cpp` | **OPEN** | ⬜ QUEUE | ViT fwd pass, pure C++17 |
| `src/multimodal_projector.cpp` | **OPEN** | ⬜ QUEUE | 2-layer GELU MLP |
| `tests/test_tool_calling.cpp` | opus-sipir | ✅ DONE | 11 tests: registry, incremental JSON parser (marker + raw modes, escapes, nested objects), 8 chat templates, tool injection. Passes standalone. |
| `tests/test_rtk_chat.cpp` | **OPEN** | ⬜ QUEUE | After rtk.cpp done |
| `server/server.cpp` | **OPEN** | ⬜ QUEUE | Refactor to use Scheduler |
| `scripts/bench_matrix.sh` | **OPEN** | ⬜ QUEUE | Full benchmark automation |

---

## Architecture decisions — locked, do not deviate

| Decision | Constraint |
|:---------|:-----------|
| Tool call JSON parser | Hand-written state machine. No nlohmann/json. No regex. |
| Tool schemas | C++ `ToolDef` structs — not JSON Schema strings. |
| Vision input type | `ImageTensor` (float32 RGB). Caller does PNG/JPEG decode. |
| Vision encoder weight names | `v.blk.N.*` (LLaVA/CLIP convention) |
| Projector weight names | `mm.0.weight`, `mm.0.bias`, `mm.2.weight`, `mm.2.bias` |
| Kosh scope | Multi-turn lifecycle: spec decode, semantic cache, context compression |
| RTK scope | Single-turn: template render, tool dispatch, image inject, prefix cache |
| KV cache default | `FP32` unchanged. `INT8` is explicit opt-in. |
| Scheduler now | Single-model FIFO. Multi-model = Phase 5. |
| Sip IR in-memory vs on-disk | Decoupled. `SipModel` ≠ `SipIRHeader`. Never cross-depend. |

---

## Status log

```
2026-08-07 23:44  coordinator    Launched kernel-agent, sip-ir-agent, mem-agent (parallel)
2026-08-07 23:45  coordinator    ✅ include/llm/kosh.h written
2026-08-07 23:48  coordinator    ✅ include/llm/rtk.h written (tool calling + vision + chat)
2026-08-07 23:50  sip-ir-agent   ✅ sip_ir_writer/reader, gguf_to_sipir, test, spec, runtime patch
2026-08-07 23:50  mem-agent      ✅ mem_manager, scheduler, kv_cache INT8, test_kv_int8
2026-08-07 23:51  coordinator    ✅ AGENTS.md fully updated (this entry)
2026-08-07 23:xx  kernel-agent   ✅ DONE: Q4_K/Q5_K/Q6_K NEON + AVX2
2026-09-04 11:06  issue-audit    ✅ Audited all open GitHub issues against implementation; closed #7, #50-#60 with full technical documentation
```

---

## What ankit is working on

_Update this before you start coding — prevents duplication._

```
[ ] File(s): _______________
[ ] Task: _______________
[ ] ETA: _______________
```
| `issue-43` | Sonnet | `include/llm/remote_weight_source.h`, `src/remote_weight_source.cpp`, `tools/remote_server.cpp`, `tools/remote_demo.cpp` | ✅ DONE Issue 43: Remote layer streaming with local compressed cache |
