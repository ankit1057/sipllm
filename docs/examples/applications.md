# Applications

<div class="hero"><p class="lead"><strong>Is SipLLM the right engine for your application?</strong> This catalog lists the kinds of software SipLLM is designed to drive, grouped by domain. Each entry says why the design fits, which real flags and APIs it would use, what it costs you, and whether the fit has been <em>measured</em> or is still a <em>design argument</em>. To start building, go to the <a href="recipes.html">Recipes</a>.</p></div>

## How to read this page

SipLLM's one distinguishing property is a **hard, configurable ceiling on peak RAM**: it streams a GGUF model's transformer layers off storage and keeps only what fits under `--ram-budget` resident. Everything else follows from that: CPU-first execution, zero runtime dependencies (C++17 and pthreads), a stable C ABI and a Flutter/Dart plugin.

Every application below carries one of two status badges:

<p><span class="badge on">Demonstrated</span> a measured result committed to this repository backs the claim. The entry names the source.</p>

<p><span class="badge exp">Candidate</span> the application fits the design, but nobody has built or measured it yet. Most entries are Candidates. That is the honest state of the project.</p>

> [!MEASURED] The numbers this page leans on
> Apple M3 unless stated, CPU-only, warm cache, peak RSS from `/usr/bin/time -l`. Authoritative source: the North Star scorecard at the top of `CHANGELOG.md`, plus [Benchmarks](benchmarks.html).
>
> - **Bigger than RAM:** Llama-3.1-8B (4.92 GB Q4) in 204 MB peak RSS, and Llama-2-13B (7.87 GB Q4) in 317 MB, on a 16 GB Mac with about 3 GB free. Decode was 0.55 and 0.35 tok/s at that extreme (`bench/results/bigger-than-ram-*-2026-07-27.json`).
> - **Half the RAM when resident:** TinyLlama Q4_K_M fully resident with `--fast` used 667 MB, against 1317 MB for llama.cpp CPU, 1.97&times; smaller.
> - **Resident decode is slower than llama.cpp:** 73% of it on the M3 (68.0 vs 92.8 tok/s; SipLLM with `--fast` and `SIPLLM_FAST_LAYOUT=il4` at 8 threads, llama.cpp at 4). On a Dimensity 900 phone it was 81% (14.7 vs 18.2 tok/s; SipLLM at 8 threads, llama.cpp at its best of 2). Prefill is 71% (M3) and 66% (phone).
> - **The low-RAM mode costs speed:** the TinyLlama Q4_K_M streaming floor is 122 MB on the M3 and 118 MB on the phone (not re-measured in wave 16). Streaming decode on the M3 was 11.7 tok/s, against 59.6 tok/s fully resident with `--fast`, both at 4 threads (CHANGELOG, wave 12).

## At a glance

| # | Application | Status | Main SipLLM features |
|:--|:------------|:-------|:---------------------|
| 1.1 | Offline chat assistant in a Flutter app | Candidate | `SipllmRuntime`, `SipllmParams.ramBudgetBytes`, `DownloadManager` |
| 1.2 | Command-line inference on an Android phone | **Demonstrated** | `--ram-budget`, `--fast`, `--schedule` |
| 1.3 | On-device semantic search over local notes | Candidate | `sipllm_embed`, `EmbeddingStore` |
| 2.1 | Field-work assistant on entry-level phones | Candidate | `--ram-budget`, resumable downloader, offline inference |
| 2.2 | Offline summarisation and lookup over local documents | Candidate | `--spec`, `--reuse` |
| 3.1 | Smartwatch assistant | Candidate | `WearTransfer`, `ramBudgetBytes` |
| 3.2 | Apps that must not be killed for using too much memory | Candidate | `--ram-budget` contract, memory planner |
| 4.1 | ARM single-board computer appliance | Candidate | Linux aarch64 bundle, runtime SDOT dispatch |
| 4.2 | Kiosk or information terminal | Candidate | `--save-session`, `--load-session`, C ABI |
| 4.3 | Natural-language interface on a robot | Candidate | C ABI, `sipllm_cancel`, `--threads` |
| 5.1 | Running a model larger than free RAM on a laptop | **Demonstrated** | `--stream-lm-head`, `--no-async`, `--ram-budget` |
| 5.2 | Assistant inside a memory-hungry desktop app | Candidate | `--ram-budget`, `--fast`, C ABI |
| 6.1 | Local processing of sensitive text | Candidate | offline inference, C ABI |
| 6.2 | Air-gapped or offline-by-policy machines | Candidate | zero dependencies, local GGUF paths |
| 7.1 | Splitting one model's layers across machines on a LAN | **Demonstrated** | `--worker`, `--layers`, `--listen`, `--nodes` |
| 7.2 | Hosting a model too large for any single box | Candidate | LAN pooling, per-node `--ram-budget` |
| 8.1 | Studying the RAM-versus-speed trade-off | **Demonstrated** | `scripts/bench_ram_budget.sh`, engine stats |
| 8.2 | Teaching how transformer inference works | Candidate | readable C++17, `inspect_gguf`, per-layer profile |
| 9.1 | Local OpenAI-compatible endpoint for development | Candidate | `llm_server`, `/v1/chat/completions` |
| 9.2 | LLM-dependent tests in CI under a memory cap | Candidate | small models, `--greedy`, `--seed`, `--ram-budget` |
| 9.3 | Coding helpers that resend a long context every turn | Candidate | `--reuse`, `--save-session`, `--spec` |
| 10.1 | Many small model processes on one host | Candidate | per-process `--ram-budget` |
| 10.2 | Batch jobs on small-memory cloud instances | Candidate | streaming, `--ram-budget` |

## 1. Phones and tablets

### 1.1 Offline chat assistant in a Flutter app

<p><span class="badge exp">Candidate</span> The engine itself has run on a phone (see 1.2). The Flutter plugin and APK path is <strong>host-verified only</strong> (Apple M3). It has not been run on a phone or watch (<code>CHANGELOG.md</code>, Wave 8).</p>

- **What it is.** A chat app that runs a small instruction-tuned model on the phone, with no server and no data leaving the device.
- **Why SipLLM fits.** A phone's low-memory killer punishes apps that hold a large model resident. SipLLM lets the app pick an exact RAM ceiling and stream the rest. The Flutter plugin runs inference on a worker isolate, so the UI never blocks. Generation can be cancelled mid-token.
- **Features used.** `SipllmRuntime.open` with `SipllmParams(ramBudgetBytes: …, fastQuant: true)`, `generate()` as a token stream, `cancel()`, `lastStats`. A resumable, sha256-verified Hugging Face `DownloadManager` handles the one-time model download.
- **Trade-off.** On the measured phone (Dimensity 900), TinyLlama Q4_K_M fully resident decodes at about 14.7 tok/s (8 threads). A 135-token prompt took 2.87 s to reach its first token (scorecard). At the streaming floor (118 MB), decode drops to 4.7 tok/s (4 threads, wave 12). The engine applies no chat template, so the app must format the conversation itself.

### 1.2 Command-line inference on an Android phone

<p><span class="badge on">Demonstrated</span> TinyLlama Q4_K_M on a Dimensity 900 phone (Android 13) over <code>adb shell</code>: 615 MB resident, or 118 MB streaming, against 1251 MB for llama.cpp on the same file (<a href="benchmarks.html">Benchmarks</a>, &ldquo;On a phone&rdquo;; <code>bench/results/device-*-2026-10-19.json</code>).</p>

- **What it is.** Running the engine binary directly on a phone, either through `adb shell` or in a terminal app such as Termux. Useful for scripting, for prototyping before writing an app, and for measuring a specific handset.
- **Why SipLLM fits.** The build is `make` plus a C++17 compiler, with no other dependencies, and it cross-compiles with the Android NDK. Since wave 13, one ARM binary checks for the SDOT instruction at run time and falls back to the exact path on cores without it.
- **Features used.** `--ram-budget`, `--fast`, `--threads`, and `--schedule fixed16`. On the measured 2+6-core phone, the `proportional2` schedule decoded at 5.7–5.8 tok/s, against 13.5–14.3 for `fixed16` (scorecard). `--no-autotune` keeps runs reproducible.
- **Trade-off.** Resident decode is about 0.8&times; llama.cpp, and prefill about 0.66&times;. The RAM saving is real, but so is the speed gap. The measured numbers come from one SoC; your handset will differ.
- **Not yet shown.** A Cortex-A53-class core (no SDOT) has not been tested.

### 1.3 On-device semantic search over local notes

<p><span class="badge exp">Candidate</span> The embedding call and the SQLite vector store pass host tests. No retrieval-quality or on-device measurement exists.</p>

- **What it is.** Index a user's notes, messages or documents on the device, then find related passages by meaning, without uploading anything.
- **Why SipLLM fits.** The same engine that generates text returns L2-normalised final-layer hidden states. The plugin ships a SQLite store with cosine top-k search, so one model file and one library serve both search and generation.
- **Features used.** `sipllm_embed()` (C) or `SipllmRuntime.embed()` (Dart) with `Pooling.last` or `Pooling.mean`. Then `EmbeddingStore.open(path, dim: …)`, `add()`, `search()`.
- **Trade-off.** `embed()` clears the KV cache, so keep a separate runtime for embeddings if a conversation is live. Each runtime has its own memory budget. Search is a full linear scan. Embeddings taken from a general chat model are not a dedicated embedding model, and their retrieval quality has not been measured.

## 2. Offline and field applications

### 2.1 Field-work assistant on entry-level phones

<p><span class="badge exp">Candidate</span> Fits the design. No field deployment has been built or measured.</p>

- **What it is.** An assistant for people who work where connectivity is poor or absent. It helps fill in forms, answers questions from a bundled reference, or drafts short reports, all offline.
- **Why SipLLM fits.** Entry-level phones have limited RAM, but storage is usually plentiful. A bounded budget lets the model coexist with the rest of the app, the camera and the OS. After the one-time download, inference needs no connection ([FAQ](faq.html)).
- **Features used.** The plugin's `DownloadManager` resumes across restarts and verifies sha256, which suits unreliable links. `SipllmParams.ramBudgetBytes`, `fastQuant`, `schedulePolicy: SchedulePolicy.fixed16`.
- **Trade-off.** Low-end phones are slower than the measured Dimensity 900, so expect lower tok/s. Streaming at the RAM floor is several times slower than resident. Battery and thermal cost are unmeasured, and energy per token is N/A.

### 2.2 Offline summarisation and lookup over local documents

<p><span class="badge exp">Candidate</span> The speed-up from speculative decoding on copy-heavy output is measured (below). A document-summarisation application is not.</p>

- **What it is.** Summarising, quoting or answering questions from documents already on the device: manuals, policy text, meeting notes.
- **Why SipLLM fits.** Output that copies spans of its input suits prompt-lookup speculative decoding. Each verify pass streams every layer once for several draft tokens. On the M3, in the streaming regime, a copy-heavy prompt went from 9.7 to 30.7 tok/s with `--spec 5` (3.2&times;). Free text gained nothing and lost nothing (CHANGELOG, wave 12).
- **Features used.** `--spec N` (CLI) or `SipllmParams.specDraft` / `sipllm_params.spec_draft`. Output is token-identical to plain decoding. Add `--reuse` to keep the KV cache of a shared prefix across turns.
- **Trade-off.** The gain depends on the hardware. On the phone, `--spec 5` gave +18–30% when streaming but −9% when resident, because the int8 kernels make the phone compute-bound. Long documents need a larger `--ctx`, and the KV cache stays in RAM (see [When SipLLM is the wrong choice](#when-sipllm-is-the-wrong-choice)).

## 3. Wearables and very-low-RAM devices

### 3.1 Smartwatch assistant

<p><span class="badge exp">Candidate</span> No watch measurement exists. The phone-to-watch transfer and on-watch inference have not been run on hardware (<a href="wear-support.html">Wear OS support</a>).</p>

- **What it is.** A short-answer assistant on a watch that works when the phone is out of range.
- **Why SipLLM fits.** Watches have the tightest RAM budgets of any target here, and the RAM ceiling is the point of the engine. The plugin includes phone-to-watch model transfer over the Wearable Data Layer, so the watch does not download the model itself.
- **Features used.** `WearTransfer.sendModel()` and `listenIncoming()`; `SipllmRuntime.open` with `ramBudgetBytes`. `DeviceProfile.suggestedRamBudgetBytes` returns 220 MB for Wear OS. That value is a code default, not a measurement.
- **Trade-off.** Watch storage speed, thermals and battery cost are all unknown. Streaming at the floor will be slow. Treat this as unproven until someone measures it.

### 3.2 Apps that must not be killed for using too much memory

<p><span class="badge exp">Candidate</span> The memory planner's contract is implemented and unit-tested. No application has been built around it.</p>

- **What it is.** Any process that runs under an external memory limit: Android's low-memory killer, a cgroup or container memory limit, or an embedded system with no swap.
- **Why SipLLM fits.** `--ram-budget` is a contract, not a hint. Before loading, the [memory planner](memory-planner.html) prices the KV cache, scratch buffers and stream buffers. It then either guarantees `peak RSS ≤ budget` for the whole run or refuses with an itemised report: `RAM budget impossible. Use --ram-budget-force to run anyway.` Failing at start-up is easier to handle than an OOM kill mid-answer.
- **Features used.** `--ram-budget`, `--ctx` (the planner prices KV at the full context), `--ram-budget-force`, `sipllm_params.ram_budget_bytes`.
- **Trade-off.** A tight budget means more layers stream, and decode slows accordingly. The KV cache is priced at the full `--ctx` up front, so a long context can make a small budget infeasible even for short conversations.

## 4. Embedded Linux, SBCs, kiosks and robotics

### 4.1 ARM single-board computer appliance

<p><span class="badge exp">Candidate</span> Linux aarch64 bundles are built by CI. No single-board computer has been measured.</p>

- **What it is.** A self-contained text or voice-assistant box (home automation, lab equipment, a classroom device) built on a low-cost ARM board.
- **Why SipLLM fits.** There is nothing to install beyond the binary. CI publishes `linux/aarch64` and `linux/x86_64` bundles, and `install.sh` fetches the right one or builds from source. One baseline ARM binary picks the SDOT kernels at run time, or uses the exact path where SDOT is absent (`SIPLLM_CPU=generic` forces it off).
- **Features used.** `--ram-budget`, `--fast`, `--threads`, `--schedule`, the CLI or the C ABI.
- **Trade-off.** Boards that boot from SD cards or eMMC have far slower storage than a laptop SSD. In streaming mode, decode is bound by storage reads, and the FAQ notes that an off-cache model at the bounded extreme runs well under 1 tok/s. Use the largest budget the board can spare, and measure on the target.

### 4.2 Kiosk or information terminal

<p><span class="badge exp">Candidate</span> Session save and restore is tested on the host. No kiosk has been built.</p>

- **What it is.** A public-facing terminal that answers questions about one venue, product or service from a fixed system prompt.
- **Why SipLLM fits.** A kiosk restarts often and runs alongside a display stack. With session persistence, the long system prompt is processed once and saved. Each restart reloads it instead of re-running prefill.
- **Features used.** `--save-session F` and `--load-session F` (the C++ `Runtime::save_session` / `load_session`), `--reuse`, `--ram-budget`. Or embed the engine with the C ABI and call `sipllm_reset()` between visitors.
- **Trade-off.** A session file is tied to its model (it carries a model id and refuses a mismatch). It holds a resident KV cache, so its size grows with context. Prefill on CPU is slower than llama.cpp, so the restore matters.

### 4.3 Natural-language interface on a robot

<p><span class="badge exp">Candidate</span> Fits the design. Nothing has been built or measured on robot hardware.</p>

- **What it is.** Spoken or typed commands turned into structured intents on a robot or drone companion computer, with no network link.
- **Why SipLLM fits.** Perception and planning processes need most of the RAM and CPU. A fixed RAM ceiling and an explicit thread count let the language model take a known, bounded share. The C ABI's cancel call is safe from another thread, so a supervisor can stop generation at the next token boundary.
- **Features used.** `sipllm_open` with `ram_budget_bytes` and `threads`, `sipllm_generate` with a token callback (return 0 to stop), `sipllm_cancel`.
- **Trade-off.** Decode speed and time-to-first-token are not suitable for anything in a control loop. Use it for the conversational layer only. CPU contention with other processes will lower tok/s below anything measured here.

## 5. Desktop applications that must leave RAM for other work

### 5.1 Running a model larger than free RAM on a laptop

<p><span class="badge on">Demonstrated</span> Llama-3.1-8B (4.92 GB Q4_K_M) in 204 MB and Llama-2-13B (7.87 GB Q4_K_M) in 317 MB peak RSS, on a 16 GB Mac with about 3 GB free, with coherent output (<code>CHANGELOG.md</code> scorecard, &ldquo;Largest Runnable Model&rdquo;; <a href="benchmarks.html">Benchmarks</a>).</p>

- **What it is.** Trying a larger model on a machine that cannot hold it, or would have to evict everything else to do so.
- **Why SipLLM fits.** Peak RSS follows the width of one layer, not the depth or total size of the model. The measured runs streamed every layer, plus the output head, from disk.
- **Features used.** `--stream-lm-head --no-async --ctx 512`, exactly as measured. Use `--ram-budget` to trade some of the freed RAM back for speed.
- **Trade-off.** This is the slowest mode SipLLM has. The 8B model decoded at 0.55 tok/s and the 13B at 0.35 tok/s, with the model far larger than the free page cache (`bench/results/bigger-than-ram-*-2026-07-27.json`). It is for running a model that would not run at all, not for interactive use.

### 5.2 Assistant inside a memory-hungry desktop app

<p><span class="badge exp">Candidate</span> The resident RAM saving is measured (1.97&times; smaller than llama.cpp for TinyLlama Q4_K_M). No desktop integration has been built.</p>

- **What it is.** An editor, IDE, design tool or note-taking app that adds a local assistant without competing with its own working set, a browser and a build.
- **Why SipLLM fits.** You give the model a fixed slice of RAM, so the app's memory use stays predictable. Linking the C ABI avoids a separate service process.
- **Features used.** `sipllm_params.ram_budget_bytes`, `fast_quant`, `threads`; or the CLI with `--ram-budget` and `--fast`.
- **Trade-off.** Fully resident with `--fast`, decode is about 0.73&times; llama.cpp on the M3. A budget below the model size makes it slower still. The x86-64 AVX2 `--fast` path has not been measured on an x86 host (README status).

## 6. Privacy-bound local processing

### 6.1 Local processing of sensitive text

<p><span class="badge exp">Candidate</span> Inference is offline by design (<a href="faq.html">FAQ</a>). No application in this category has been built.</p>

- **What it is.** Redacting, classifying or summarising text that must not leave the machine or the device: personal records, internal documents, private messages.
- **Why SipLLM fits.** Inference makes no network calls. The only network use is downloading a model, which you can do once, or skip by passing a local path. A bounded footprint means this can run on ordinary staff laptops or phones rather than a dedicated server.
- **Features used.** The CLI with a local `.gguf` path, or the C ABI. Both open no listening socket outside LAN worker mode.
- **Trade-off.** Prefer the CLI or the C ABI. The HTTP server binds 127.0.0.1 by default but has no authentication, so any local process can use it (see [Recipes](recipes.html)). Do not use LAN pooling on an untrusted network either: its traffic is authenticated but not encrypted, and activations can reveal prompt content. Model accuracy on your task is your responsibility to evaluate.

### 6.2 Air-gapped or offline-by-policy machines

<p><span class="badge exp">Candidate</span> Fits the design. No air-gapped deployment has been reported.</p>

- **What it is.** Workstations with no internet access by policy: secure labs, industrial control rooms, exam halls.
- **Why SipLLM fits.** Building from source needs only `make` and a C++17 compiler. There are no Python wheels, no BLAS and no GPU drivers to bring across the air gap. A GGUF file copied in on removable media runs by path.
- **Features used.** `make`, `./build/llm <path>.gguf`, `--ram-budget`.
- **Trade-off.** The same speed trade-offs as any CPU deployment apply. You must vet the model file and the source yourself before bringing them in.

## 7. Pooling small machines on a LAN

### 7.1 Splitting one model's layers across machines on a LAN

<p><span class="badge on">Demonstrated</span> TinyLlama Q4_K_M split between an Apple M3 coordinator and an Android phone worker over Wi-Fi, 400M budget on each. Output was identical to a single machine on the default path. The worker's peak RSS was 301 MB (<code>bench/results/issue70-lan-2026-10-19.json</code>).</p>

- **What it is.** One model, its layers divided across two or more machines on the same network. Each machine holds only its own slice of RAM.
- **Why SipLLM fits.** Only the residual stream crosses the network, about 16 KB per token in the measured run. Each node keeps its own `--ram-budget`, pins only its own layers and holds KV only for them.
- **Features used.** On a worker: `--worker --layers A:B --listen IP:PORT`. On the coordinator: `--nodes host:port:A-B`. Both read a pre-shared token from `SIPLLM_LAN_TOKEN` or `SIPLLM_LAN_TOKEN_FILE`.
- **Trade-off.** Network latency dominates. Over Wi-Fi, decode was 2.6–2.8 tok/s, against 15.2–15.3 on one machine at the same 400M budget. Over loopback on one machine, the split reached 22.4–22.8 tok/s, about the fully resident speed, because each half fitted its budget. With `--fast` on both nodes, the M3 and the phone diverged at a near-tie, because their int8 kernels round differently. Traffic is not encrypted.

### 7.2 Hosting a model too large for any single box

<p><span class="badge exp">Candidate</span> The split mechanism is demonstrated (7.1), but only on a model that also fits on one machine.</p>

- **What it is.** Pooling several small machines (old laptops, spare phones, single-board computers) so that together they keep a larger model resident.
- **Why SipLLM fits.** The same layer split applies to any number of contiguous worker ranges. Each node donates a bounded slice of RAM.
- **Features used.** `--nodes h1:p1:A-B,h2:p2:B-C,…`, per-node `--ram-budget`, `--ctx` (identical on every node).
- **Trade-off.** Every extra hop adds a network round trip per token, and prefill and decode both pay it. On Wi-Fi, a single machine streaming from a fast SSD may well be faster. Measure both.

## 8. Education and research

### 8.1 Studying the RAM-versus-speed trade-off

<p><span class="badge on">Demonstrated</span> The project's own budget sweeps are committed as JSON and summarised in <code>CHANGELOG.md</code> (wave 12 re-sweep table: TinyLlama Q4_K_M from 128M to 768M, decode against pinned layers, peak RSS under every budget).</p>

- **What it is.** Measuring how much RAM buys how much speed on a given device and storage medium. This is the question behind the &ldquo;storage is cheaper than RAM&rdquo; argument.
- **Why SipLLM fits.** The budget is one dial, and the engine reports what it did: pinned layers, bytes streamed from disk, prefetch hits and misses, TTFT, prefill and decode rate. The harness takes peak RSS from the OS and refuses to run on a busy machine.
- **Features used.** `scripts/bench_ram_budget.sh` (with `BUDGETS`, `N`, `CTX`, `THREADS`), `scripts/bench.sh`, `--ram-budget`, `--no-autotune`, `--schedule`, `/usr/bin/time`.
- **Trade-off.** Published numbers are warm-cache. Cold-cache behaviour depends on the storage device and needs a cache purge (`PURGE=1` in `scripts/bench.sh` on macOS). Energy per token is not measured.

### 8.2 Teaching how transformer inference works

<p><span class="badge exp">Candidate</span> Fits the design. No course or teaching material has been built on it.</p>

- **What it is.** A codebase students can read end to end: a GGUF parser, tokenizer, quantised matmuls, attention, KV cache and sampler, in plain C++17 with no framework underneath.
- **Why SipLLM fits.** There are no dependencies to hide behind, and the exact fp32 path remains the correctness reference beside the fast int8 kernels. The web server's `/api/generate` stream emits per-layer I/O, dequantisation and compute timings for every token.
- **Features used.** `build/inspect_gguf` (metadata and tensor table), `build/dump_logits`, `/api/generate?profile=1` (localhost only; see the server caution in [Recipes](recipes.html)).
- **Trade-off.** It is an inference engine, not a training framework. Several architecture families run only on synthetic test weights so far; the release notes say which ones are checked against llama.cpp.

## 9. Developer tooling and CI

### 9.1 Local OpenAI-compatible endpoint for development

<p><span class="badge exp">Candidate</span> The endpoint exists and parses OpenAI chat requests. It binds 127.0.0.1 by default, has no authentication, and has not been measured as a development backend.</p>

- **What it is.** Pointing an existing tool that speaks the OpenAI chat API at a model running on the same machine.
- **Why SipLLM fits.** `POST /v1/chat/completions` accepts `messages`, `max_tokens`, `temperature` and `tools`, and returns tool calls in OpenAI format.
- **Features used.** `make server`; `llm_server <model> --port N --ctx N --threads N`.
- **Trade-off.** The server serves one model and one generation at a time. It accepts only `--port`, `--host`, `--allow-agent`, `--residency`, `--mmap`, `--threads` and `--ctx`, and silently ignores any other flag, so `--ram-budget` and `--fast` do not apply and it always runs the default streaming configuration. It does not stream responses. Agent requests (`"agent": true`, which run shell and file tools) are refused unless the server is started with `--allow-agent`.

### 9.2 LLM-dependent tests in CI under a memory cap

<p><span class="badge exp">Candidate</span> Fits the design. No external CI pipeline using it has been reported.</p>

- **What it is.** Running integration tests that call a real (tiny) model on standard CI runners, with no GPU and modest RAM.
- **Why SipLLM fits.** It builds from source in one `make`, the Linux bundles come from CI already, and small models stream in very little memory: SmolLM2-135M Q8_0 measured 54 MB streaming ([Benchmarks](benchmarks.html)). Greedy decoding with a fixed seed makes outputs repeatable on one machine.
- **Features used.** `--greedy`, `--seed`, `--ram-budget`, `-n`, the `sipllm pull` model cache (`~/.sipllm/models`).
- **Trade-off.** Outputs are not bit-identical across different CPUs. The measured M3 and phone builds agreed on argmax, with logits differing by about 1e-5. With `--fast`, greedy output diverged at near-ties. Assert on loose properties, not exact strings, if tests run on mixed hardware.

### 9.3 Coding helpers that resend a long context every turn

<p><span class="badge exp">Candidate</span> Context reuse and cross-process session restore are tested on the host. The <code>--spec</code> speed-up on copy-heavy output is measured (see 2.2). No end-to-end coding tool has been measured.</p>

- **What it is.** A local assistant that resends a system prompt and repository context on every turn and mostly edits or quotes code.
- **Why SipLLM fits.** `--reuse` keeps the KV of the longest common prefix and reprocesses only the changed tail. `--save-session` and `--load-session` carry that context across process restarts. Copy-heavy output, such as code edits, benefits from `--spec`.
- **Features used.** `--reuse`, `--save-session F`, `--load-session F`, `--spec N`, `--chat` (interactive mode turns on reuse automatically).
- **Trade-off.** A long repository context means a long `--ctx`, and the KV cache for it stays in RAM, priced in full by the planner. Prefill on CPU is slower than llama.cpp, so the first turn is slow.

## 10. Server-side density

### 10.1 Many small model processes on one host

<p><span class="badge exp">Candidate</span> Fits the design. No multi-tenant deployment has been measured.</p>

- **What it is.** Many isolated, low-traffic model instances (per user, per tenant, per task) on one CPU host, each with a hard memory limit.
- **Why SipLLM fits.** Each process gets its own `--ram-budget`, so the host's RAM can be divided predictably. Streamed reads of the same GGUF file go through the OS page cache, which processes share.
- **Features used.** The CLI or the C ABI per process, with `--ram-budget` and `--threads`.
- **Trade-off.** The shipped HTTP server is not a multi-tenant server: one generation at a time and no authentication. Density means many separate processes behind your own access control. CPU, not RAM, becomes the limit, and per-process decode is slower than llama.cpp. The page cache is shared and reclaimable, but it is still memory. Warm-cache numbers assume it holds the model.

### 10.2 Batch jobs on small-memory cloud instances

<p><span class="badge exp">Candidate</span> Fits the design. No cloud measurement exists.</p>

- **What it is.** Offline, throughput-insensitive jobs (tagging, extraction, summarisation of a backlog) on cheap instances that have little RAM but fast local disk.
- **Why SipLLM fits.** This is the economic bet behind the project: storage bandwidth is cheap, RAM is not. A bounded budget lets a model larger than the instance's RAM run at all.
- **Features used.** `--ram-budget`, `--spec` for copy-heavy outputs, `--greedy`.
- **Trade-off.** Network-attached block storage can be far slower than local NVMe, and streaming decode is bound by it. Measure before sizing. For anything latency-sensitive, a resident runtime on a larger instance is likely faster.

## When SipLLM is the wrong choice

Choose something else if any of these describe you:

- **You have a usable GPU and want to use it.** SipLLM is CPU-only today. The Vulkan backend detects devices but always falls back to CPU ([FAQ](faq.html)).
- **You want the most tokens per second and have RAM to spare.** Where the model fits in memory, llama.cpp is faster. SipLLM's resident decode is about 0.73&times; (M3) to 0.81&times; (phone) of llama.cpp's, and its prefill about 0.66&ndash;0.71&times;. SipLLM wins on RAM, not on speed.
- **You need to serve many concurrent users.** There is no batching across requests. The server runs one generation at a time and is not hardened for network exposure. A dedicated GPU serving stack is the right tool.
- **Your workload is dominated by very long contexts.** Layer streaming bounds weight memory, not the KV cache. KV stays resident and grows with `--ctx`; under a budget, it is priced at the full context up front.
- **You need first-token latency on long prompts.** Prefill trails llama.cpp on every measured device.
- **You need native Windows.** The targets are Linux, macOS, Android and embedded Linux.
- **You need a specific model family to be correct today.** This page makes no claim about which model families produce correct output. The release notes and the README's model-architecture table say which families are checked against llama.cpp; validate your exact GGUF file (for example, greedy output on a few known prompts against a reference engine) before you depend on it.

> [!NOTE] Something missing?
> If your application is not listed and fits the bounded-RAM idea, the [Recipes](recipes.html) cover the building blocks. Promoting a Candidate to Demonstrated takes a committed, reproducible measurement, the same bar as everything in the [Benchmarks](benchmarks.html).
