# Flutter runtime & SDK

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** This is how the C++ streaming engine becomes a Dart SDK you can `await for` over. The engine's interface is C++ (`std::string`, `std::function`, `std::unique_ptr`) — none of it FFI-safe — so a stable C ABI (`bindings/flutter/sipllm_flutter/ffi/sipllm_ffi.h`) is the only surface Dart talks to. For the Dart-facing API see [api-dart](api-dart.html); for the ABI itself see [api-c](api-c.html); for phone→watch model transfer see [wear-support](wear-support.html).

## Problem

Inference is synchronous, blocking, and lives in C++. A phone UI cannot call it directly: (1) the engine's C++ types cross no FFI boundary; (2) `sipllm_generate` blocks its thread for the full prefill+decode, which would freeze the Flutter UI isolate; and (3) the user must be able to *cancel* a running generation and see tokens *stream* as they are produced, not wait for a monolithic result. The SDK's job is to wrap the engine so that a Dart consumer gets `Stream<SipllmToken>`, cancellation, embeddings, and per-run stats — with the native work off the UI thread.

## Design

Four layers stack from the metal up: a C ABI, a Dart binding + worker isolate, device/download/embedding services, and the example app that ties them together.

### 1. The C ABI — 18 functions, 4 POD structs, one opaque handle

`sipllm_ffi.h` exposes exactly **18 `SIPLLM_API` functions** over four plain-old-data structs (`sipllm_params`, `sipllm_sampler`, `sipllm_model_info`, `sipllm_stats`) and one opaque handle (`sipllm_ctx`). `sipllm_ffi.cpp` is a thin translation layer: it owns a `llm::Runtime`, marshals the PODs to/from the engine's C++ types, and adapts the `std::function` streaming callback to a C function pointer — no inference logic lives there.

The header pins a **threading contract**:

- A `sipllm_ctx` is **NOT thread-safe** for concurrent `generate()` calls — drive one context from one worker at a time.
- `sipllm_cancel()` **IS thread-safe**: it flips a `std::atomic<bool>` the decode loop checks at every token boundary, so it can be called from another thread while `generate()` runs.

`sipllm_params` zero-initialized reproduces the CLI's behavior; `sipllm_params_default()` fills the recommended edge defaults (streaming, quantized residency, hardware threads, `Proportional2`, async prefetch).

<div class="card-grid"><div class="card"><h3>Lifecycle</h3><p><code>sipllm_params_default</code>, <code>sipllm_sampler_default</code>, <code>sipllm_open</code>, <code>sipllm_close</code>.</p></div><div class="card"><h3>Generation</h3><p><code>sipllm_generate</code> (blocking, streams via <code>sipllm_token_cb</code>), <code>sipllm_cancel</code>, <code>sipllm_reset</code>.</p></div><div class="card"><h3>Introspection</h3><p><code>sipllm_get_model_info</code>, <code>sipllm_get_threads</code>, <code>sipllm_embed_dim</code>, <code>sipllm_embed</code>.</p></div><div class="card"><h3>Device / build</h3><p><code>sipllm_version</code>, <code>sipllm_vulkan_compiled</code>, <code>sipllm_vulkan_available</code>, <code>sipllm_vulkan_info</code>, <code>sipllm_hardware_concurrency</code>, <code>sipllm_optimal_threads</code>, <code>sipllm_set_log_level</code>.</p></div></div>

### 2. `SipllmRuntime` — one isolate, one context, a token Stream

`SipllmRuntime` (`lib/src/runtime/sipllm_runtime.dart`) owns **one worker isolate** holding **one native `sipllm_ctx`**. Because native inference blocks its thread, it runs in that isolate (`sipllm_isolate.dart`), never on the UI isolate. Tokens flow back over a `SendPort`: the token callback is a `NativeCallable<...>.isolateLocal` the engine invokes *synchronously* on the worker's thread during the blocking `generate()`, and each invocation is just a `port.send({'token': ..., 'id': ...})`. The UI-side `generate()` turns those messages into a `Stream<SipllmToken>`.

Cancellation is the subtle part. The worker isolate is *busy* inside `generate()` and cannot service a "cancel" message — so cancellation **does not route through the worker**. `SipllmRuntime.open()` loads its *own* bindings handle on the main isolate and keeps the shared context's integer address; `cancel()` calls the thread-safe `sipllm_cancel()` directly on that address (`Pointer<SipllmCtx>.fromAddress`) while the worker blocks. The stream's `onCancel` wires consumer-side unsubscription to that same cancel, so dropping the listener stops the engine.

```text
UI isolate                         worker isolate                 native
----------                         --------------                 ------
generate(prompt) --------port----> _generate() ---------------->  sipllm_generate()
   Stream<SipllmToken>  <--token--   isolateLocal callback  <----   (per token)
cancel()  ── sipllm_cancel(ctxAddr) directly on shared ctx ────>  atomic flag @ boundary
```

A runtime is **single-context**: `generate()` throws if one is already in flight (`_busy`), `complete()` collects the stream into a `String`, `embed()` returns an `L2`-normalized `Float32List`, `reset()` clears KV state, and `close()` cancels any in-flight run, frees the context, and kills the isolate.

### 3. Device, download, embedding services

- **`SipllmDevice`** (`lib/src/device/sipllm_device.dart`) — model-free engine/accelerator caps: `engineVersion`, `hardwareConcurrency`, `vulkanCompiled` / `vulkanAvailable` / `vulkanInfo`, and `optimalThreads(modelPath)` (benchmarks 1..hw threads and caches the fastest).
- **`ArchDetector` / `DeviceProfile`** (`lib/src/device/arch.dart`) — resolves the ABI and Wear OS flag from `Build.SUPPORTED_ABIS` + `PackageManager.FEATURE_WATCH` over a `MethodChannel`, falling back to `dart:io` inference on desktop/tests. `suggestedRamBudgetBytes` returns **~220 MB** on a watch (`isWearOs`) and **0** (unlimited streaming) on a phone, where the user raises the dial themselves. The GGUF is arch-independent; only `libsipllm_ffi.so` is per-ABI, and the OS resolves that at load time.
- **`DownloadManager` / `DownloadTask`** (`lib/src/download/`) — a resumable, multi-connection Hugging Face downloader on `dart:io` `HttpClient` alone (host-VM testable). The manager owns a shared client and a **global connection budget** (a permit semaphore): individual tasks may request more segments, but only `maxConnections` sockets ever stream across *all* tasks. `autoUncompress` is off so ranged bytes arrive identity-encoded. A task splits the resource into contiguous byte segments streamed into `<dest>.partN`, persists committed offsets to a **`<dest>.sipdl.json` sidecar** so a killed process resumes instead of restarting, then concatenates and verifies against `expectedSize` + lowercase-hex **`sha256`**. `HuggingFaceRepo.resolveUrl` builds the `resolve/<revision>/<file>` URL the CDN 302-redirects.
- **`EmbeddingStore`** (`lib/src/embedding/embedding_store.dart`) — a local SQLite vector store on `package:sqlite3`. Vectors are little-endian **float32 BLOBs** reconstructed in place via `Float32List.view`; a `store_meta` row pins the fixed `dim`. Because the engine emits **L2-normalized** vectors, cosine similarity is a plain **dot product**, and search is a full **linear scan** — adequate for the few-thousand-row scale kept on a phone or watch. ANN indexing (HNSW/IVF) is future work behind the same API.

### 4. The example app — Provider + a single controller

`example/lib/` is a reference companion app. State is a **Provider + one `SipllmController` god-object** (`ChangeNotifier`, `core/controller/sipllm_controller.dart`) that owns device probing, model management, the loaded runtime, streaming chat, an embedding-backed memory, phone→watch transfer, and benchmark capture; widgets are thin `Consumer`s. The phone shell is a **4-tab `IndexedStack`** (Home / Models / Chat / Bench) so a streaming chat keeps running while you peek at Benchmarks; `_Bootstrap` routes to the **Wear shell** instead when the probe reports a watch. Theme is dark **Material 3** (`core/theme/app_theme.dart`: `#0F1115` background, `#00E676` primary, Space Mono type). A **6-model catalog** (`core/models/model_catalog.dart`) lists ungated Llama-architecture GGUFs (SmolLM2 135M/360M/1.7B, Llama 3.2 1B, TinyLlama 1.1B, Llama 3.1 8B) with a `watchFriendly` flag.

Because **the engine applies no chat/prompt template** — inference is over raw text, BOS added only on a fresh sequence — the app formats conversations itself. `PromptTemplate` (`core/chat/prompt_format.dart`) renders **chatml** (SmolLM2/Qwen), **llama3**, **zephyr** (TinyLlama), or **raw** (base models), emitting special tokens as literal text that the byte-level BPE tokenizer encodes verbatim.

## Alternatives considered

| Approach | Why not |
|:---------|:--------|
| Call the engine synchronously from the UI isolate | Blocks the UI for the whole prefill+decode — no streaming, no responsiveness. The worker isolate exists precisely to move that off the UI thread. |
| `dart:ffi` bindings only, no isolate | Same blocking problem, and no clean place to host the synchronous streaming callback. |
| Route cancel through the worker isolate | The worker is blocked inside `generate()` and cannot service a message; the whole point of the thread-safe `sipllm_cancel()` atomic is to be called from *another* thread. |
| `package:http` for downloads | Pulls a Flutter/plugin dependency; `dart:io` `HttpClient` keeps the downloader host-VM testable and lets us force identity encoding for byte-exact ranges. |
| A vector DB dependency for embeddings | Overkill at a few-thousand rows; a normalized-dot-product linear scan in SQLite has zero extra deps and the API can grow an ANN index later. |

## Tradeoffs

- **One context per runtime.** Concurrency safety is bought by construction — a runtime refuses a second concurrent `generate()`. For parallel work, open more runtimes (more isolates, more contexts).
- **`embed()` clears KV state.** `sipllm_embed` prefills and pools the final hidden state, which resets the conversation. **Use a dedicated runtime for embeddings**, never one mid-conversation.
- **Dart `SipllmStats` is a subset** of the C `sipllm_stats`: it drops the per-phase seconds (`load_s`, `prefill_s`, `decode_s`) and keeps TTFT, prefill/decode tok/s, and the byte/count fields the dashboard shows.
- **Vulkan is detection-only.** `vulkanCompiled`/`vulkanAvailable` surface build/enumeration state, but the engine's `vulkan_matmul` always falls back to CPU — the SDK exposes no working GPU acceleration.
- **No on-device performance numbers exist.** Wave 8 (Flutter) is host-verified only; the measured figures in [benchmarks](benchmarks.html) are Apple M3, CPU-only. Phone/watch decode rates are **N/A** until measured on hardware.

## Source files

| File | Role |
|:-----|:-----|
| `ffi/sipllm_ffi.h` | the stable C ABI — 18 functions, 4 PODs, opaque `sipllm_ctx`, threading contract |
| `ffi/sipllm_ffi.cpp` | thin translation layer: owns `llm::Runtime`, marshals PODs, adapts the streaming callback, atomic cancel |
| `lib/src/runtime/sipllm_runtime.dart` | `SipllmRuntime`: isolate-backed streaming, main-isolate cancel on the shared ctx address |
| `lib/src/runtime/sipllm_isolate.dart` | worker isolate: `isolateLocal` token callback, `_generate`/`_embed`, `_readStats` (the subset) |
| `lib/src/runtime/sipllm_types.dart` | ergonomic Dart POD mirrors (`SipllmParams`/`Sampler`/`ModelInfo`/`Stats`/`Token`) |
| `lib/src/ffi/sipllm_bindings.dart`, `native_library.dart` | generated FFI struct/function bindings + `.so`/dylib loader |
| `lib/src/device/sipllm_device.dart`, `arch.dart` | engine caps; ABI + Wear OS profile and suggested RAM budget |
| `lib/src/download/hf_download_manager.dart`, `download_task.dart` | shared client + global connection budget; segmented `.sipdl` resume + sha256 verify |
| `lib/src/embedding/embedding_store.dart` | SQLite float32 vector store, linear-scan cosine |
| `example/lib/core/controller/sipllm_controller.dart` | the single `ChangeNotifier` state object |
| `example/lib/core/chat/prompt_format.dart` | `PromptTemplate` (chatml/llama3/zephyr/raw) — the app templates, the engine does not |
| `example/lib/core/models/model_catalog.dart`, `theme/app_theme.dart`, `main.dart` | 6-model catalog, dark Material 3 theme, 4-tab shell + Wear routing |
| `src/CMakeLists.txt` | builds `libsipllm_ffi` from engine sources + the shim; per-ABI SIMD tuning |

## Build

`src/CMakeLists.txt` globs the engine's `src/*.cpp` plus `sipllm_ffi.cpp` into one shared `libsipllm_ffi` (C++17, `-O3`, hidden visibility, no deps beyond pthreads). The Android Gradle build runs it **once per ABI via the NDK**: `arm64-v8a` compiles with **`-march=armv8.2-a+dotprod`** — universal on the target SoCs (Dimensity 8300, Snapdragon W5) and what lights up the engine's int8 SDOT kernels — while `armeabi-v7a` gets `-mfpu=neon-vfpv4` and keeps the scalar fallback. `-DSIPLLM_VULKAN=ON` compiles the experimental backend, but with no SPIR-V shader it stays device-enumeration-only and matmul remains on CPU.

## Future work

- **On-device measurement.** Every number here is host-verified; the phone/watch scorecard is unwritten (`sudo powermetrics`-class energy is deliberately never fabricated).
- **ANN embedding index.** Swap the linear scan for HNSW/IVF behind the unchanged `EmbeddingStore` API.
- **Multi-context orchestration.** A pool of runtimes for concurrent generations, and surfacing the dropped per-phase stats when a UI needs them.
- **Real GPU offload.** Contingent on the engine's Vulkan matmul dispatch shipping — see [quantization](quantization.html).
