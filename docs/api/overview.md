# API reference

SipLLM exposes three programming surfaces, layered strictly from bottom to top. Each is a thin wrapper over the one below it — the C++ engine does the math, a stable C ABI flattens it into FFI-safe POD, and the Dart/Flutter package drives that ABI from a background isolate.

> [!NOTE] Reference pages
> These are reference pages: signatures are reproduced verbatim from source (`include/llm/*.h`, `bindings/flutter/sipllm_flutter/ffi/sipllm_ffi.h`, `lib/src/**`), grouped by type, one line of semantics each. For the *why* behind each subsystem, follow the links into the [architecture book](architecture.html).

## The three surfaces

| Surface | Header / entry point | Language | Audience |
|:--|:--|:--|:--|
| [C++ engine](api-cpp.html) | `include/llm/*.h` (namespace `llm`) | C++17 | Embedders linking the engine directly; the CLI (`build/llm`) and web server build on it. |
| [Stable C ABI](api-c.html) | `bindings/flutter/sipllm_flutter/ffi/sipllm_ffi.h` | C | Any language with a C FFI. Opaque handles + POD structs + function-pointer callbacks. |
| [Dart / Flutter](api-dart.html) | `package:sipllm_flutter` | Dart | Flutter apps. Isolate-backed runtime plus download / embedding / device / Wear helpers. |

## How they relate

The C++ engine (`llm::Runtime` and friends) has a rich C++17 interface — `std::string`, `std::function`, `std::unique_ptr` — none of which is FFI-safe. The C ABI (`sipllm_ffi.h`) is the *only* surface non-C++ callers touch: it is intentionally **additive** over the engine, adding no math and changing no defaults (`sipllm_params` zero-initialized reproduces the CLI's behavior). Dart's `SipllmRuntime` in turn wraps that ABI, running `sipllm_generate` on a worker isolate so the UI thread never blocks, and calling the thread-safe `sipllm_cancel` from the main isolate.

<div class="diagram"><svg viewBox="0 0 640 150" role="img" aria-label="Dart wraps the C ABI which wraps the C++ engine"><g font-family="ui-monospace,monospace" font-size="13"><rect x="20" y="20" width="180" height="110" rx="8" fill="#eef2ff" stroke="#c7d2fe"/><text x="110" y="44" text-anchor="middle" fill="#3730a3">Dart / Flutter</text><text x="110" y="66" text-anchor="middle" fill="#6366f1" font-size="11">SipllmRuntime (isolate)</text><text x="110" y="84" text-anchor="middle" fill="#6366f1" font-size="11">Download · Embedding</text><text x="110" y="102" text-anchor="middle" fill="#6366f1" font-size="11">Device · Wear</text><rect x="230" y="20" width="170" height="110" rx="8" fill="#ecfeff" stroke="#a5f3fc"/><text x="315" y="44" text-anchor="middle" fill="#155e75">C ABI</text><text x="315" y="66" text-anchor="middle" fill="#0891b2" font-size="11">sipllm_ffi.h</text><text x="315" y="84" text-anchor="middle" fill="#0891b2" font-size="11">opaque ctx + POD</text><text x="315" y="102" text-anchor="middle" fill="#0891b2" font-size="11">18 SIPLLM_API fns</text><rect x="430" y="20" width="190" height="110" rx="8" fill="#f0fdf4" stroke="#bbf7d0"/><text x="525" y="44" text-anchor="middle" fill="#166534">C++ engine</text><text x="525" y="66" text-anchor="middle" fill="#16a34a" font-size="11">Runtime · LayerLoader</text><text x="525" y="84" text-anchor="middle" fill="#16a34a" font-size="11">Transformer · KVCache</text><text x="525" y="102" text-anchor="middle" fill="#16a34a" font-size="11">ops · quant · tokenizer</text><path d="M200 75 L230 75" stroke="#94a3b8" stroke-width="2"/><path d="M400 75 L430 75" stroke="#94a3b8" stroke-width="2"/></g></svg></div>

## Pick a surface

<div class="card-grid"><div class="card"><h3>C ABI &rarr;</h3><p>Bind SipLLM from any language. Four POD structs, three enums, one callback typedef, and <code>18</code> <code>SIPLLM_API</code> functions. See <a href="api-c.html">the C ABI reference</a>.</p></div><div class="card"><h3>C++ engine &rarr;</h3><p>Link <code>namespace llm</code> directly for the full streaming stack: <code>open_model()</code>, <code>Runtime</code>, <code>LayerLoader</code>, <code>Transformer</code>, <code>ModelConfig</code>, <code>Tokenizer</code>, and the <code>ops</code>/<code>quant</code> kernels. See <a href="api-cpp.html">the C++ engine reference</a>.</p></div><div class="card"><h3>Dart / Flutter &rarr;</h3><p>Add <code>package:sipllm_flutter</code>: <code>SipllmRuntime</code>, plus a resumable Hugging Face downloader, a SQLite embedding store, device/arch detection, and phone&harr;Wear transfer. See <a href="api-dart.html">the Dart reference</a>.</p></div></div>

> [!WARNING] Capability caveats that cut across all three surfaces
> - **GPU offload is detection-only.** `sipllm_vulkan_available()` / `SipllmDevice.vulkanAvailable` can report a device, but `vulkan_matmul` always falls back to CPU — never assume working GPU acceleration.
> - **The engine applies no chat template.** All three surfaces run inference over raw text (BOS added only on a fresh sequence). Conversation formatting is the caller's job (the Flutter app does it in `PromptTemplate`).
> - **`embed()` clears KV state** on every surface — use a dedicated context/runtime for embeddings, not one mid-conversation.
> - **`Q8_1` and `Q8_K` are not dequantizable** (they throw); the `--fast` int8 SDOT kernel is Q8_0-only and ARM-only.

## See also

- [Architecture book](architecture.html) — the module-by-module design behind these APIs.
- [Flutter runtime](flutter-runtime.html) — how the Dart isolate wrapper is built.
- [Quantization](quantization.html) · [Tokenizer](tokenizer.html) · [Thread pool](thread-pool.html) — subsystem deep-dives.
