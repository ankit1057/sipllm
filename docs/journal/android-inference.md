# Android inference

**SipLLM was built to run on a phone: standard C++17 and `pthread`, no BLAS, no ggml, no PyTorch — the runtime that streams a 13 B model in 317 MB on a Mac is the same object code that a Dimensity 8300 compiles under Termux. Wave 8 wraps that engine in a stable C ABI, a per-ABI NDK build, and a Flutter app whose Dart isolate streams tokens to the UI, plus a phone→watch transfer path for a Wear OS budget. But honesty first: Wave 8 is *host-verified only*. The Android APK, on-device inference, and the Wear transfer have not yet been run on real hardware, and there are no device benchmark numbers — none are invented on this page.**

> [!WARNING] Not yet device-verified
> Everything below is verified on the host (macOS / Apple M3): the C ABI, the full Dart → isolate → FFI path, the downloader, and the embedding store all pass. **The on-device APK, phone inference (POCO X6 Pro), and Wear transfer (OnePlus Watch 2) are NOT yet runtime-verified, and device benchmarks are pending.** Any tok/s or peak-RSS figure you see on this site is Apple M3 host data — no phone or watch numbers exist yet.

## The per-ABI NDK build

The engine has no Android-specific fork. `bindings/flutter/sipllm_flutter/src/CMakeLists.txt` globs the *shared* `src/*.cpp`, adds the C-ABI shim, and builds `libsipllm_ffi` once per ABI through the NDK — no runtime code is duplicated. The only per-ABI difference is SIMD tuning:

| ABI | Compile flags | Effect |
|:--|:--|:--|
| `arm64-v8a` | `-march=armv8.2-a+dotprod` | lights up the SDOT kernels in `simd.h` |
| `armeabi-v7a` | `-mfpu=neon-vfpv4` | NEON, no dot-product |
| `x86_64` | (baseline) | scalar fallback |

The `+dotprod` on arm64 is the important one: Armv8.2 dot-product is universal on the target SoCs (Dimensity 8300 A715/A510, Snapdragon W5 A53), and it enables the int8 SDOT path.

> [!CAUTION] SDOT accelerates Q8_0 only
> The `--fast` int8 SDOT kernel is **Q8_0-only** and ARM-only, and it is an approximation (activation quantized) — the fp32-dequant path stays the correctness oracle. It does *not* yet accelerate K-quants, so a `Q4_K_M` model on a phone still decodes through the scalar/NEON fp32 path. Do not expect `--fast` speedups on 4-bit models.

The same CMake can compile the experimental Vulkan backend, but per the runtime's standing caveat that backend is **detection-only**: `vulkan_matmul` always falls back to CPU. `make VULKAN=1` (or `-DSIPLLM_VULKAN=ON`) enables device *enumeration*, never working GPU offload.

## The Termux path

Android was the original target, and Termux is still the primary one. Because the engine is just C++17 + `pthread`, it builds and runs on-device with no cross-compilation ceremony — `make` on the phone produces the same `build/llm` CLI you get on a laptop. The README's own status notes SipLLM was "built and tested on a phone (Termux / Android, Dimensity 8300)"; that CLI path is the most mature Android story. The APK below is the productization layer on top of the same binary.

## The Flutter APK + Dart isolate

The app never blocks the UI thread on inference. `SipllmRuntime` runs the C++ engine on a **worker isolate**; tokens flow back to the UI isolate over a `Stream<SipllmToken>`, and generation is cancelled mid-flight by the UI calling `sipllm_cancel` — a thread-safe atomic the callback checks, so cancellation needs no polling and never stalls the UI. The generation knobs are plumbed through a zero-init `SipllmParams`:

```text
SipllmParams(
  ramBudgetBytes,      // 0 = unlimited streaming
  threads,             // 0 = hardware_concurrency
  maxCtx,
  fastQuant,           // Q8_0 SDOT, opt-in
  streamLmHead,
  schedulePolicy,      // proportional2 by default
)
```

`SipllmController` (the example app's single source of truth) owns model download/import, the loaded runtime, streaming chat, an optional embedding-backed memory, and benchmark capture. On load it also formats conversations with a `PromptTemplate` (chatml/llama3/zephyr/raw) — because the engine itself applies **no chat template**; inference is over raw text and the app is responsible for the prompt shape.

> [!NOTE] Dart stats are a subset
> The Dart `SipllmStats` drops the per-phase seconds (`load_s`/`prefill_s`/`decode_s`) that the C `sipllm_stats` carries; it keeps TTFT, prefill/decode tok/s, and the byte/count fields. The benchmark harness records device model, ABI, cores, threads, and budget — but no device runs have populated it yet.

## The watch's tight RAM budget

A watch has far less RAM than a phone, so the controller special-cases it: when `ArchDetector` reports `isWearOs`, the app defaults `ramBudgetMiB` to a tight streaming budget (`suggestedRamBudgetBytes`) instead of unlimited — exactly the [`--ram-budget`](memory-planner.html) dial, turned down hard. Because peak RSS tracks a single layer's width, a bounded budget is what makes a watch a plausible target at all.

Getting a model onto the watch is a **phone → Wear OS transfer** over the Wearable Data Layer `ChannelClient`: Bluetooth for control, Wi-Fi High-Bandwidth for the bulk bytes, with a resumable header/ack protocol. RFCOMM is deliberately avoided because Wear OS does not expose it. This is the part most in need of hardware validation — see the caveat below.

## Status: what is and is not proven

To be unambiguous, because it matters:

- **Verified on host (macOS / Apple M3):** C-ABI smoke test (generate + cancel + embed); the full Dart → isolate → FFI path; the resumable Hugging Face downloader (5 tests); the SQLite embedding store (13 tests); engine `make test` green; `flutter analyze` clean.
- **NOT yet runtime-verified:** the Android APK and on-device inference on the POCO X6 Pro; the Wear transfer on the paired OnePlus Watch 2. These are the remainder of milestone M6.5.
- **No device numbers exist.** Every tok/s, TTFT, and peak-RSS figure on this portal is Apple M3 host data. Phone and watch benchmarks are pending; none are fabricated.

Once the APK runs on device this page graduates from "host-verified" to measured. Until then it is an honest description of code that compiles, links, and passes on the host. For the runtime wrapper details see the [Flutter runtime](flutter-runtime.html) chapter, and for the watch transfer protocol see [Wear OS support](wear-support.html).
