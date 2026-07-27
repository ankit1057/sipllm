# C ABI reference

`bindings/flutter/sipllm_flutter/ffi/sipllm_ffi.h` is the stable C ABI over the C++ streaming engine — the *only* surface Dart (or any other language) talks to. The engine's headers use `std::string`, `std::function`, and `std::unique_ptr`, none of which is FFI-safe; this header flattens them into opaque handles, plain-old-data structs, and C function-pointer callbacks. It is intentionally **additive** over the engine: it adds no math and changes no defaults — a zero-initialized `sipllm_params` reproduces the CLI's behavior.

> [!NOTE] Reference
> Signatures below are copied verbatim from `sipllm_ffi.h`. Every exported function is annotated `SIPLLM_API` (`__attribute__((visibility("default")))`, or `__declspec(dllexport)` on Windows). The higher-level [Dart wrapper](api-dart.html) mirrors these one-to-one; the underlying math lives in the [C++ engine](api-cpp.html).

## Threading contract

The header states the contract explicitly:

- A `sipllm_ctx` is **NOT** thread-safe for concurrent `generate()` calls — drive one context from one worker isolate/thread at a time. A context is **thread-affine after open**.
- `sipllm_cancel()` **IS** safe to call from another thread while `generate()` runs; it flips an atomic the decode loop checks at every token boundary.
- `sipllm_embed()` **clears KV state** — use a dedicated context for embeddings if you also hold a live conversation.

## Enums

```c
/* ThreadPool schedule policy (octa-core work distribution). Mirrors
 * llm::ThreadPool::SchedulePolicy. Proportional2 is the engine's CLI default. */
enum {
  SIPLLM_SCHED_STATIC = 0,
  SIPLLM_SCHED_FIXED8 = 1,
  SIPLLM_SCHED_FIXED16 = 2,
  SIPLLM_SCHED_FIXED32 = 3,
  SIPLLM_SCHED_PROPORTIONAL2 = 4,
  SIPLLM_SCHED_PROPORTIONAL4 = 5,
  SIPLLM_SCHED_ADAPTIVE = 6
};

/* Embedding pooling strategy for sipllm_embed(). */
enum {
  SIPLLM_POOL_LAST = 0, /* last-token hidden state (correct for causal decoders) */
  SIPLLM_POOL_MEAN = 1  /* mean over per-token last-layer hidden states          */
};
```

The scheduler enum mirrors `llm::ThreadPool::SchedulePolicy` (see the [C++ engine](api-cpp.html#threadpool-thread-pool-h) reference). `SIPLLM_SCHED_PROPORTIONAL2` is the CLI default. Note the [auto-tuner](auto-tuning.html) only benchmarks four of these seven policies.

## Opaque handle

```c
typedef struct sipllm_ctx sipllm_ctx; /* opaque runtime handle */
```

Wraps a `llm::Runtime`. Created by `sipllm_open`, destroyed by `sipllm_close`.

## POD structs

### `sipllm_params` — open-time knobs

Zero-initialize then override; `sipllm_params_default()` fills the recommended edge defaults. All bool-ish fields are `0`/non-`0`.

```c
typedef struct {
  uint64_t ram_budget_bytes; /* 0 = unlimited streaming; else hard peak-RSS ceiling (--ram-budget) */
  int32_t threads;           /* >0 fixed; 0 = hardware_concurrency; -1 = auto-tune+cache profile   */
  int32_t max_ctx;           /* 0 = engine default (4096)                                          */
  int32_t n_buffers;         /* async prefetch ring buffers (>=1)                                  */
  int32_t use_mmap;          /* mmap backend instead of pread                                      */
  int32_t async_prefetch;    /* background double-buffered prefetch thread                         */
  int32_t fast_quant;        /* int8 SDOT kernel for Q8_0 (--fast); numerically equivalent         */
  int32_t stream_lm_head;    /* stream non-tied LM head off disk (RAM<->speed knob)                */
  int32_t residency_fp32;    /* FP32 residency (else Quantized — the memory-bounded default)       */
  int32_t force_budget;      /* honor ram_budget even below the safe floor (--ram-budget-force)    */
  int32_t schedule_policy;   /* SIPLLM_SCHED_*                                                     */
} sipllm_params;
```

### `sipllm_sampler` — sampling configuration

Mirrors `llm::SamplerConfig`. `temperature <= 0` selects greedy decoding.

```c
typedef struct {
  float temperature;
  int32_t top_k;         /* <=0 disables */
  float top_p;           /* 1.0 disables */
  float repeat_penalty;  /* 1.0 disables */
  int32_t repeat_last_n; /* history window */
  uint64_t seed;
} sipllm_sampler;
```

### `sipllm_model_info` — static model description

Filled by `sipllm_get_model_info()`.

```c
typedef struct {
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;
  int64_t dim;
  int64_t vocab_size;
  int64_t ctx_len;
  int32_t tokenizer_kind; /* 0=SentencePiece, 1=BPE, 2=byte */
  char arch[32];          /* e.g. "llama", "qwen2", "gemma2" */
} sipllm_model_info;
```

### `sipllm_stats` — per-generation metrics

Filled by `sipllm_generate()`. Mirrors `llm::GenStats` plus engine-owned peak RSS. (The [Dart `SipllmStats`](api-dart.html#sipllmstats) is a *subset* of this — it drops the three per-phase seconds fields.)

```c
typedef struct {
  double load_s;
  double ttft_s;
  double prefill_s;
  double decode_s;
  double prefill_tok_s;
  double decode_tok_s;
  uint64_t peak_rss_bytes;
  uint64_t weights_resident_bytes;
  uint64_t kv_bytes;
  uint64_t bytes_read;
  uint64_t prefetch_hits;
  uint64_t prefetch_misses;
  int32_t pinned_layers;
  int32_t n_layers;
  int32_t prompt_tokens;
  int32_t gen_tokens;
  int32_t ctx_used;
  int32_t ctx_max;
} sipllm_stats;
```

## Callback typedef

```c
/* Streaming token callback. `piece` is UTF-8, NUL-terminated, valid only for
 * the duration of the call. Return non-zero to continue, 0 to stop early. */
typedef int32_t (*sipllm_token_cb)(const char* piece, int64_t token_id, void* user);
```

## Functions

All 18 exported functions, grouped as the header groups them.

### Lifecycle

```c
SIPLLM_API void sipllm_params_default(sipllm_params* out);
```
Fill `out` with recommended edge defaults (streaming, quantized residency, hw threads, Proportional2, async prefetch, repeat_penalty 1.1).

```c
SIPLLM_API void sipllm_sampler_default(sipllm_sampler* out);
```
Fill `out` with default sampler configuration.

```c
SIPLLM_API sipllm_ctx* sipllm_open(const char* model_path, const sipllm_params* p,
                                   char* err_buf, int32_t err_cap);
```
Open a GGUF/.llmw model. On failure returns `NULL` and writes a message into `err_buf` (NUL-terminated, truncated to `err_cap`). Thread-affine after open.

```c
SIPLLM_API void sipllm_close(sipllm_ctx* ctx);
```
Destroy a context and free the underlying runtime.

### Introspection

```c
SIPLLM_API int32_t sipllm_get_model_info(sipllm_ctx* ctx, sipllm_model_info* out);
```
Fill `out` with the loaded model's static description.

```c
SIPLLM_API int32_t sipllm_get_threads(sipllm_ctx* ctx); /* active worker count */
```
Return the active worker count.

### Generation

```c
SIPLLM_API int32_t sipllm_generate(sipllm_ctx* ctx, const char* prompt, int32_t max_new,
                                   const sipllm_sampler* scfg, sipllm_token_cb cb, void* user,
                                   sipllm_stats* stats, char* err_buf, int32_t err_cap);
```
Blocking. Prefills `prompt`, decodes up to `max_new` tokens, streaming each piece through `cb`. Returns tokens generated (`>=0`) or `-1` on error (message in `err_buf`). `stats` and `cb` may be `NULL`.

```c
SIPLLM_API void sipllm_cancel(sipllm_ctx* ctx);
```
Ask the in-flight `generate()` to stop at the next token boundary. Thread-safe.

```c
SIPLLM_API void sipllm_reset(sipllm_ctx* ctx);
```
Clear KV cache / conversation state (start a fresh conversation).

### Embeddings

```c
SIPLLM_API int32_t sipllm_embed_dim(sipllm_ctx* ctx);
```
Hidden size of embedding vectors this model produces (`== model dim`).

```c
SIPLLM_API int32_t sipllm_embed(sipllm_ctx* ctx, const char* text, int32_t pooling,
                                float* out, int32_t out_cap, char* err_buf, int32_t err_cap);
```
Prefill `text`, pool the final-layer hidden state, L2-normalize into `out` (`embed_dim` floats). `pooling` is `SIPLLM_POOL_*`. Returns dim (`>0`) or `-1`. **Clears KV state** — use a dedicated context for embeddings if you also hold a live conversation.

### Device / build info (no context required)

```c
SIPLLM_API const char* sipllm_version(void);
```
Return the engine version string.

```c
SIPLLM_API int32_t sipllm_vulkan_compiled(void);
```
Non-zero if the library was built with the Vulkan backend compiled in.

```c
SIPLLM_API int32_t sipllm_vulkan_available(void);
```
Non-zero if a usable Vulkan device was found at runtime.

> [!CAUTION] Vulkan is detection-only
> `sipllm_vulkan_available()` deliberately rejects CPU software rasterizers (llvmpipe), but even when it reports a device the compute path is **stubbed**: `vulkan_matmul` always falls back to CPU and returns false. `make VULKAN=1` enables device *detection* only. Never treat a non-zero return as working GPU acceleration.

```c
SIPLLM_API const char* sipllm_vulkan_info(void); /* pointer valid until next call, per-context-free */
```
Human-readable accelerator status (device name / fallback reason).

```c
SIPLLM_API int32_t sipllm_hardware_concurrency(void);
```
Logical CPU count reported by the runtime.

```c
SIPLLM_API int32_t sipllm_optimal_threads(const char* model_path, uint64_t ram_budget);
```
Benchmark 1..hw threads for the model and return the fastest count (caches the [auto-tuning](auto-tuning.html) profile). Note: the auto-tuner picks thread count + scheduler policy, **not** the RAM budget.

```c
SIPLLM_API void sipllm_set_log_level(int32_t level);
```
`0`=silent `1`=error `2`=warn `3`=info `4`=debug. Default `3`; set `0`/`1` on mobile.

## See also

- [Dart / Flutter reference](api-dart.html) — the `SipllmRuntime` isolate wrapper over this ABI.
- [C++ engine reference](api-cpp.html) — the `llm::Runtime` this ABI wraps.
- [Flutter runtime](flutter-runtime.html) — architecture of the FFI + isolate layer.
