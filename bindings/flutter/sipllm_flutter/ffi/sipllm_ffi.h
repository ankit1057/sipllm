/* sipllm_ffi.h — stable C ABI over the SipLLM streaming inference engine.
 *
 * The engine (headers under include/llm) is C++17 with std::string, std::function,
 * std::unique_ptr in its interface — none of which are FFI-safe. This header
 * is the ONLY surface Dart (or any other language) talks to: opaque handles,
 * plain-old-data structs, and C function-pointer callbacks. It is intentionally
 * additive over the engine (see CLAUDE.md rule 3) — it adds no math and changes
 * no defaults; `sipllm_params` zero-initialized reproduces the CLI's behavior.
 *
 * Threading contract:
 *   - A `sipllm_ctx` is NOT thread-safe for concurrent generate() calls; drive
 *     one context from one worker isolate/thread at a time.
 *   - sipllm_cancel() IS safe to call from another thread while generate() runs;
 *     it flips an atomic the decode loop checks at every token boundary.
 */
#ifndef SIPLLM_FFI_H
#define SIPLLM_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SIPLLM_API __declspec(dllexport)
#else
#define SIPLLM_API __attribute__((visibility("default")))
#endif

/* ThreadPool schedule policy (octa-core work distribution). Mirrors
 * llm::ThreadPool::SchedulePolicy. Fixed16 is the engine default (CLI and
 * sipllm_params_default); a zero-initialized sipllm_params selects STATIC (0). */
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

typedef struct sipllm_ctx sipllm_ctx; /* opaque runtime handle */

/* #78: ram_budget_bytes value meaning "pick for me": half of the RAM the OS
 * reports as available, best effort (an infeasible plan falls back to the
 * minimum configuration). sipllm_params_default() uses it. */
#define SIPLLM_RAM_BUDGET_AUTO UINT64_MAX

/* Open-time knobs. Zero-initialize then override; sipllm_params_default() fills
 * the recommended edge defaults. All bool-ish fields are 0/non-0. */
typedef struct {
  uint64_t ram_budget_bytes; /* SIPLLM_RAM_BUDGET_AUTO (default) = half of available RAM; 0 = stream every layer; else hard peak-RSS ceiling */
  int32_t threads;           /* >0 fixed; 0 = hardware_concurrency; -1 = auto-tune+cache profile   */
  int32_t max_ctx;           /* 0 = engine default (4096)                                          */
  int32_t n_buffers;         /* async prefetch ring buffers (>=1)                                  */
  int32_t use_mmap;          /* mmap backend instead of pread                                      */
  int32_t async_prefetch;    /* background double-buffered prefetch thread                         */
  int32_t fast_quant;        /* int8 kernels for Q8_0/Q4_K/Q6_K (--fast); int8-activation approx. */
  int32_t stream_lm_head;    /* stream non-tied LM head off disk (RAM<->speed knob)                */
  int32_t residency_fp32;    /* FP32 residency (else Quantized — the memory-bounded default)       */
  int32_t force_budget;      /* honor ram_budget even below the safe floor (--ram-budget-force)    */
  int32_t schedule_policy;   /* SIPLLM_SCHED_*                                                     */
  int32_t spec_draft;        /* prompt-lookup speculative decoding, max draft tokens (--spec N);   */
                             /* 0 = off (default). Output is token-identical to plain decoding.   */
                             /* Appended last, so earlier offsets are unchanged, but NOT binary-   */
                             /* compatible: the library copies sizeof(sipllm_params), so a caller  */
                             /* compiled against the older header must be rebuilt.                 */
} sipllm_params;

/* Sampler configuration. Mirrors llm::SamplerConfig. temperature<=0 => greedy. */
typedef struct {
  float temperature;
  int32_t top_k;         /* <=0 disables */
  float top_p;           /* 1.0 disables */
  float repeat_penalty;  /* 1.0 disables */
  int32_t repeat_last_n; /* history window */
  uint64_t seed;
} sipllm_sampler;

/* Static model description, filled by sipllm_get_model_info(). */
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

/* Per-generation metrics, filled by sipllm_generate(). Mirrors llm::GenStats
 * plus engine-owned peak RSS. */
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

/* Streaming token callback. `piece` is UTF-8, NUL-terminated, valid only for
 * the duration of the call. Return non-zero to continue, 0 to stop early. */
typedef int32_t (*sipllm_token_cb)(const char* piece, int64_t token_id, void* user);

/* ---- lifecycle ---------------------------------------------------------- */

/* Fill `out` with recommended edge defaults (streaming, quantized residency,
 * hw threads, Fixed16, async prefetch, repeat_penalty 1.1). */
SIPLLM_API void sipllm_params_default(sipllm_params* out);
SIPLLM_API void sipllm_sampler_default(sipllm_sampler* out);

/* Open a GGUF/.llmw model. On failure returns NULL and writes a message into
 * `err_buf` (NUL-terminated, truncated to err_cap). Thread-affine after open. */
SIPLLM_API sipllm_ctx* sipllm_open(const char* model_path, const sipllm_params* p,
                                   char* err_buf, int32_t err_cap);
SIPLLM_API void sipllm_close(sipllm_ctx* ctx);

/* ---- introspection ------------------------------------------------------ */
SIPLLM_API int32_t sipllm_get_model_info(sipllm_ctx* ctx, sipllm_model_info* out);
SIPLLM_API int32_t sipllm_get_threads(sipllm_ctx* ctx); /* active worker count */

/* ---- generation --------------------------------------------------------- */
/* Blocking. Prefills `prompt`, decodes up to `max_new` tokens, streaming each
 * piece through `cb`. Returns tokens generated (>=0) or -1 on error (message in
 * err_buf). `stats` and `cb` may be NULL. */
SIPLLM_API int32_t sipllm_generate(sipllm_ctx* ctx, const char* prompt, int32_t max_new,
                                   const sipllm_sampler* scfg, sipllm_token_cb cb, void* user,
                                   sipllm_stats* stats, char* err_buf, int32_t err_cap);

/* Ask the in-flight generate() to stop at the next token boundary. Thread-safe. */
SIPLLM_API void sipllm_cancel(sipllm_ctx* ctx);

/* Clear KV cache / conversation state (start a fresh conversation). */
SIPLLM_API void sipllm_reset(sipllm_ctx* ctx);

/* ---- embeddings --------------------------------------------------------- */
/* Hidden size of embedding vectors this model produces (== model dim). */
SIPLLM_API int32_t sipllm_embed_dim(sipllm_ctx* ctx);

/* Prefill `text`, pool the final-layer hidden state, L2-normalize into `out`
 * (embed_dim floats). `pooling` is SIPLLM_POOL_*. Returns dim (>0) or -1.
 * NOTE: clears KV state — use a dedicated context for embeddings if you also
 * hold a live conversation. */
SIPLLM_API int32_t sipllm_embed(sipllm_ctx* ctx, const char* text, int32_t pooling,
                                float* out, int32_t out_cap, char* err_buf, int32_t err_cap);

/* ---- agent (Nishachar), stepped by the caller -------------------------- *
 * The engine owns the conversation, the chat template, tool-call parsing and
 * the step bound; the caller owns the tools, so a tool can be asynchronous (a
 * UI action, a permission prompt) and nothing executes inside the engine.
 *
 *   a = sipllm_agent_new(ctx, &params, err, cap);
 *   sipllm_agent_add_tool(a, "read_file", "Read a text file.");
 *   sipllm_agent_add_param(a, "read_file", "path", SIPLLM_PARAM_STRING, 1, "relative path");
 *   sipllm_agent_start(a, "What does notes.txt say?", err, cap);
 *   for (;;) {
 *     int32_t ev = sipllm_agent_step(a, &sampler, cb, user, &stats, err, cap);
 *     if (ev != SIPLLM_AGENT_TOOL_CALL) break;      // FINAL, a stop, or -1
 *     ... run sipllm_agent_tool_name(a) with its arguments ...
 *     sipllm_agent_submit_result(a, result_text, ok);
 *   }
 *
 * A step generates one model turn (streaming pieces through `cb`) and stops at
 * a closing </tool_call>. sipllm_cancel(ctx) stops an in-flight step at the
 * next token boundary (so does a token callback returning 0); the step then
 * returns SIPLLM_AGENT_CANCELLED and the session is unchanged. Prefill is not
 * interruptible: a cancel lands once the first token is produced. An agent uses its context's KV cache and turns context
 * reuse on for its own steps only: reset the context before chatting on it
 * again. Strings returned by the accessors stay valid until the next step,
 * submit_result, start or free. The agent must be freed before its context. */
enum {
  SIPLLM_PARAM_STRING = 0,
  SIPLLM_PARAM_INT = 1,
  SIPLLM_PARAM_FLOAT = 2,
  SIPLLM_PARAM_BOOL = 3,
  SIPLLM_PARAM_STRING_ARRAY = 4
};

enum {
  SIPLLM_TOOL_FORMAT_AUTO = 0,    /* the model's own format when its template has one */
  SIPLLM_TOOL_FORMAT_COMPACT = 1, /* SipLLM's compact tool list */
  SIPLLM_TOOL_FORMAT_QWEN = 2     /* Qwen2.5 <tools> / <tool_response> */
};

/* sipllm_agent_step() results. */
enum {
  SIPLLM_AGENT_ERROR = -1,      /* message in err_buf; the session is unchanged      */
  SIPLLM_AGENT_TOOL_CALL = 1,   /* run the tool, then sipllm_agent_submit_result()   */
  SIPLLM_AGENT_FINAL = 2,       /* final answer in sipllm_agent_text()               */
  SIPLLM_AGENT_MAX_STEPS = 3,   /* ended: step bound reached without a final answer  */
  SIPLLM_AGENT_EMPTY = 4,       /* ended: the model produced no text                 */
  SIPLLM_AGENT_TOOL_ERROR = 5,  /* ended: a tool failed and stop_on_tool_error is set */
  SIPLLM_AGENT_CANCELLED = 6    /* sipllm_cancel() stopped this step                 */
};

typedef struct {
  int32_t max_steps;          /* 0 = 8 */
  int32_t max_new_tokens;     /* per step; 0 = 256 */
  int32_t tool_format;        /* SIPLLM_TOOL_FORMAT_* */
  int32_t stop_on_tool_error; /* 0: feed the error back to the model and continue */
  const char* system_preamble; /* NULL = Nishachar's default preamble */
} sipllm_agent_params;

typedef struct sipllm_agent sipllm_agent;

SIPLLM_API void sipllm_agent_params_default(sipllm_agent_params* out);
SIPLLM_API sipllm_agent* sipllm_agent_new(sipllm_ctx* ctx, const sipllm_agent_params* p,
                                          char* err_buf, int32_t err_cap);
SIPLLM_API void sipllm_agent_free(sipllm_agent* a);

/* Declare tools before sipllm_agent_start(); both return 0, or -1 on bad input
 * or while a run is active. Re-adding a tool name replaces it. */
SIPLLM_API int32_t sipllm_agent_add_tool(sipllm_agent* a, const char* name, const char* description);
SIPLLM_API int32_t sipllm_agent_add_param(sipllm_agent* a, const char* tool, const char* name,
                                          int32_t type, int32_t required, const char* description);

/* Begin a run for `goal` (ending any previous one). 0, or -1 with err_buf. */
SIPLLM_API int32_t sipllm_agent_start(sipllm_agent* a, const char* goal, char* err_buf, int32_t err_cap);

/* Generate one model turn; returns SIPLLM_AGENT_*. `scfg`, `cb`, `stats` may be
 * NULL (NULL scfg = greedy). */
SIPLLM_API int32_t sipllm_agent_step(sipllm_agent* a, const sipllm_sampler* scfg,
                                     sipllm_token_cb cb, void* user, sipllm_stats* stats,
                                     char* err_buf, int32_t err_cap);

/* The pending tool call (after SIPLLM_AGENT_TOOL_CALL). Arguments are the
 * call's top-level keys; string values are unescaped, other JSON values are
 * returned verbatim. */
SIPLLM_API const char* sipllm_agent_tool_name(sipllm_agent* a);
SIPLLM_API const char* sipllm_agent_tool_args_json(sipllm_agent* a);
SIPLLM_API int32_t sipllm_agent_arg_count(sipllm_agent* a);
SIPLLM_API const char* sipllm_agent_arg_key(sipllm_agent* a, int32_t i);
SIPLLM_API const char* sipllm_agent_arg_value(sipllm_agent* a, int32_t i);

/* The last step's raw model output (the final answer after SIPLLM_AGENT_FINAL). */
SIPLLM_API const char* sipllm_agent_text(sipllm_agent* a);

/* Feed the pending call's result back (ok=0: an error the model should see).
 * Returns 1 if the run continues (call sipllm_agent_step again), 0 if it ended
 * (sipllm_agent_stop_reason), -1 if no call was pending. */
SIPLLM_API int32_t sipllm_agent_submit_result(sipllm_agent* a, const char* result, int32_t ok);

/* Steps taken in the current run, and why it ended (SIPLLM_AGENT_FINAL,
 * _MAX_STEPS, _EMPTY, _TOOL_ERROR), or 0 while it is still running. */
SIPLLM_API int32_t sipllm_agent_steps(sipllm_agent* a);
SIPLLM_API int32_t sipllm_agent_stop_reason(sipllm_agent* a);

/* ---- device / build info (no context required) ------------------------- */
SIPLLM_API const char* sipllm_version(void);
SIPLLM_API int32_t sipllm_vulkan_compiled(void);
SIPLLM_API int32_t sipllm_vulkan_available(void);
SIPLLM_API const char* sipllm_vulkan_info(void); /* pointer valid until next call, per-context-free */
SIPLLM_API int32_t sipllm_hardware_concurrency(void);
SIPLLM_API int32_t sipllm_optimal_threads(const char* model_path, uint64_t ram_budget);

/* 0=silent 1=error 2=warn 3=info 4=debug. Default 3; set 0/1 on mobile. */
SIPLLM_API void sipllm_set_log_level(int32_t level);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SIPLLM_FFI_H */
