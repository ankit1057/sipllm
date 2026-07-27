# C++ engine reference

The engine lives under `include/llm/` in `namespace llm` and is standard C++17 with no runtime dependencies (pthreads only). This page reproduces the key public surface verbatim, grouped by header. The math flows top-down: `Runtime` owns a `WeightSource` &rarr; `LayerLoader` &rarr; `KVCache` &rarr; `Transformer`, plus a `Tokenizer` and sampler. The [C ABI](api-c.html) is a thin flattening of this surface for FFI callers.

> [!NOTE] Reference
> Signatures are copied verbatim from the headers. For the *design* behind each module see the [architecture book](architecture.html) — this page links each header to its chapter. Overloads and private members are omitted; only the public surface a caller drives is shown.

## Runtime — `runtime.h`

End-to-end generation: prompt prefill + token-by-token decode. Owns the whole stack.

```cpp
// Open a model file by content sniffing (GGUF or .llmw). Returns a WeightSource.
std::unique_ptr<WeightSource> open_model(const std::string& path, bool use_mmap = false);
```

The `GenStats` struct `generate()` fills:

```cpp
struct GenStats {
    int    prompt_tokens = 0;
    int    gen_tokens = 0;
    double load_s = 0;          // model open + config
    double ttft_s = 0;          // time to first generated token (incl. prefill)
    double prefill_s = 0;       // prompt processing time
    double decode_s = 0;        // time generating the rest
    double prefill_tok_s = 0;   // prompt tokens / prefill_s
    double decode_tok_s = 0;    // gen tokens / decode_s
    size_t weights_resident_bytes = 0;
    int    pinned_layers = 0;   // #37: layers pinned resident under --ram-budget
    size_t kv_bytes = 0;
    uint64_t bytes_read = 0;    // total streamed from disk
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_misses = 0;
    int    ctx_used = 0;
    int    ctx_max = 0;
};
```

The `Runtime` class:

```cpp
// Takes ownership of the source. opt controls residency/prefetch/mmap.
// ram_budget_total (bytes, 0 = unlimited) is a TOTAL peak-RSS target; the
// ctor derives the loader's weight ceiling from it (subtracting KV + scratch).
Runtime(std::unique_ptr<WeightSource> src, LayerLoader::Options opt,
        int max_ctx = 0, int threads = 0, size_t ram_budget_total = 0, bool force_budget = false);

const ModelConfig& config() const;
const Tokenizer&   tokenizer() const;
size_t weights_resident_bytes() const;   // loader_->resident_bytes()
size_t kv_bytes() const;                  // kv_->bytes()
ThreadPool* thread_pool();

// Streaming callback: each newly produced piece + token id. Return false to
// stop early. `on_token` may be null.
using TokenCallback = std::function<bool(const std::string& piece, int64_t id)>;

// Generate up to max_new tokens continuing `prompt`. Fills `stats`.
std::string generate(const std::string& prompt, int max_new,
                     SamplerConfig scfg, const TokenCallback& on_token,
                     GenStats* stats);

size_t peak_rss() const;

// Reset conversation state (KV cache).
void reset();
```

See [Runtime](runtime.html) for the design.

## WeightSource — `weight_source.h`

The abstraction the loader streams weights through: a tensor directory, typed metadata, and positional reads. Implemented by both `GgufFile` and the toy `ModelFile`, so swapping loaders touches no math. See [Streaming loader](streaming-loader.html).

```cpp
// Directory entry for one tensor.
struct TensorInfo {
    std::string          name;
    DType                dtype = DType::F32;
    std::vector<int64_t> shape;    // row-major; shape[0] is the outer dim
    uint64_t             offset = 0;   // absolute file offset of raw bytes
    uint64_t             nbytes = 0;   // bytes on disk (post-quant)
    int64_t numel() const;
    int64_t row_elems() const;         // product of all dims except the first
};
```

```cpp
class WeightSource {
public:
    virtual ~WeightSource() = default;

    virtual const std::vector<TensorInfo>& tensors() const = 0;
    virtual const TensorInfo* find(const std::string& name) const = 0;

    // Read the tensor's raw on-disk bytes into `dst` (capacity >= t.nbytes).
    // Positional read (pread) — thread-safe, never loads the whole file.
    virtual void read_raw(const TensorInfo& t, void* dst) const = 0;

    // Read `n` bytes at an absolute file offset (stream a single row).
    virtual void read_raw_at(uint64_t offset, void* dst, uint64_t n) const = 0;

    virtual const uint8_t* mmap_base() const;   // nullptr unless mmap'd
    virtual uint64_t file_size() const = 0;

    virtual bool has_meta(const std::string& key) const = 0;
    virtual const MetaValue* meta(const std::string& key) const = 0;

    int64_t     meta_int(const std::string& key, int64_t def = 0) const;
    double      meta_float(const std::string& key, double def = 0) const;
    std::string meta_str(const std::string& key, const std::string& def = "") const;
};
```

> [!CAUTION] mmap is not zero-copy
> Even with the mmap backend, the loader's `read_raw_at` memcpys from mapped pages into a resident buffer — the win is OS page-cache buffering, not zero copy. `mmap_base()` returns the base only if the file is mapped. See [Why not mmap everything?](why-not-mmap.html).

## LayerLoader — `loader.h`

Drives transformer blocks through the `WeightSource` seam so only one (or, double-buffered, two) block's weights are resident at a time. See [Layer residency](layer-residency.html) and [Memory planner](memory-planner.html).

```cpp
// Per-block weight roles, in consumption order. The core nine:
enum class Role {
    AttnNorm = 0, AttnQ, AttnK, AttnV, AttnOut,
    FfnNorm, FfnGate, FfnUp, FfnDown,
    // ...optional roles (Qwen2 biases, Gemma post/QK norms, Phi-3 fused QKV,
    //    Mixtral MoE tensors, GPT-2/Phi-2 biases)...
    COUNT
};

enum class Residency { Quantized, FP32 };
```

```cpp
struct LayerLoader::Options {
    Residency residency      = Residency::Quantized;
    int       n_buffers       = 2;      // 1 = strict single block, 2 = double buffer
    bool      async           = true;   // run the background prefetch thread
    bool      use_mmap        = false;  // pread by default
    bool      stream_lm_head  = false;  // stream non-tied LM head off disk
    ThreadPool* dequant_pool  = nullptr;
    size_t    ram_budget_bytes = 0;     // 0 = unlimited; else hard weight-RSS ceiling
    bool      fast_quant      = false;  // opt-in int8 SDOT Q8_0 kernel (--fast)
};
```

```cpp
LayerLoader(WeightSource* src, ModelConfig cfg, Options opt);

// Make `layer` resident and current. Blocks until ready. Returns true.
bool loadLayer(int layer);
// Release the current block for reuse (double-buffer recycling).
void unloadLayer();
// Weight from the current resident block.
WeightRef getWeight(Role role) const;

// ---- global (always-resident) weights ----
// Write the embedding row for `token` (cfg.dim fp32 values) into dst.
void embed_token(int64_t token, float* dst) const;
WeightRef output_norm_weight() const;   // fp32, [dim]
WeightRef output_weight() const;         // native dtype, [vocab, dim]
// y = lm_head @ x -> logits. Streams the LM head off disk when non-tied.
void project_output(const float* x, float* y, ThreadPool* pool) const;

const ModelConfig& config() const;
size_t resident_bytes() const;           // approx current RAM for weights
int    pinned_layers() const;            // resident hot layers (#37)
```

## Transformer — `transformer.h`

The decoder forward pass; every weight is fetched through `LayerLoader`. See [Transformer](transformer.html).

```cpp
Transformer(LayerLoader* loader, KVCache* kv, ThreadPool* pool = nullptr);

// Run one token at absolute position `pos`; returns a pointer to the vocab
// logits (owned by the Transformer, valid until the next forward call).
const float* forward(int64_t token, int64_t pos);

// Single-pass batched prefill (RFC-007): stream every layer EXACTLY ONCE over
// all `n` positions [start_pos, start_pos+n). Returns the LAST position's
// logits; KV cache + logits are IDENTICAL to calling forward() per token.
const float* prefill(const int64_t* tokens, int64_t n, int64_t start_pos);

// RoPE (exposed for testing). Optional llama3 frequency scaling via RopeScaling.
struct RopeScaling {
    bool  llama3 = false;
    float factor = 8.f;
    float low_freq_factor = 1.f;
    float high_freq_factor = 4.f;
    float orig_ctx_len = 8192.f;
};
static void apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                       int64_t pos, float theta_base);
static void apply_rope(float* vec, int64_t n_heads, int64_t head_dim,
                       int64_t pos, float theta_base, const RopeScaling& rs);

const ModelConfig& config() const;
size_t peak_rss() const;
```

> [!NOTE] RoPE convention
> `apply_rope` is adjacent-pair (ggml `rope_norm`), not HF `rotate_half`; it relies on GGUF weights being permuted at conversion. Sliding-window attention is **not** modeled for Mistral/Gemma3 (they attend the full causal range).

## ModelConfig — `model.h`

Decoder configuration, resolved from GGUF metadata (`general.architecture` &rarr; the `Arch` dispatch enum). See [GGUF parser](gguf-parser.html).

```cpp
// Build from a WeightSource's metadata (GGUF "<arch>.*" or toy .llmw keys).
static ModelConfig from_source(const WeightSource& src);
```

Key fields (defaults keep every Llama/toy model on the reference path):

```cpp
std::string arch = "llama";          // raw GGUF string
Arch        arch_kind = Arch::Llama; // resolved dispatch enum
int64_t n_layers   = 0;
int64_t n_heads    = 0;
int64_t n_kv_heads = 0;              // < n_heads => GQA
int64_t dim        = 0;              // hidden size / embedding length
int64_t head_dim   = 0;              // usually dim / n_heads
int64_t ffn_dim    = 0;              // feed-forward intermediate size
int64_t vocab_size = 0;
int64_t ctx_len    = 0;              // trained context length
float   rope_theta = 10000.f;
float   rms_eps    = 1e-5f;
bool    tie_embeddings = false;      // output projection shares token_embd
```

Derived dimensions:

```cpp
int64_t q_dim()  const { return n_heads * head_dim; }
int64_t kv_dim() const { return n_kv_heads * head_dim; }
int64_t gqa_group() const { return n_heads / n_kv_heads; }
```

> [!NOTE] Architecture coverage
> The `Arch` enum dispatches `Llama, Mistral, Qwen2, Gemma2, Gemma3, Phi3, Phi2, GPT2, Unknown`. An unrecognized string (e.g. "nemo"/Mistral-Nemo) falls back to `Unknown` &rarr; the Llama path, which works because such models are Llama-like and `head_dim` is read explicitly.

The resident-weight value type the transformer consumes:

```cpp
struct WeightRef {
    const void* data   = nullptr;
    DType       dtype  = DType::F32;
    int64_t     n_out  = 0;
    int64_t     n_in   = 0;
    bool valid() const;
};
```

## Tokenizer — `tokenizer.h`

Text &harr; token ids, driven from GGUF `tokenizer.ggml.*` metadata. See [Tokenizer](tokenizer.html).

```cpp
enum class Kind { Byte, SentencePiece, BPE };

// Build from a model's tokenizer metadata; falls back to Byte if absent.
static Tokenizer from_source(const WeightSource& src);
static Tokenizer byte_tokenizer(int64_t vocab = 256);

std::vector<int64_t> encode(const std::string& text, bool add_bos = true) const;
std::string          decode(const std::vector<int64_t>& ids) const;
std::string          decode_token(int64_t id) const;   // one token (streaming UI)

Kind    kind() const;
int64_t vocab_size() const;
int64_t bos_id() const;
int64_t eos_id() const;
bool    is_eog(int64_t id) const;      // end-of-generation?
```

> [!NOTE] No chat template
> The engine adds BOS only on a fresh sequence and applies **no** chat/prompt template — inference is over raw text. Conversation formatting is the caller's job.

## KVCache — `kv_cache.h`

Grow-on-demand K/V storage indexed `[layer][pos][kv_dim]`. Starts small and doubles (capped at `max_ctx`), re-laying-out rows on growth — bitwise-identical to full preallocation. See [KV cache](kv-cache.html).

```cpp
KVCache(int64_t n_layers, int64_t kv_dim, int64_t max_ctx);

int64_t max_ctx()  const;
int64_t kv_dim()   const;
int64_t n_layers() const;
int64_t seq_len()  const;
int64_t capacity() const;              // currently-resident positions

void set_seq_len(int64_t n);           // grows backing store if needed
void clear();

float*       k(int64_t layer, int64_t pos);        // write path: grows to fit
float*       v(int64_t layer, int64_t pos);
const float* k(int64_t layer, int64_t pos) const;  // read path: never grows
const float* v(int64_t layer, int64_t pos) const;

size_t bytes() const;                  // true resident footprint
```

## ops — `ops.h`

Pure numeric kernels over fp32 buffers. Weight convention: `W` is `[n_out, n_in]` row-major. See [Transformer](transformer.html).

```cpp
void vec_add(float* out, const float* a, const float* b, int64_t n);
void vec_add_inplace(float* x, const float* y, int64_t n);

// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
void rmsnorm(float* out, const float* x, const float* weight, int64_t n, float eps = 1e-5f);
// Gemma-style RMSNorm: learned scale is (1 + weight)
void rmsnorm_gemma(float* out, const float* x, const float* weight, int64_t n, float eps = 1e-6f);

void softmax(float* x, int64_t n);                     // numerically stable, in place
void silu_inplace(float* x, int64_t n);                // x * sigmoid(x)
void gelu_inplace(float* x, int64_t n);                // tanh-approx GELU
void softcap_inplace(float* x, int64_t n, float cap);  // cap * tanh(x/cap); cap<=0 no-op

// LayerNorm (GPT-2 / Phi-2): (x-mean)/sqrt(var+eps) * weight + bias; bias may be null.
void layernorm(float* out, const float* x, const float* weight, const float* bias,
               int64_t n, float eps = 1e-5f);

// Single-vector (batch=1) linear: y = W @ x. Row-parallel across the pool.
void matmul(float* y, const float* W, const float* x,
            int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);
// Batched linear: Y = X @ W^T (prompt prefill).
void matmul_batch(float* Y, const float* X, const float* W,
                  int64_t m, int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);
```

## quant — `quant.h`

Dequantization for the common GGUF family + reference quantizers. See [Quantization](quantization.html).

```cpp
float    fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float    bf16_to_fp32(uint16_t h);

// Expand `n` logical elements of type `t` from `src` into `dst` (fp32).
void dequantize_row(DType t, const void* src, float* dst, int64_t n);

// y = W @ x where W is [n_out, n_in] stored as type `t`. Dequantizes one output
// row at a time into scratch — why a whole layer stays quantized in RAM.
void matmul_quant(float* y, const void* W, DType t, const float* x,
                  int64_t n_out, int64_t n_in, ThreadPool* pool = nullptr);

// Reference quantizers (tests / fixtures). n a multiple of 32.
void quantize_q8_0(const float* src, void* dst, int64_t n);
void quantize_q4_0(const float* src, void* dst, int64_t n);
```

> [!CAUTION] Dequant coverage
> Supported: F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL. **Q8_1 and Q8_K are NOT dequantizable** (they throw). Only Q4_K/Q6_K have NEON fast paths; the int8 SDOT `--fast` kernel is Q8_0-only, ARM-only (`__ARM_FEATURE_DOTPROD`), and an approximation — the fp32-dequant path is the correctness oracle.

## linear — `linear.h`

The single projection call the transformer uses; dispatches on the resident weight's dtype so callers never branch on quantization.

```cpp
inline void linear(float* y, const WeightRef& W, const float* x, ThreadPool* pool = nullptr);
// F32 -> matmul; Q8_0 + --fast -> int8 SDOT; else -> matmul_quant.
```

## DType — `dtype.h`

Element types whose numeric values mirror ggml's `enum ggml_type`.

```cpp
enum class DType : int32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
    Q6_K = 14, Q8_K = 15, IQ4_NL = 20, BF16 = 30, COUNT = 39,
};

const TypeTraits& type_traits(DType t);
int64_t type_nbytes(DType t, int64_t n_elements);   // bytes for n elements
const char* dtype_name(DType t);
```

## ThreadPool — `thread-pool.h`

A minimal persistent worker pool for data-parallel loops (the matmul row split). See [Thread pool](thread-pool.html).

```cpp
explicit ThreadPool(int n_threads = 0);   // <=0 => hardware_concurrency

int size() const;

enum class SchedulePolicy {
    Static, Fixed8, Fixed16, Fixed32, Proportional2, Proportional4, Adaptive
};
void set_policy(SchedulePolicy p);

void parallel_for(int64_t total,
                  const std::function<void(int, int64_t, int64_t)>& fn);

// Process-wide default pool, lazily created.
ThreadPool& default_pool();
```

## See also

- [C ABI reference](api-c.html) — the FFI-safe flattening of this surface.
- [Architecture book](architecture.html) — the design chapter for every module above.
- [Dart / Flutter reference](api-dart.html) — the top-level binding.
