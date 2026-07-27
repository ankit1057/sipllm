# Transformer forward pass

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** This is the math the [streaming loader](streaming-loader.html) feeds. Every projection is a `linear()` call that never learns whether its weights are fp32 or [quantized](quantization.html); every past key/value lives in the [KV cache](kv-cache.html). For each function named here, see the [C++ engine API](api-cpp.html).

## Problem

The loader makes exactly one transformer block resident at a time. The forward pass has to consume that stream **in lock-step** — touch block `L`'s weights only while `L` is resident, keep no cross-layer state except the residual stream and the KV cache — and it has to do so for *nine* different decoder architectures without duplicating the attention/FFN core nine times. A second constraint is subtler: prefill. Naively, running a P-token prompt through `forward()` streams the whole model P times before decode even starts.

## Design

### One block loop, arch dispatch on a flag

`Transformer::forward(token, pos)` (`src/transformer.cpp`) is the whole decoder:

```text
embed_token(token) -> x_            (one row streamed off disk, or resident if tied)
  + embedding_scale (Gemma), + learned pos-emb (GPT-2)
for L in 0 .. n_layers-1:
    loadLayer(L)                    (stream/await block L's weights)
    block(L, pos)                   (RMSNorm -> QKV -> RoPE -> GQA -> proj -> FFN)
    unloadLayer()                   (release block L for reuse)
final norm + project_output -> logits   (+ final_logit_softcap on Gemma 2)
```

`block()` is a single dispatch seam. It routes on two fields of `ModelConfig`: `is_moe()` first (Mixtral ships as arch `"llama"` with `expert_count>0`), then a `switch` on `arch_kind`:

| `arch_kind` | Handler | Distinguishing shape |
|:--|:--|:--|
| `Llama`, `Mistral`, `Qwen2` | `block_llama` | RMSNorm + RoPE + GQA + SwiGLU; Qwen2 adds q/k/v biases (`add_bias`, no-op when absent) |
| `Gemma2`, `Gemma3` | `block_gemma2` | GeGLU, `(1+w)` RMSNorm, pre **and** post norms, attn/final softcap; Gemma 3 adds QK-norm + per-layer local RoPE base |
| `Phi3` | `block_phi3` | fused `attn_qkv`, fused `ffn_up=[gate;up]`, partial-rotary RoPE |
| `Phi2` | `block_phi2` | *parallel* block — one shared LayerNorm feeds both attn and FFN — fused QKV, partial RoPE, GELU MLP |
| `GPT2` | `block_gpt2` | LayerNorm + learned absolute positions, no RoPE, non-gated GELU MLP, biases everywhere |
| `Unknown` (default) | `block_llama` | recognized string with no dedicated block → treated as Llama |

`is_moe()` short-circuits ahead of the switch to `block_moe`. The full set implemented is Llama, Mistral, Qwen2/2.5, Gemma 2, Gemma 3 (text), Phi-3, Phi-2, GPT-2, and Mixtral/MoE — see the [architectures overview](what-is-sipllm.html).

### Shared attention

Llama and Mixtral share one attention sublayer, `attention_llama()`: RMSNorm → Q/K/V projections (+ optional Qwen2 biases) → RoPE → write K/V to the cache at `pos` → causal GQA (query head `h` reads kv head `h / gqa_group()`) → output projection → residual add. `block_moe` reuses it verbatim and swaps only the FFN: a router `softmax`es over all experts, `partial_sort` takes the top-k, renormalizes their gate weights to sum to 1, and accumulates each selected expert's SwiGLU output. Expert weights live in packed 3-D tensors resident *quantized*; only the selected experts are dequantized per token, so peak RAM stays one layer regardless of expert count.

### RoPE is adjacent-pair, not rotate_half

`apply_rope()` rotates **adjacent element pairs** `(p[2i], p[2i+1])` — this is ggml's `rope_norm`. It reproduces HF's `rotate_half` *only because GGUF Llama weights are permuted at conversion time*; the engine relies on that permutation and never rotates halves itself. Two variants extend the pristine hot loop:

- **llama3 frequency scaling** (`llama3_scale_freq`, issue #9): when `use_llama3_rope()` holds, each wavelength is stretched — high frequencies pass through, low frequencies are divided by `factor`, the band between interpolates. Off by default; identical codegen to plain RoPE on every pre-Llama3 model.
- **partial-rotary** (`rope_rot`): rotate only the first `rope_dim` dims per head, pass the rest through (Phi-2/Phi-3). `rope_dim == head_dim` is full RoPE.
- **per-layer base** (Gemma 3): local (sliding) layers use `rope_theta_local`; every Nth layer (`sliding_window_pattern`) uses the global `rope_theta`.

### Single-pass prefill

`prefill(tokens, n, start_pos)` streams each layer **once** and pushes all P prompt positions through it while resident, shuttling P residual vectors (`resid_`, the only buffer that grows with prompt length) into and out of the single-position scratch `x_` around each `block()` call. Positions are swept ascending, so position `t`'s K/V is committed before any later position attends to it. The computation is byte-for-byte identical to calling `forward()` per token — same kernels, same reduction order — so the resulting KV cache and logits match exactly; only the redundant re-streaming is gone. Only the last position's logits are projected (decode needs no others).

> [!MEASURED] Prefill, single pass vs per-token (M3, warm, t=4)
> smollm2 122-tok prompt: streamed bytes **14,456 MB → 790 MB (18.3×)**, TTFT **3.44 s → 1.96 s**. tinyllama 50-tok prompt: streamed **31,128 MB → 2,882 MB (10.8×)**. See [Single-pass prefill](j-prefill.html).

## Alternatives considered

| Approach | Why not |
|:--|:--|
| One block function per arch | Nine copies of attention to keep in numeric sync; instead one `attention_llama` + config flags, and a `block_gemma2` that folds Gemma 3 in via valid/invalid weight refs. |
| Branch `linear()` on quantization inside the block | The block would carry a dtype `switch` per projection; instead `linear()` dispatches once (fp32 → `matmul`, Q8_0 `--fast` → SDOT, else → `matmul_quant`). The math never sees quantization. See [Quantization](quantization.html). |
| HF `rotate_half` RoPE | Would need weights re-permuted away from the GGUF layout; adjacent-pair matches ggml and consumes unmodified files. |
| Per-token prefill | P full-model streams for a P-token prompt — the dominant TTFT cost in the streaming regime. |

## Tradeoffs

- **Optional weight roles drive behavior, not `if (arch==…)`.** Gemma 3's QK-norm, post-attention/post-FFN norms, and all biases are applied only when the corresponding `Role` resolves to a valid ref, so one code path serves several architectures. The cost is that shape correctness depends on the loader's role→tensor mapping being right.
- **Sliding-window attention is NOT modeled.** Mistral and Gemma 3 attend the full causal range `[0, pos]`; Gemma 3 only varies the RoPE base per layer, it does not mask a window. This is a known deviation from the reference models, not a bug in the loop.
- **No chat template.** The engine runs raw text; BOS is added only on a fresh sequence. Conversation formatting is the caller's job (the [Flutter runtime](flutter-runtime.html)'s `PromptTemplate`).
- **`"nemo"` / Mistral-Nemo is not a recognized arch string** → it falls through `arch_from_name` to `Unknown` → `block_llama`. That happens to work because Nemo is Llama-like and `head_dim` is read explicitly rather than assumed to be `dim / n_heads`.

## Source files

| File | Role |
|:--|:--|
| `include/llm/transformer.h` | `Transformer`: `forward`, `prefill`, `apply_rope`, per-arch `block_*` declarations, reused scratch buffers |
| `src/transformer.cpp` | the forward/prefill loops, `block()` dispatch, all six block bodies, `attention_llama`, RoPE variants |
| `include/llm/model.h` | `ModelConfig` (arch flags: `fused_qkv`, `rope_dim`, `gemma_rmsnorm`, softcaps, MoE counts), `Arch` enum, tensor-name helpers, `WeightRef` |
| `src/model.cpp` | `arch_from_name` / `arch_name`, `ModelConfig::from_source` metadata resolution |
| `include/llm/linear.h` | `linear()` — the one dtype dispatch the block uses for every projection |
| `include/llm/kv_cache.h` | the `[layer][pos][kv_dim]` store the attention loop reads/writes |

## Future work

- **Model sliding-window attention** for Mistral and Gemma 3 (mask the causal range to the trained window) so long-context behavior matches the reference exactly.
- **Recognize `"nemo"` / more arch strings** explicitly instead of leaning on the Unknown→Llama fallback.
- **Batched decode** — the prefill machinery already sweeps P positions through a resident block; extending it to speculative/parallel decode is the throughput lever in the exceeds-RAM regime.
