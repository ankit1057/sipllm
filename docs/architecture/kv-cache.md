# KV cache

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** Once weights stream one layer at a time (the [streaming loader](streaming-loader.html)), the KV cache becomes the dominant resident term. Growing it on demand is what keeps a short chat from paying for the full trained context. See also the [transformer forward pass](transformer.html) that reads and writes it.

## Problem

Without a cache, generating token `T` re-projects and re-attends over all `T` past positions — O(T²) per step, O(T³) overall. A KV cache stores each position's projected K and V once, so a decode step only projects the *new* token and dots against stored history. But the cache is **activation memory that must survive across every streamed layer and every token**: its footprint is `n_layers · ctx · kv_dim · 2 · 4` bytes. Committed densely for the full trained context up front, it swamps the streamed weights — on SmolLM2-135M a dense 8k-context fp32 KV was **188.7 MB = 79% of peak RSS**, even for a 16-token chat.

## Design

`KVCache` (`include/llm/kv_cache.h`) is a `[layer][pos][kv_dim]` fp32 store that **grows on demand**. It tracks a live capacity `cap_` in positions, starts at `kInitialCap = 64` (or `max_ctx` if smaller), and doubles it — capped at `max_ctx` — only as the sequence advances. Doubling makes growth amortized O(1).

The layout uses `cap_` as the **per-layer stride**, so growing `cap_` changes where every layer's rows live. `grow_to(need)` therefore allocates the wider `k_`/`v_` buffers and re-lays-out each layer's existing rows with `memcpy` from the old stride to the new:

```text
offset(layer, pos) = ((layer * cap_) + pos) * kv_dim

grow_to(need):                       # need > cap_
    nc = cap_; while nc < need: nc <<= 1     # double
    if nc > max_ctx: nc = max_ctx            # hard ceiling (runtime-enforced)
    alloc nk, nv of n_layers * nc * kv_dim
    for each layer l:                        # copy old_row = cap_*kv_dim floats
        memcpy(nk + l*nc*kv_dim, k_ + l*cap_*kv_dim, old_row)
    swap(k_, nk); swap(v_, nv); cap_ = nc
```

Values are copied verbatim, so results are **bitwise-identical** to the old full-preallocation path — a pure RAM win with zero accuracy cost. `bytes()` reports the true resident footprint, which feeds `GenStats.kv_bytes`.

Growth is driven by the accessors, split write vs read:

- **Write path** — the non-const `k(layer,pos)` / `v(layer,pos)` grow the store (`if (pos >= cap_) grow_to(pos+1)`) so `pos` is resident *before* handing back the pointer. The transformer `memcpy`s the freshly projected K/V for `pos` here.
- **Read path** — the const overloads never grow: every position an attention loop reads has already been written, hence is within `cap_`.

`set_seq_len(n)` also grows ahead if a caller sets the filled length before writing; `clear()` resets `seq_len_` to 0 (weights and capacity untouched).

## Alternatives considered

| Approach | Resident KV | Why not the default |
|:--|:--|:--|
| Full `max_ctx` preallocation (original) | ∝ trained context | Pays for an 8k window on a 16-token chat; 79% of peak RSS on smollm2. |
| Grow-on-demand, doubling (**chosen**) | ∝ actual sequence | Amortized O(1); bitwise-identical output; footprint tracks use. |
| Fixed stride + append (no re-layout) | ∝ actual sequence | Would need a stride decoupled from `cap_`; the `cap_`-as-stride layout is simpler and the memcpy is a one-off per doubling (≤ log₂ ctx times). |
| Quantized / paged KV | smaller still | Adds accuracy risk and bookkeeping; the dense-fp32-but-small store already collapses the dominant term. |

## Tradeoffs

- **Re-layout cost vs steady state.** Each doubling memcpys the whole live cache once. Across a sequence that is ≤ log₂(ctx) copies total — negligible against per-token attention — and decode throughput/TTFT are unchanged (measured identical; only the allocation strategy differs).
- **Still fp32, still full history.** This does not shrink the cache below one fp32 K+V per position per layer, and no window is dropped — consistent with the transformer attending the full causal range (no sliding-window attention).
- **`max_ctx` is a hard ceiling** the runtime enforces before every `forward`; `grow_to` never exceeds it.
- **`embed()` clears KV state** — embeddings must use a dedicated runtime, not one mid-conversation.

## KV grow-on-demand — measured

MacOS M3, warm cache, `-n 16`, threads=4; peak RSS external via `/usr/bin/time -l`, matched by the internal counter.

| Model | Peak RSS before | Peak RSS after | KV before | KV after |
|:--|--:|--:|--:|--:|
| SmolLM2-135M | 239 MB | **53.5 MB** | 188.7 MB | 2.9 MB |
| TinyLlama-1.1B Q4_K_M | 210.6 MB | **121 MB** | 92.3 MB | 2.9 MB |

The KV term collapses from the dominant contributor to a rounding error, and with streamed weights already flat, peak RSS follows it down. Reproduce with `/usr/bin/time -l ./build/llm <model> -p Hello -n 16 --greedy --threads 4`.

## Source files

| File | Role |
|:--|:--|
| `include/llm/kv_cache.h` | `KVCache`: `cap_`/`seq_len_`, `k()/v()` write-grow + const read, `offset`, `grow_to`, `bytes` |
| `src/transformer.cpp` | writes K/V at `pos` (`kv_->k/v(layer,pos)`), reads history in the GQA loop, calls `set_seq_len` |
| `docs/kv-cache.md` | the module note and measured before/after |

## Future work

- **Quantized or paged KV** for very long contexts, if the dense-fp32 store ever becomes the ceiling again.
- **Windowed eviction** to pair with real sliding-window attention once the [transformer](transformer.html) models it.
- **Sequence reuse across turns** so a growing chat need not re-`clear()` when the runtime can prove the prefix is unchanged.
