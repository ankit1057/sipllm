# Streaming the LM head

**Layer streaming flattened the transformer blocks, but one weight refused to shrink.** In a non-tied model the output projection (`output.weight`, `[vocab, dim]`) is a single large matrix held resident and used once per token. It does not participate in per-layer streaming, so it sat as a fixed, model-scaling floor on peak RSS — ~51&nbsp;MB on TinyLlama, hundreds of MB on an 8&nbsp;B, ~700&nbsp;MB on a 70&nbsp;B-class head. On the models we most wanted to run, that floor was the binding constraint.

## The problem: a resident weight that doesn't stream

The [layer loader](j-streaming-layers.html) reduced weight RSS to one block, but the LM head is not a block — it lives outside the layer loop, projecting the final hidden state onto the vocabulary. Kept resident, it dominated peak RSS on large models precisely because its size scales with `vocab × dim`, exactly the dimension that grows with model class.

## The fix: read it once per token, in row blocks

`--stream-lm-head` streams the head off disk during the final projection instead of holding it resident. In `LayerLoader::project_output` (`src/loader.cpp`), the non-tied path walks the `[vocab, dim]` matrix in **1024-row chunks** (`BLK = 1024`): `read_raw_at` the chunk into a small buffer, `linear` it against the hidden state, advance. Only ~1K rows are ever resident — the same read-once-per-token pattern as a transformer layer.

```text
for r0 in 0, 1024, 2048, ... n_out:
    rows = min(1024, n_out - r0)
    read_raw_at(offset + r0*row_bytes, buf, rows*row_bytes)   // stream one chunk
    linear(y + r0, buf, x)                                    // project onto vocab slice
```

Tied models (where the LM head *is* the token-embedding table) keep the shared table resident and are unaffected — there is no separate weight to stream. The default stays **resident** because it is fastest and covers the common case where the model fits; you enable streaming when RAM is the constraint.

## Measured: 43% less peak for 3% less decode

TinyLlama-1.1B Q4_K_M, M3, warm cache, 4 threads:

| Mode | Peak RSS | Resident weights | Decode |
|:--|--:|--:|--:|
| default (resident) | 121 MB | 106.5 MB | 11.48 tok/s |
| `--stream-lm-head` | 69 MB | 52.7 MB | 11.11 tok/s |

−43% peak RSS for −3% decode. The head was roughly half the resident-weight budget on TinyLlama, and the reduction scales with `vocab × dim`, so it matters *most* on the large models that would otherwise not fit — it is what makes the [bigger-than-RAM](j-bigger-than-ram.html) 8&nbsp;B/13&nbsp;B runs land at 204&nbsp;MB / 317&nbsp;MB.

> [!MEASURED] Correctness is untouched
> The streamed and resident paths produce byte-identical logits, guarded by `tests/test_e2e.cpp::e2e_streamed_lm_head_matches_resident`. Streaming changes *where* the head lives, never the arithmetic.

## What it unlocked

Removing the last resident floor is what let the expansion factor (model size ÷ RAM) climb into the 20–25× range on real 8–13&nbsp;B models. A future `--ram-budget` will engage `--stream-lm-head` automatically only under RAM pressure. The projection path and the three streaming backends are documented on the [streaming layer loader](streaming-loader.html) architecture page.
