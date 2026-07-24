# Streaming the LM head (`--stream-lm-head`)

In a non-tied model the output projection (`output.weight`, `[vocab, dim]`) is a
large weight held resident and used once per token. Unlike the transformer
layers it does **not** shrink under layer streaming, so it is a fixed,
model-scaling floor on peak RSS: ~51 MB on TinyLlama-1.1B, ~350 MB on
Llama-3-8B, ~700 MB on a 70B-class head.

`--stream-lm-head` streams it off disk in row blocks during the final projection
(`LayerLoader::project_output`), so only a small window (~1K rows) is resident.
It is read once per token — the same pattern as a transformer layer. Tied models
(lm_head == token embeddings) keep the shared table resident and are unaffected.

Default is **resident** (fastest — the common case where the model fits). Enable
streaming when RAM-constrained: it trades a little decode speed for a large
peak-RSS reduction, and the output is byte-identical either way.

## Measured (macOS M3, TinyLlama-1.1B Q4_K_M, warm cache, threads=4)

| mode | peak RSS | resident weights | decode |
|------|---------:|-----------------:|-------:|
| default (resident) | 121 MB | 106.5 MB | 11.48 tok/s |
| `--stream-lm-head` | **69 MB** | 52.7 MB | 11.11 tok/s |

−43% peak RSS for −3% decode. The reduction scales with `vocab × dim`, so it
matters most on the large models that would otherwise not fit — directly raising
the expansion factor (model size / RAM). A future `--ram-budget` will engage it
automatically only when RAM pressure requires it.

Identical logits with vs without streaming are guarded by
`tests/test_e2e.cpp::e2e_streamed_lm_head_matches_resident`.
