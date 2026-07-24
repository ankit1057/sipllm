# Prefill (single-pass, batched over prompt positions)

Prefill runs the prompt through the model to build the KV cache and produce the
first token's logits. Decode then generates one token at a time.

## Why single-pass matters for a streaming runtime

In a resident-weight runtime the whole prompt is processed in one batch for free.
SipLLM streams weights layer-by-layer, so the naive prefill — one `forward()` per
prompt token — streams the **entire model once per prompt token**. A P-token
prompt reads the model from storage P times before the first token appears. This
is invisible on a warm dev box (the page cache absorbs the re-reads) but lethal on
the real target: cold cache, slow phone storage, or a model larger than RAM, where
each re-stream is genuine disk IO.

## How it works

`Transformer::prefill(tokens, n, base_pos)` visits each layer **once**: it loads
(streams) the block, then sweeps all `n` prompt positions through it before
unloading. Causality is preserved because position `i` is processed after every
earlier position, so the K/V history it attends over (`kv_[layer][0..i]`) is
already written. Each per-position computation is identical to the old per-token
path, so output is **bitwise-identical**; only the last position's logits feed the
sampler (exactly as before). Peak RSS is unchanged: only a small per-position
residual scratch (`n × dim` floats) is added — the weights are still 1–2 resident
blocks.

`Runtime::generate` calls `prefill(...)` once instead of looping `forward()` per
token.

## Measured impact (macOS M3, warm cache, threads=4)

| workload | metric | before | after |
|---|---|---|---|
| smollm2, 122-tok prompt | streamed | 14,456 MB | **790 MB** (18.3×) |
| smollm2, 122-tok prompt | TTFT | 3.44 s | 1.96 s |
| tinyllama, 50-tok prompt | streamed | 31,128 MB | **2,882 MB** (10.8×) |
| tinyllama, 50-tok prompt | TTFT | 5.75 s | 4.40 s |

Generation is byte-identical to the per-token path (verified by diff and by
`tests/test_prefill.cpp`, which asserts identical logits + KV + sampled token).
The streamed-bytes reduction is the fundamental win; the wall-time TTFT gain is
modest on a warm-cache SSD and grows with how IO-bound the target is.

Reproduce: `./build/llm <model> -p "<long prompt>" -n 4 --greedy` and compare the
`streamed` / `TTFT` stat lines against a per-token build.
