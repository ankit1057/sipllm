# Single-pass prefill

**Streaming weights turned a free operation into a catastrophically expensive one — and we almost didn't notice.** Prefill runs the prompt through the model to build the KV cache and produce the first token's logits. In a resident-weight runtime it is a single batched pass, essentially free. But SipLLM streams weights layer-by-layer, and the naive prefill — one `forward()` per prompt token — streams the *entire model once per prompt token*. A P-token prompt reads the whole model from storage P times before the first token appears.

## The problem: the warm cache hid a disaster

On a warm dev box the OS page cache absorbs the re-reads, so the bug was invisible: TTFT looked fine and nobody streamed the weights *from disk* twice. But the target is a cold-cache phone with slow storage, or a model larger than RAM where every re-stream is genuine I/O. There, a 122-token prompt meant reading the model 122 times before decode even began — the streaming design's one pathological case.

## The fix: batch over positions, stream each layer once

`Transformer::prefill(tokens, n, start_pos)` (`src/transformer.cpp`) inverts the loop nesting. Instead of *for each token, stream every layer*, it does *for each layer, stream once, then sweep every prompt position through it*:

```text
for layer L in 0 .. n_layers-1:
    loadLayer(L)                 -> stream block L ONCE
    for i in 0 .. n-1:           -> sweep all n prompt positions through it
        block(L, pos = start_pos + i)
    unloadLayer()
final RMSNorm + project_output   -> logits (last position only, feeds the sampler)
```

Causality is preserved for free: position `i` is processed after every earlier position, so the K/V history it attends over (`kv_[layer][0..i]`) is already written. Each per-position computation is identical to the old per-token path, so **output is bit-identical**; only the last position's logits feed the sampler, exactly as before. Peak RSS is unchanged — the only addition is a small `n × dim` residual scratch; the weights are still 1–2 resident blocks. `Runtime::generate` now calls `prefill(...)` once instead of looping `forward()`.

## Measured: the prefill table

M3, warm cache, 4 threads. "Streamed" is total bytes read from the weight source during prefill:

| Workload | Metric | Before | After |
|:--|:--|--:|--:|
| smollm2, 122-tok prompt | streamed | 14,456 MB | 790 MB (18.3×) |
| smollm2, 122-tok prompt | TTFT | 3.44 s | 1.96 s |
| tinyllama, 50-tok prompt | streamed | 31,128 MB | 2,882 MB (10.8×) |

The streamed-bytes reduction — 18.3× and 10.8× — is the fundamental win. The wall-time TTFT gain (3.44&nbsp;s → 1.96&nbsp;s) is modest on a warm-cache SSD and grows with how I/O-bound the target is; on a cold-cache phone the byte reduction *is* the latency.

> [!MEASURED] Provably the same generation
> The output is byte-identical to the per-token path — verified by diff and by `tests/test_prefill.cpp`, which asserts identical logits, identical KV cache, and identical sampled token. This was a pure I/O optimization with zero accuracy cost.

## What it unlocked

Prefill stopped being the streaming model's Achilles' heel: TTFT now scales with prompt length once, not P times, which is what makes long prompts viable on the target hardware. It pairs with [grow-on-demand KV](kv-cache.html) so activation memory stays bounded too. The batched-prefill path is part of the [runtime](runtime.html) and [transformer](transformer.html) architecture pages.
