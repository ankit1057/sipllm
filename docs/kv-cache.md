# KV cache

The KV cache stores each position's projected key/value vectors so a decode step
attends over history without recomputing it. It is indexed `[layer][pos][kv_dim]`
and is **activation memory**, not weights — it must survive across every streamed
layer and every token.

## Grow-on-demand allocation

Because SipLLM streams weights, resident weight RAM is already flat (~1–2 layers).
That makes the KV cache the dominant peak-RSS term: allocated densely in fp32 for
the *full* trained context, it swamped the streamed weights (on SmolLM2-135M the
dense 8k-context KV was 188.7 MB = 79% of peak RSS, even for a 16-token chat).

`KVCache` therefore grows on demand. It keeps a live capacity `cap_` (in
positions), starts at 64, and doubles it (capped at `max_ctx`) only as the
sequence advances. The `[layer][pos][kv_dim]` layout uses `cap_` as the per-layer
stride, so growing changes that stride: `grow_to()` allocates the wider buffer and
re-lays-out each layer's existing rows with `memcpy`. Values are copied verbatim,
so output is **bitwise-identical** to full preallocation — a pure RAM win with
zero accuracy cost. `bytes()` reports the true resident footprint (feeds
`GenStats.kv_bytes`).

Write accessors (`k()/v()` non-const) grow the store so `pos` is resident before
returning the pointer; read accessors (const) never grow, since every position
read has already been written.

## Measured impact (macOS M3, warm cache, `-n 16`, threads=4)

| model | peak RSS before | peak RSS after | KV before | KV after |
|-------|----------------:|---------------:|----------:|---------:|
| SmolLM2-135M | 239 MB | **53.5 MB** | 188.7 MB | 2.9 MB |
| TinyLlama-1.1B Q4_K_M | 210.6 MB | **121 MB** | 92.3 MB | 2.9 MB |

Decode throughput and TTFT are unchanged (values are identical; only the
allocation strategy differs). Peak RSS is measured externally with
`/usr/bin/time -l`; the internal counter matches.

Reproduce: `./scripts/bench.sh` (see the harness) or
`/usr/bin/time -l ./build/llm <model> -p Hello -n 16 --greedy --threads 4`.
