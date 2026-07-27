# Architecture book

> [!NOTE] How to read this section
> This is the map. SipLLM is a layered stack with one load-bearing seam — the `WeightSource` — that lets the transformer run identical math whether weights come from a real GGUF, a toy `.llmw`, resident RAM, or streamed off disk. Each card below is a chapter; start with the [streaming layer loader](streaming-loader.html) if you want the central idea first.

## The stack

Bottom to top, every layer talks only to the one beneath it. The seam is `WeightSource` (`include/llm/weight_source.h`): a tensor directory, typed metadata, and positional `read_raw`/`read_raw_at` reads. Swapping the loader, the residency strategy, or the on-disk format touches no line of transformer math.

```text
                    ┌──────────────────────────────────────────┐
   feeds config &   │  Runtime  (open_model → cfg → generate)   │  src/runtime.cpp
   tokens into ───▶ │  KV cache · sampler · stats               │
                    └────────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │  Transformer  (per-block forward pass)    │  src/transformer.cpp
                    └────────────────────┬─────────────────────┘
                                         │  loadLayer(L) / unloadLayer()
                    ┌────────────────────▼─────────────────────┐
                    │  LayerLoader  (slots · roles · residency  │  src/loader.cpp
                    │  · async prefetch ring)                   │
                    └────────────────────┬─────────────────────┘
                                         │  read_raw / read_raw_at (pread or mmap-memcpy)
                    ┌────────────────────▼─────────────────────┐
                    │  WeightSource seam                        │  include/llm/weight_source.h
                    │    GgufFile (.gguf)  ·  ModelFile (.llmw)  │  src/gguf.cpp · src/format.cpp
                    └───────────────────────────────────────────┘

   underneath everything:  ThreadPool · Quant/dequant kernels · SIMD (NEON/SDOT)
   off to the side:        Device profile + Auto-tuner (thread count & scheduler)
```

`GGUF` parsing and the `Tokenizer` feed this stack from the left — GGUF metadata becomes the `ModelConfig` and the tokenizer vocab; both are built once through the same `WeightSource`. The compute floor (`ThreadPool`, quant kernels, SIMD) and the device/auto-tune path sit underneath as shared services.

## The forward pass in one line

`embed_token → for L in 0..n_layers: loadLayer(L) · RMSNorm · QKV · RoPE · GQA attention · out-proj · RMSNorm · SwiGLU · unloadLayer() → final RMSNorm · project_output → logits` — and only **one block's weights are resident at a time** unless you pin more with `--ram-budget`.

## Chapters

<div class="card-grid"><div class="card"><h3>Streaming layer loader</h3><p>The seam that makes memory track <em>layer</em> size, not model size. <a href="streaming-loader.html">Read →</a></p></div><div class="card"><h3>Layer residency</h3><p>Pinned vs streamed layers and the quantized-vs-fp32 residency modes. <a href="layer-residency.html">Read →</a></p></div><div class="card"><h3>Memory planner</h3><p>The <code>--ram-budget</code> contract planner that pins hot layers under a hard peak-RSS ceiling. <a href="memory-planner.html">Read →</a></p></div><div class="card"><h3>Predictive prefetch</h3><p>The async double-buffered ring that materializes block <code>L+1</code> while block <code>L</code> computes. <a href="prefetch.html">Read →</a></p></div><div class="card"><h3>Runtime</h3><p>Owns the whole stack: <code>open_model</code> → <code>ModelConfig</code> → KV cache → sampler → generate loop. <a href="runtime.html">Read →</a></p></div><div class="card"><h3>Transformer</h3><p>The block math: RMSNorm, GQA attention, RoPE, SwiGLU, per-arch dispatch across nine architectures. <a href="transformer.html">Read →</a></p></div><div class="card"><h3>KV cache</h3><p>Grow-on-demand key/value storage sized to the actual sequence, not the context ceiling. <a href="kv-cache.html">Read →</a></p></div><div class="card"><h3>Quantization</h3><p>Per-row dequantize-then-dot, K-quant NEON fast paths, and the Q8_0 int8 SDOT <code>--fast</code> kernel. <a href="quantization.html">Read →</a></p></div><div class="card"><h3>GGUF parser</h3><p>The real GGUF v2/v3 reader (and its <code>.llmw</code> sibling) behind the <code>WeightSource</code> seam. <a href="gguf-parser.html">Read →</a></p></div><div class="card"><h3>Tokenizer</h3><p>One tokenizer, three kinds — Byte, SentencePiece, byte-level BPE — all built from GGUF metadata. <a href="tokenizer.html">Read →</a></p></div><div class="card"><h3>Thread pool</h3><p>The pthread work-sharing pool and its schedule policies that parallelize every matmul. <a href="thread-pool.html">Read →</a></p></div><div class="card"><h3>Auto-tuning</h3><p>Micro-benchmarks that pick thread count and scheduler per machine, cached under <code>~/.sipllm</code>. <a href="auto-tuning.html">Read →</a></p></div><div class="card"><h3>Flutter runtime</h3><p>The FFI bridge and Dart SDK that turn the engine into an offline on-device runtime. <a href="flutter-runtime.html">Read →</a></p></div><div class="card"><h3>Wear support</h3><p>Bringing the streaming runtime down to a watch-class device. <a href="wear-support.html">Read →</a></p></div></div>

## Where to go next

New to the thesis? Read [Why streaming?](why-streaming.html) and the [streaming-layers journal](j-streaming-layers.html). Want the measured payoff? See [Benchmarks](benchmarks.html). Building on the engine? Jump to the [API reference](api.html).
