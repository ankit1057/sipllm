# Glossary

The vocabulary SipLLM's docs and source use, defined precisely. Each term is
grounded in the engine's behavior; where a term names a measured quantity, the
measurement method is stated. Alphabetical.

## Decode
The autoregressive phase after [prefill](glossary.html): the model generates one
token at a time, each pass appending a single position to the [KV cache](glossary.html).
This is where streaming cost dominates, because every token re-streams the layers
it does not have pinned — hence [decode tok/s](glossary.html) is the headline
throughput number.

## Expansion factor
Model size on disk ÷ peak RSS at the minimum ([streaming](glossary.html)) budget —
how many times larger than its resident footprint a model can be. Measured
**2.7×** (smollm2) and **5.5×** (tinyllama); at the largest-runnable extreme,
Llama-2-13B is **25×** its 317&nbsp;MB peak RSS.

## GGUF tensor
A single named weight in a GGUF file — e.g. `blk.<L>.attn_q.weight` — with a
dtype (often quantized), shape, and byte offset in the tensor directory. SipLLM's
`WeightSource` exposes this directory and reads a tensor's raw bytes positionally
(`pread`) without loading the whole file.

## KV cache
The stored key/value projections for every past position, so attention need not
recompute them. It grows with context length, so SipLLM allocates it
**grow-on-demand** (RFC-003 / #35) instead of pre-sizing to `--ctx`, cutting peak
RSS while staying bitwise-identical. See [KV cache](kv-cache.html).

## Layer residency
Which transformer blocks are currently held in RAM. SipLLM keeps only a bounded
set resident — in the extreme, exactly one streamed block — and the
[RAM budget](glossary.html) contract tunes how many. Residency is contiguous:
layers `[0, n_pinned)`. See [layer residency](layer-residency.html).

## Peak RSS
Peak resident set size — the maximum physical memory the process ever held.
SipLLM treats the value from **`/usr/bin/time -l`** as authoritative because it is
comparable across runtimes (independent of a runtime's own accounting). It is the
project's primary success metric.

## Pinned layer
A transformer block the [memory planner](memory-planner.html) keeps permanently
resident (served with zero I/O) because it fit under the [RAM budget](glossary.html).
Pinned layers are the leading contiguous run `[0, n_pinned)`; the rest are
streamed. Pinning is a pure cache — it changes *where* bytes live, never the
logits.

## Prefetch hit / miss
In the async double-buffered loader, a background worker materializes block `L+1`
while the compute thread runs block `L`. A **hit** means `L+1` was ready when
needed (compute never stalls on I/O); a **miss** means compute blocked waiting for
the load. See [predictive prefetch](prefetch.html).

## Prefill
The first forward pass over the whole prompt, populating the [KV cache](glossary.html)
for all prompt positions before [decode](glossary.html) begins. SipLLM does this
**single-pass** (RFC-007 / #36): it streams the model once for the entire prompt
rather than once per prompt token, cutting streamed bytes by up to 18×. See the
[prefill journal](j-prefill.html).

## Quant block
The fixed-size group of weights a quantization format encodes together (with
shared scale/zero-point metadata). SipLLM's fused `matmul_quant` dequantizes *one
output row's* blocks into a tiny scratch buffer and dots them with the activation,
so a whole layer stays quantized in RAM instead of expanding to fp32.

## RAM budget
The hard peak-RSS ceiling set by `--ram-budget N` (#37). The loader pins as many
hot layers as fit under it (after reserving the KV cache and scratch) and streams
the rest, guaranteeing `resident_bytes() ≤ budget`. `--ram-budget 0` (default) is
unlimited streaming. This is SipLLM's defining RAM↔speed dial — see
[memory planner](memory-planner.html).

## Resident weights
The bytes of model weights actually held in RAM at once (distinct from
[peak RSS](glossary.html), which also counts KV cache + scratch). Measured **FLAT
~1.5&nbsp;MB** across toy 4/16/32-layer models — proof the streaming thesis holds:
resident weights track layer width, not model depth. Real single-layer:
37.6&nbsp;MB (smollm2) / 106.5&nbsp;MB (tinyllama).

## SDOT / `--fast`
`--fast` routes Q8_0 projections through an int8 SDOT (signed dot-product) kernel
(`matmul_q8_0_i8`), ARM-only (`__ARM_FEATURE_DOTPROD`), quantizing the activation
for near-parity Q8 decode. It is **opt-in and an approximation** — the exact
fp32-dequant path stays the correctness oracle. Q8_0-only; no AVX2 equivalent.

## Streaming
Loading a transformer block's weights from disk, running it, freeing it, and
reusing that memory for the next block — so peak weight memory tracks *layer* size
(model width), not *model* size (depth). The core execution model that lets a
model larger than RAM run. See [streaming loader](streaming-loader.html).

## Tokenizer kind
Which tokenizer the engine builds from GGUF metadata: **SentencePiece** (Llama) or
**byte-level BPE** (GPT-2 / Llama-3). The engine applies **no chat/prompt
template** — inference is over raw text (BOS only on a fresh sequence); the Flutter
app formats conversations itself. See [tokenizer](tokenizer.html).

## TTFT
Time-to-first-token — wall-clock from generate start to the first emitted token,
i.e. the cost of [prefill](glossary.html). Measured smollm2 ~**0.10&nbsp;s** /
tinyllama ~**0.68&nbsp;s** (vs llama.cpp 0.003 / 0.021&nbsp;s); SipLLM trades TTFT
for footprint in the streaming regime.

## WeightSource
The interface (`include/llm/weight_source.h`) the transformer talks to instead of
a file: a tensor directory (`tensors()`, `find()`), typed metadata
(`meta_int/meta_float/meta_str`), and positional reads (`read_raw` /
`read_raw_at`). Both the real GGUF parser (`GgufFile`) and the toy `.llmw` reader
(`ModelFile`) implement it, so swapping loader or residency strategy touches no
line of math.
