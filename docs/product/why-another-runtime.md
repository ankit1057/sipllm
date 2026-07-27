# Why another runtime?

<div class="hero"><p class="lead"><strong>Because every mature runtime requires the model to fit in RAM or VRAM.</strong> SipLLM's differentiator is one capability none of them offers: <em>bounded RSS under a hard ceiling for a model that does not fit</em>. It is not trying to beat llama.cpp at resident inference — it explores a different execution model.</p></div>

## The gap in the landscape

llama.cpp, vLLM, TensorRT-LLM, MLX, MLC, and ExecuTorch are excellent — and they share one assumption: the model's weights are resident (in RAM or VRAM) for the duration. That assumption is invisible until you hit an edge device with lots of storage but little RAM and almost no usable VRAM. There, the model simply does not fit, and no amount of kernel tuning helps.

> [!KEY] The one differentiator
> SipLLM holds peak memory to a **bounded, configurable ceiling** while running models **significantly larger than available RAM**. `--ram-budget` turns *bounded-RSS XOR speed* into a tunable continuum — llama.cpp's `mmap` has [no hard ceiling](why-not-mmap.html); vLLM/TRT-LLM/MLX/MLC/ExecuTorch require the model to fit in RAM/VRAM.

For the full feature-by-feature breakdown, see [Competitive analysis](competitive-analysis.html).

## Positioning, honestly

<div class="card-grid"><div class="card"><h3>vs llama.cpp</h3><p>The closest peer and SipLLM's measurement baseline. Where the model fits, llama.cpp is faster. SipLLM's advantage is footprint: <strong>2.1× less RAM at ~88% of its decode</strong> (TinyLlama Q8, resident), or <strong>13× less RAM</strong> streaming — and it runs models llama.cpp cannot fit at all.</p></div><div class="card"><h3>vs vLLM / TensorRT-LLM</h3><p>Server-class, GPU-first, throughput-optimized runtimes. They assume datacenter memory and hardware. SipLLM targets the opposite end — a single edge device, CPU, no dependencies.</p></div><div class="card"><h3>vs MLX / MLC</h3><p>On-device runtimes, but the weights are resident. SipLLM's streaming loader is orthogonal: it makes the resident set a <em>layer</em>, not the model.</p></div><div class="card"><h3>vs ExecuTorch</h3><p>Mobile-focused export/runtime, still fit-in-memory. SipLLM adds the missing axis — a hard RAM ceiling with graceful degradation to pure streaming below it.</p></div></div>

## The proof

Bounded-memory execution is measured, not claimed — on a 16&nbsp;GB Mac with ~3&nbsp;GB free, where loading these resident is impossible:

<div class="stat-grid"><div class="stat"><div class="stat-value">317 MB</div><div class="stat-label">Llama-2-13B peak RSS</div><div class="stat-sub">7.87 GB Q4 &rarr; 25&times; smaller</div></div><div class="stat"><div class="stat-value">204 MB</div><div class="stat-label">Llama-3.1-8B peak RSS</div><div class="stat-sub">4.92 GB Q4 &rarr; 24&times; smaller</div></div><div class="stat"><div class="stat-value">2.1&times;</div><div class="stat-label">less RAM than llama.cpp</div><div class="stat-sub">at ~88% of its decode (TinyLlama Q8, resident)</div></div></div>

And it is a real engine, not a demo: unmodified GGUF v2/v3 from Hugging Face, nine architectures (Llama, Mistral, Qwen2/2.5, Gemma&nbsp;2, Gemma&nbsp;3 text, Phi-3, Phi-2, GPT-2, Mixtral/MoE), numerically validated against llama.cpp layer-by-layer, with zero runtime dependencies. See [Benchmarks](benchmarks.html).

## Where SipLLM is the wrong choice

Honesty is the point. Reach for another runtime when:

- **The model fits comfortably in RAM/VRAM and you want maximum throughput.** A resident runtime wins — SipLLM's streaming machinery is overhead you do not need.
- **You need working GPU acceleration today.** SipLLM's Vulkan path is <span class="badge exp">experimental</span> (detection-only; `vulkan_matmul` falls back to CPU). It is CPU-first by design.
- **You need server-scale batched throughput.** That is vLLM/TRT-LLM territory.

SipLLM earns its existence in exactly one regime: **the model is bigger than the memory you have, and you still need a hard ceiling.** See [Why streaming?](why-streaming.html) and the [vision](vision.html) for where that leads.
