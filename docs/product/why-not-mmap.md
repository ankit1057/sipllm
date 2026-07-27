# Why not just mmap?

<div class="hero"><p class="lead"><strong>Because <code>mmap</code> has no hard ceiling.</strong> It maps the file into the address space and lets the OS decide residency — and the OS can fault the whole model in. SipLLM offers <code>mmap</code> as one backend, but pairs it with an <em>explicit budget contract</em> so peak RSS never exceeds a byte ceiling you set.</p></div>

## What mmap actually buys you

Be fair: `mmap` is a genuinely good technique, and SipLLM ships it as a first-class backend (`--mmap`). Its win is **OS page-cache buffering** — mapped pages are shared with the page cache, so re-reading weights that are still cached is cheap, and cold pages fault in on demand instead of being read up front. For a model that fits comfortably in RAM, letting the kernel manage residency is simple and fast.

> [!NOTE] mmap in SipLLM is not zero-copy
> In the loader, `read_raw_at` **memcpys** from mapped pages into a resident buffer — it does not hand the transformer a pointer into the mapping. The benefit is page-cache buffering, not zero-copy. `FileBacking::prefetch()` (an `fadvise`/`madvise` readahead hook) exists but is currently unused.

## The problem: residency is the OS's decision, not yours

`mmap` gives you `∝ working set, uncontrolled`. There is no mechanism to say *"never exceed 300 MB."* If the access pattern touches every page — which a full forward pass over every layer does, every token — the kernel is free to keep all of them resident. Under memory pressure it evicts by its own policy, not yours. That is fine on a workstation; on a phone it means the model can fault fully in, blow the memory budget, and get the process OOM-killed.

| Approach | Peak RSS | Hard ceiling? |
|:---------|:---------|:--------------|
| Load everything resident | ∝ model size | no — cannot run a model larger than RAM at all |
| `mmap` the whole file | ∝ working set, uncontrolled | **no** — the OS decides residency and can fault the whole model in |
| SipLLM streaming + `--ram-budget` | ≤ your budget | **yes** — the loader enforces it per layer |

## The seam: mmap as a backend, not the strategy

SipLLM separates **how bytes arrive** from **how much stays resident**. Both live behind the `WeightSource` seam, so the transformer math is identical either way:

<div class="card-grid"><div class="card"><h3>Backend (how bytes arrive)</h3><p>Synchronous <code>pread</code>, an async double-buffered <a href="prefetch.html">prefetch</a> ring, or <code>mmap</code> — switchable per run, all producing bit-identical logits. See <a href="streaming-loader.html">the loader</a>.</p></div><div class="card"><h3>Budget contract (how much stays resident)</h3><p>The <a href="memory-planner.html">memory planner</a> (<code>src/mem_plan.cpp</code>) reserves the KV cache and a scratch allowance, derives a weight ceiling from <code>--ram-budget</code>, and the loader pins <code>[0, n_pinned)</code> and streams the rest — with a per-layer guard that keeps <code>resident_bytes() ≤ budget</code>.</p></div></div>

This is the capability no page-cache trick gives you: **bounded RSS under a hard ceiling**, on a smooth RAM↔speed continuum, with output bit-for-bit invariant to the budget (`tests/test_ram_budget.cpp`).

> [!KEY] The distinction
> `mmap` optimizes *throughput on a model that fits*. SipLLM's budget contract guarantees *a ceiling on a model that may not fit*. You can even combine them — use the `mmap` backend for its page-cache win while the budget planner enforces the hard bound.

## When to reach for which

- **Model fits in RAM, want simplicity/speed** → plain `mmap` (or a mature resident runtime) is great.
- **Model must not exceed a fixed memory budget**, or is **larger than RAM** → SipLLM streaming with `--ram-budget`. See [Why streaming?](why-streaming.html) and the [benchmarks](benchmarks.html).
