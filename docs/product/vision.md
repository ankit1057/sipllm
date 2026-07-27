# Vision — SipLLM Studio

<div class="hero"><p class="lead"><strong>SipLLM Studio is the offline AI computer: a six-surface workspace built entirely on top of the streaming runtime.</strong> Every screen exists to make one idea legible &mdash; running a model <em>under a hard memory budget</em>, larger than the RAM you have. This page describes the destination; badges mark what ships today versus what is planned.</p></div>

> [!WARNING] Read the badges
> The runtime is real and measured. Most of the *Studio* surfaces below are **product vision, not shipped code.** Shipped surfaces are <span class="badge on">shipped</span>; planned ones are <span class="badge no">planned</span>. Nothing here is measured on-device yet — the Flutter/FFI layer (Wave 8) is host-verified on Apple M3, **not** yet runtime-verified on a phone or watch.

## The guardrail

Every surface must reinforce the same thesis: **streaming under a hard memory budget.** That constraint is the product.

> [!KEY] Non-goals for Studio
> Do **not** add new inference architectures or model families to chase feature parity. The engine's architecture set (Llama, Mistral, Qwen2/2.5, Gemma&nbsp;2, Gemma&nbsp;3 text, Phi-3, Phi-2, GPT-2, Mixtral/MoE) is deliberate. Studio's job is to *visualize and control* bounded-memory streaming, not to become a general model zoo.

## The six surfaces

<div class="card-grid"><div class="card"><h3>Chat <span class="badge on">shipped</span></h3><p>Streamed token-by-token conversation on a worker isolate, cancellable mid-generate. The app applies the chat template itself (chatml/llama3/zephyr/raw) — the engine runs raw text and adds no template. Host-verified via the Dart&rarr;isolate&rarr;FFI path.</p></div><div class="card"><h3>Models <span class="badge on">shipped</span></h3><p>Model management + the resumable Hugging Face downloader (multi-connection HTTP Range, sidecar resume across restarts, sha256 verify). Backed by the device/engine capability dashboard.</p></div><div class="card"><h3>Bench <span class="badge on">shipped</span></h3><p>On-device benchmark surface over the runtime's own stats (TTFT, prefill/decode tok/s, byte/count fields). Note the Dart <code>SipllmStats</code> is a <em>subset</em> of the C struct — it drops the per-phase <code>load_s</code>/<code>prefill_s</code>/<code>decode_s</code> seconds.</p></div><div class="card"><h3>Play <span class="badge no">planned</span></h3><p>Offline HTML5 games in a WebView — the "AI computer" earns its keep even with no model loaded and no internet.</p></div><div class="card"><h3>Labs <span class="badge no">planned</span></h3><p>Experimental features behind a toggle, kept out of the way of the shipped surfaces until they are proven.</p></div><div class="card"><h3>Settings <span class="badge no">planned</span></h3><p>The RAM budget, thread count, scheduler policy, and backend selection — the dials that make bounded streaming tangible.</p></div></div>

## The killer surfaces (all planned)

These are what make Studio *the offline AI computer* rather than another chat app. All are <span class="badge no">planned</span>.

<div class="card-grid"><div class="card"><h3>AI Playground</h3><p>Live dials (<code>--ram-budget</code>, threads, scheduler) wired to real-time graphs of peak RSS and decode tok/s — the <a href="why-streaming.html">RAM&harr;speed dial</a> made visible as you drag it.</p></div><div class="card"><h3>Live Runtime Visualizer</h3><p>Per-token view of every layer's state: resident · pinned · streaming · loading · evicted. Watch the streaming window slide across the model as it generates — the streaming thesis rendered live.</p></div><div class="card"><h3>Storage Explorer</h3><p>Tap a tensor in the GGUF directory to see its dtype, byte size, file offset, and whether it is currently streamed or resident. The model's on-disk layout, browsable.</p></div><div class="card"><h3>Prompt Arena</h3><p>Run the same prompt across models/budgets side by side and compare output, footprint, and speed.</p></div><div class="card"><h3>AI Arcade</h3><p>The model plays games — a playful stress test of on-device generation that doubles as a demo.</p></div></div>

## Why a workspace, not just a chat box

The engine's most interesting properties — a flat resident footprint across depth, a hard budget you can drag, weights streaming layer by layer — are invisible in a plain chat UI. Studio's surfaces exist to *show* them: the Visualizer proves peak RSS tracks layer width; the Playground proves the budget is a smooth dial; the Storage Explorer proves the model never fully materializes. Each is an argument for [why streaming](why-streaming.html), made interactive.

> [!NOTE] Status, plainly
> Shipped and host-verified (Apple M3, not on-device): the Chat / Models / Bench surfaces and the FFI + isolate + downloader + embedding-store path. Planned: Play, Labs, Settings, and every killer surface above. Android APK and Wear OS transfer are the remaining work of the current wave — see [Roadmap](roadmap.html) and the [Android journal](j-android.html).
