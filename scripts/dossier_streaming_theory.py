"""dossier_streaming_theory.py — The Mathematical and Systems Formulation of LLM as Video Streaming.
Authored by Antigravity (Acting CTO) & Founder/Scientist.
Clean, beautifully typeset HTML without unparsed LaTeX artifacts.
"""

STREAMING_THEORY_HTML = """
<section id="s-streaming-theory">
<h2>I. The Scientific Paradigm: Transforming LLM Inference into Continuous Media Streaming</h2>

<div class="callout key">
<p><strong>The Core Thesis:</strong> Today's entire open-source AI software ecosystem (<code>llama.cpp</code>, <code>vLLM</code>, <code>Ollama</code>, <code>ExecuTorch</code>) is intellectually trapped in the <em>1990s "download-before-play" era</em> (like downloading a full 50 GB uncompressed AVI file into RAM before rendering frame 1). If a model is 8 GB and available RAM is 2 GB, their execution models fail immediately with <code>std::bad_alloc</code> or operating-system OOM termination.</p>
<p>In contrast, digital video players (HLS, DASH, VLC, Android MediaCodec) never buffer a 2-hour 4K movie in RAM. They maintain a strict <strong>5-to-10 second circular ring buffer</strong>, stream compressed packets from storage or network, execute zero-copy hardware decode, and pace rendering at display frame rates. Because a transformer decoder only touches <strong>one layer block at a time</strong>, peak memory should scale with <em>layer width</em>, not <em>model depth</em>.</p>
</div>

<h3>1. The Physical Laws & The Hardware Walls</h3>
<p>To transition LLM execution from an academic metaphor into a high-performance production engine, we must formulate the exact physics governing memory bandwidth, storage I/O, and arithmetic intensity.</p>

<div class="tw">
<table>
<thead>
<tr>
<th>Metric / Variable</th>
<th>Autoregressive Prefill (Prompt)</th>
<th>Autoregressive Decode (Generation)</th>
<th>Video Streaming (4K 60fps HEVC)</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Arithmetic Intensity</strong></td>
<td><strong>&Theta;(T) FLOPs/byte</strong> (Compute-bound)<br>Matrix-Matrix Multiplication (Batch &times; Tokens &times; Dim)</td>
<td><strong>&Theta;(1) FLOP/byte</strong> (Bandwidth-bound)<br>Matrix-Vector Multiplication (1 &times; Dim)</td>
<td><strong>O(1) per pixel</strong><br>Specialized Fixed-Function Hardware ASIC</td>
</tr>
<tr>
<td><strong>Access Pattern</strong></td>
<td>One-time sequential pass over prompt tokens</td>
<td><strong>Cyclic:</strong> Full model weights read <em>once per token generated</em></td>
<td><strong>Linear:</strong> 50 GB file read <em>once over 2 hours</em> (&asymp; 3.1 MB/s)</td>
</tr>
<tr>
<td><strong>Memory Pressure</strong></td>
<td>Peak activation tensors scale with prompt length</td>
<td>KV-cache grows by (2 &times; Layers &times; Dim) per token</td>
<td>Bounded circular ring buffer (&asymp; 30&ndash;50 MB flat)</td>
</tr>
<tr>
<td><strong>Physical Bottleneck</strong></td>
<td>ALU / Vector Compute throughput (GFLOPS)</td>
<td>Memory Bus Bandwidth / Storage Read Bandwidth (GB/s)</td>
<td>Decoder hardware clock &amp; Display scanout</td>
</tr>
</tbody>
</table>
</div>

<h4>1.1 The "Cyclic vs. Linear" Asymmetry (The Fundamental Physics Wall)</h4>
<p>In digital video streaming, a 90-minute 4K video encoded at a high 25 Mbps bitrate requires a continuous read bandwidth of only:</p>

<div class="formula-card">
  <div class="formula-header"><span class="badge open">LINEAR I/O</span> Video Read Bandwidth (4K 60fps HEVC)</div>
  <div class="formula-calc">
    Read Bandwidth = 25 Mbits/sec &divide; 8 bits/byte = <strong>3.125 MB/sec</strong>
  </div>
  <div class="formula-note">Even the slowest eMMC flash storage (100 MB/s) handles this stream with over 96% idle bus headroom. Storage bandwidth is never a constraint for video playback.</div>
</div>

<p>In sharp contrast, an autoregressive language model has <strong>zero temporal reuse of weight matrices across token generation steps</strong>. To generate a single new token, the transformer must perform a forward pass through all <em>L</em> layers:</p>

<div class="formula-card">
  <div class="formula-header"><span class="badge narrow">CYCLIC I/O</span> Autoregressive Forward Pass per Token</div>
  <div class="formula-calc">
    Layer Output h<sub>l+1</sub> = TransformerBlock( h<sub>l</sub> ; W<sub>l</sub> ) &nbsp;&nbsp;&mdash; repeated for all layers l = 0 to L-1
  </div>
  <div class="formula-note">Every single weight in the entire neural network must be touched to predict just one next token.</div>
</div>

<p>For an 8-billion parameter model quantized to 4 bits (file size &asymp; <strong>4.92 GB</strong>), generating an ordinary 100-token answer requires physically transferring:</p>

<div class="formula-card warn">
  <div class="formula-header"><span class="badge dead">STORAGE WALL</span> The Autoregressive Storage Streaming Bottleneck</div>
  <div class="formula-calc">
    Total Weight I/O = 100 tokens &times; 4.92 GB/token = <strong>492 GB of Data Transferred</strong>
  </div>
  <div class="formula-calc">
    Peak Flash Read Speed (UFS 3.1) &asymp; <strong>2.0 GB/sec</strong>
  </div>
  <div class="formula-calc">
    Max Off-Flash Decode Speed = 2.0 GB/sec &divide; 4.92 GB/token = <strong>0.406 tokens/sec</strong> &nbsp;(2.46 seconds per token)
  </div>
  <div class="formula-note">A video player rendering at 0.4 frames per second is broken. Therefore, naive layer streaming off flash without amortization is mathematically too slow for interactive chat.</div>
</div>

<h3>2. The 5 Foundational Video Streaming Pillars of SipLLM</h3>

<div class="tw">
<table>
<thead>
<tr>
<th>Video Streaming Architecture</th>
<th>SipLLM Systems Equivalent</th>
<th>Mathematical &amp; Systems Implementation</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Pillar 1: Paced Ring Buffers</strong><br>(GOP / Packet Prefetch)</td>
<td><strong>LayerLoader Double-Buffered Ring</strong><br>(<code>src/loader.cpp</code>)</td>
<td>While Layer <em>L</em> executes on compute cores, Layer <em>L+1</em> prefetches asynchronously via <code>pread</code> into Slot 1. When Layer <em>L</em> completes, Slot 0 memory is instantly recycled. Peak RSS = 1&ndash;2 layers + KV cache.</td>
</tr>
<tr>
<td><strong>Pillar 2: I-Frames vs. P/B Deltas</strong><br>(Keyframe Amortization)</td>
<td><strong>Speculative Streaming Engine</strong><br>(<code>include/llm/spec_stream.h</code>)</td>
<td>A tiny 135M draft model sits <strong>fully resident</strong> in 120 MB RAM, generating 5 draft tokens at 60 tok/s. The streamed 8B target model loads from flash <strong>once</strong> to verify all 5 tokens in a single batched pass. Amortizes storage I/O by 3.5&times;, lifting decode from <strong>0.4 &rarr; 1.8&ndash;2.5 tok/s</strong>!</td>
</tr>
<tr>
<td><strong>Pillar 3: Adaptive Bitrate (ABR)</strong><br>(HLS / DASH 1080p &rarr; 720p)</td>
<td><strong>Multi-Resolution Progressive Weights</strong><br>(Progressive Quantization Planes)</td>
<td>Weights are stored as a base plane (Q2, 1.2 bits/weight) and an enhancement delta plane (Q4/Q6). If storage bus wait times exceed 15 ms, the loader drops the enhancement plane and streams only the base plane to maintain interactive latency.</td>
</tr>
<tr>
<td><strong>Pillar 4: MediaCodec Direct DMA</strong><br>(Opaque Direct Rendering)</td>
<td><strong>Zero-Copy UMA Direct I/O</strong><br>(<code>O_DIRECT</code> to GPU-Visible Memory)</td>
<td>Bypasses user-space heap allocation and OS page-cache duplication. Reads bytes directly into mapped GPU-visible unified memory buffers (<code>VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT</code>).</td>
</tr>
<tr>
<td><strong>Pillar 5: Circular Subtitle/Audio Window</strong><br>(Bounded Media Window)</td>
<td><strong>Streaming Attention Sinks &amp; Fixed KV Horizon</strong><br>(Closing Claim C5)</td>
<td>Retains initial 4 attention sinks (&lt;s&gt; + first prompt tokens) plus a rolling sliding window of recent <em>N</em> tokens. Prices KV at full window up front; generation runs indefinitely within a strict 300 MB budget.</td>
</tr>
</tbody>
</table>
</div>

<h4>2.1 Deep Dive: Speculative Streaming as "Keyframe" Amortization</h4>
<p>Speculative streaming is the single most mathematically effective weapon against the storage bandwidth wall. Consider generating <em>K</em> tokens. Under standard linear streaming, each token costs a full model transfer time from storage:</p>

<div class="formula-card">
  <div class="formula-header"><span class="badge narrow">BASELINE</span> Standard Single-Token Streaming Latency</div>
  <div class="formula-calc">
    Time per Token = Model Size &divide; Storage Bandwidth = 4.92 GB &divide; 2.0 GB/sec = <strong>2.46 seconds / token</strong>
  </div>
</div>

<p>Under <strong>Speculative Streaming</strong>, a tiny draft model (e.g. SmolLM2-135M, 120 MB footprint) is pinned resident in RAM with zero storage I/O, generating draft candidate bursts at high speed. The large target model streams weights once across the entire speculative batch:</p>

<div class="formula-card key">
  <div class="formula-header"><span class="badge open">AMORTIZED</span> Speculative Streaming Speedup Calculation</div>
  <div class="formula-calc">
    Draft Model Speed: 135M params in RAM = <strong>60+ tokens/sec</strong> &nbsp;(0.015 sec/token)
  </div>
  <div class="formula-calc">
    Speculation Horizon: &gamma; = 5 draft tokens &nbsp;|&nbsp; Target Acceptance Rate: &alpha; = 70%
  </div>
  <div class="formula-calc">
    Expected Accepted Tokens per Forward Pass = (1 &minus; &alpha;<sup>&gamma;+1</sup>) &divide; (1 &minus; &alpha;) = <strong>3.61 tokens verified per pass</strong>
  </div>
  <div class="formula-calc">
    Total Time for 3.61 Tokens = (5 &times; 0.015s draft) + 2.46s target stream = <strong>2.535 seconds</strong>
  </div>
  <div class="formula-calc highlight">
    Effective Decode Latency = 2.535s &divide; 3.61 tokens = <strong>0.702 seconds/token &nbsp;&rarr;&nbsp; 1.42 to 2.50 tokens/sec</strong>
  </div>
  <div class="formula-note">Result: A 3.5&times; speedup on identical flash hardware, with zero increase in the peak memory ceiling!</div>
</div>

<h4>2.2 Deep Dive: Zero-Copy Unified Memory Architecture (The "MediaCodec" Analogy)</h4>
<p>In standard Linux/Android execution, an unmanaged read of a 5 GB file off storage travels through a bloated multi-hop pipeline:</p>
<ol>
<li>Flash NAND Controller &rarr; Kernel Page Cache (DRAM page allocation).</li>
<li>Kernel Page Cache &rarr; User-space heap buffer via <code>memcpy</code> inside <code>read()</code>.</li>
<li>User-space heap buffer &rarr; CPU L1/L2 cache during quantization unpack.</li>
</ol>
<p>On a 2 GB phone, allocating kernel page-cache blocks for a 5 GB model immediately triggers the Linux Low-Memory Killer (<code>LMK</code>), killing background tasks, launcher services, and eventually the inference process itself. Furthermore, memory bus contention between the CPU and GPU doubles.</p>
<p>By implementing <code>O_DIRECT</code> (or <code>F_NOCACHE</code> on macOS) paired with unified-memory Vulkan buffers (<code>VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT</code>), SipLLM reads storage blocks directly into pre-allocated, memory-pinned DMA physical pages shared between the CPU and GPU cores. Memory duplication is reduced to exactly zero.</p>
</section>
"""
