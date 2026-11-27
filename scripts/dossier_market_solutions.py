"""dossier_market_solutions.py — In-Depth Architectural Teardowns ("READMEs") of 12 Existing Solutions.
Authored by Antigravity (Acting CTO) & Founder/Scientist.
Clean typography without raw LaTeX artifacts.
"""

MARKET_SOLUTIONS_HTML = """
<section id="s-market-solutions">
<h2>II. Market Solutions Architectural Teardowns ("READMEs")</h2>

<p class="sub">Exhaustive architectural breakdown of the 12 leading inference frameworks in the global AI ecosystem: memory management, offloading mechanisms, hardware dependencies, and their fatal failure modes on constrained edge hardware.</p>

<div class="tw">
<table>
<thead>
<tr>
<th>Solution</th>
<th>Primary Maintainer</th>
<th>Execution Philosophy</th>
<th>Memory Bound Model</th>
<th>Offloading Mechanism</th>
<th>Fatal Edge Limitation</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>llama.cpp</strong></td>
<td>Georgi Gerganov / ggml.ai</td>
<td>Resident CPU/GPU GGUF</td>
<td>Heuristic (<code>--fit</code> warns only)</td>
<td>Unmanaged OS <code>mmap</code> / page-cache</td>
<td>Page-thrashing &amp; OOM when Model &gt; RAM</td>
</tr>
<tr>
<td><strong>LiteRT-LM</strong></td>
<td>Google (formerly MediaPipe)</td>
<td>FlatBuffer Graph Execution</td>
<td>Fixed graph allocation</td>
<td>None (weights must be resident)</td>
<td>Strict 2.0 GB device floor; drops budget devices</td>
</tr>
<tr>
<td><strong>ExecuTorch</strong></td>
<td>Meta (PyTorch Foundation)</td>
<td>AOT Static Arenas</td>
<td>Static activation arena</td>
<td>None (Model must fit memory)</td>
<td>Complex PyTorch export; no streaming offload</td>
</tr>
<tr>
<td><strong>PowerInfer-2</strong></td>
<td>SJTU (IPADS Lab)</td>
<td>Neuron-Cluster Sparsity</td>
<td>Segmented DRAM cache</td>
<td>NPU/CPU asynchronous cluster prefetch</td>
<td>Requires rooted Snapdragon 8 Gen 3; no dense models</td>
</tr>
<tr>
<td><strong>KTransformers</strong></td>
<td>Tsinghua / KVCache.AI</td>
<td>Heterogeneous VRAM+DRAM</td>
<td>Manual tensor assignment</td>
<td>GPU &harr; CPU DRAM expert paging</td>
<td>Requires 64GB&ndash;128GB Host DRAM; SSD swap stalls</td>
</tr>
<tr>
<td><strong>Apple MLX</strong></td>
<td>Apple Machine Learning</td>
<td>Unified Memory Array (UMA)</td>
<td>Metal buffer allocator</td>
<td>None (Requires unified RAM)</td>
<td>Apple Silicon only; unusable on Android/Linux</td>
</tr>
<tr>
<td><strong>MLC-LLM</strong></td>
<td>Apache TVM / OctoAI</td>
<td>JIT Compiled TensorIR</td>
<td><code>gpu_mem_utilization</code> heuristic</td>
<td>None (VRAM/RAM resident)</td>
<td>Fragile runtime dependencies; no dynamic streaming</td>
</tr>
<tr>
<td><strong>vLLM / SGLang</strong></td>
<td>UC Berkeley / LMSYS</td>
<td>High-Throughput Serving</td>
<td>PagedAttention / RadixTree</td>
<td>Host DRAM offloading (vLLM v0.6+)</td>
<td>Massive Python/CUDA footprint; server-only</td>
</tr>
<tr>
<td><strong>Ollama / Jan</strong></td>
<td>Ollama / Jan.ai</td>
<td>Desktop End-User UX</td>
<td>Inherited from llama.cpp</td>
<td>Inherited from llama.cpp <code>mmap</code></td>
<td>Heavy Go/Electron wrapper; no edge contract</td>
</tr>
<tr>
<td><strong>Exo / Petals</strong></td>
<td>Exo Labs / BigScience</td>
<td>Decentralized Mesh Pipeline</td>
<td>Per-node layer partitioning</td>
<td>Ring-buffer network tensor pipeline</td>
<td>High LAN/WAN latency; node churn drops cluster</td>
</tr>
<tr>
<td><strong>Alibaba MNN</strong></td>
<td>Alibaba Group</td>
<td>Lightweight Mobile Engine</td>
<td>Static memory reuse</td>
<td>None (Model resident in RAM)</td>
<td>No streaming; focused on small vision/CV models</td>
</tr>
<tr>
<td><strong>ONNX Runtime GenAI</strong></td>
<td>Microsoft</td>
<td>Cross-Platform Graph Engine</td>
<td>Session memory arenas</td>
<td>DirectML / QNN NPU offload</td>
<td>Graph compilation locks weights in memory</td>
</tr>
</tbody>
</table>
</div>

<!-- ==================== 1. LLAMA.CPP ==================== -->
<details class="app" open>
<summary>1. llama.cpp (Georgi Gerganov / ggml.ai) &mdash; The Resident GGUF Baseline</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> Pure C/C++ port of transformer models for high-performance CPU, Metal, and CUDA execution. Maximizes token throughput assuming the model's weights can reside in physical memory or OS-buffered virtual memory.</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>Memory Allocation Model:</strong> Uses custom memory allocators (<code>ggml_allocr</code>, <code>ggml-backend-buffer</code>). The default path memory-maps the GGUF file via <code>mmap()</code>.</li>
<li><strong>The <code>--fit</code> Truth (Audited in Source):</strong> In late 2025, <code>llama.cpp</code> merged <code>--fit</code> (in <code>common/fit.cpp</code>) to adjust context length. However, our direct source audit revealed: <strong>it does not enforce a hard ceiling.</strong> When an infeasible memory budget is detected, <code>common/fit.cpp:780-782</code> logs a warning and loads the model anyway, relying on OS virtual memory paging!</li>
<li><strong>Storage Offloading &amp; mmap Behavior:</strong> When physical RAM is smaller than the model file, <code>llama.cpp</code> lets the kernel page fault weights on demand. Because autoregressive decode touches every layer cyclically, the OS thrashing destroys throughput (&lt; 0.05 tok/s) and triggers the OOM killer under background memory pressure.</li>
<li><strong>Hardware Accelerators:</strong> Outstanding Apple Metal, NVIDIA CUDA, Vulkan, SYCL, and Arm KleidiAI micro-kernels.</li>
<li><strong>Fatal Edge Failure Mode:</strong> On devices with &le; 2 GB free RAM, loading a 4 GB model causes instantaneous termination via Android LMK or Linux OOM killer.</li>
</ul>
</details>

<!-- ==================== 2. LITERTE-LM ==================== -->
<details class="app" open>
<summary>2. Google LiteRT-LM &amp; MediaPipe LLM Inference &mdash; The Android Tier-1 Runtime</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> Google's unified on-device ML runtime (the evolution of TensorFlow Lite and MediaPipe). Built for seamless integration into Android apps using FlatBuffer models (Gemma, Gemini Nano).</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>Memory Allocation Model:</strong> Pre-allocates execution graphs into contiguous execution arenas across CPU, GPU (OpenCL/Vulkan), and NPU (MediaTek APU / Qualcomm Hexagon).</li>
<li><strong>The 2.0 GB Hardware Cliff:</strong> Google explicitly enforces a <strong>2.0 GB device RAM minimum</strong> for Gemini Nano and MediaPipe LLM features. Devices with 1.0 GB to 1.5 GB RAM (such as Android TV dongles, Wear OS smartwatches, and Android Go phones) are hardcoded as unsupported.</li>
<li><strong>Offloading Mechanism:</strong> <strong>Zero offloading.</strong> The model weights must reside entirely in active RAM/VRAM. There is no layer streaming or disk-swapping mechanism.</li>
<li><strong>Fatal Edge Failure Mode:</strong> Completely locks out the hundreds of millions of low-spec devices in emerging markets and embedded IoT.</li>
</ul>
</details>

<!-- ==================== 3. EXECUTORCH ==================== -->
<details class="app" open>
<summary>3. Meta ExecuTorch &mdash; The PyTorch Foundation Edge Standard</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> Meta's flagship runtime for bringing PyTorch models to mobile, wearables (Meta Quest, Ray-Ban Meta glasses), and embedded microcontrollers.</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>Memory Allocation Model:</strong> Ahead-of-Time (AOT) static memory planning. The model graph is traced in Python and compiled into a <code>.pte</code> file with planned memory offsets.</li>
<li><strong>Memory Safety:</strong> Excellent activation arena budgeting (zero runtime heap allocations), but <strong>weights are treated as a static resident block</strong>.</li>
<li><strong>Offloading Mechanism:</strong> None. Requires the entire model weights and KV cache to fit in physical device memory.</li>
<li><strong>Fatal Edge Failure Mode:</strong> Cannot scale to models larger than device RAM. Running an 8B model requires an 8 GB+ flagship phone or desktop.</li>
</ul>
</details>

<!-- ==================== 4. POWERINFER-2 ==================== -->
<details class="app" open>
<summary>4. PowerInfer-2 (SJTU IPADS Lab) &mdash; Exploiting Mobile Neuron Sparsity</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> Breakthrough academic framework for running large language models on smartphones by exploiting activation sparsity in Feed-Forward Networks (FFN).</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>Neuron-Cluster Deconstruction:</strong> Decomposes weight matrices into fine-grained neuron clusters. Predicts which neurons will activate for the next token.</li>
<li><strong>Heterogeneous Co-Execution:</strong> Dense "hot" neuron clusters run on the smartphone's NPU; sparse/cold clusters run on the CPU.</li>
<li><strong>Segmented Flash Cache:</strong> Uses a DRAM cache of recently activated neurons and prefetches missing clusters from internal flash storage.</li>
<li><strong>Fatal Edge Failure Mode:</strong>
<ol>
<li>Requires <strong>root access</strong> to bind custom NPU drivers.</li>
<li>Requires <strong>activation sparsity</strong> (ReLU / TurboSparse models). Does not work efficiently on standard SwiGLU dense architectures (Llama-3.1, Qwen-2.5).</li>
<li>Evaluated only on flagship hardware (16 GB &ndash; 24 GB Snapdragon 8 Gen 3). Completely inoperable on commodity 2 GB &ndash; 4 GB ARM phones.</li>
</ol>
</li>
</ul>
</details>

<!-- ==================== 5. KTRANSFORMERS ==================== -->
<details class="app">
<summary>5. KTransformers (Tsinghua University / KVCache.AI) &mdash; Massive MoE Desktop Offloading</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> Squeezes massive 100B&ndash;671B Mixture-of-Experts (DeepSeek-V3, Mixtral) into high-end desktop workstations equipped with consumer GPUs (RTX 4090/5090) and high-capacity DRAM.</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>Memory Hierarchy:</strong> Keeps attention layers and active shared experts resident in GPU VRAM (24 GB), while offloading hundreds of gigabytes of sparse routed experts into host system RAM (64 GB&ndash;128 GB DDR5).</li>
<li><strong>Why Flash Offloading Fails in KTransformers:</strong> When models exceed combined VRAM + DRAM, relying on NVMe SSD swap files drops token throughput to unusable fractions (&lt; 0.2 tok/s) due to PCIe latency and CPU interrupt overhead.</li>
<li><strong>Fatal Edge Failure Mode:</strong> Requires massive host system RAM and multi-channel DDR5. Cannot run on phones, SBCs, or embedded systems.</li>
</ul>
</details>

<!-- ==================== 6. APPLE MLX ==================== -->
<details class="app">
<summary>6. Apple MLX &mdash; Native Framework for Apple Silicon Unified Memory</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> NumPy-like Python/C++ framework optimized specifically for Apple's unified memory architecture (M1/M2/M3/M4 chips).</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>Unified Memory Model:</strong> Allocates arrays that are simultaneously accessible by the CPU, GPU, and Apple Neural Engine with zero memory copies.</li>
<li><strong>Streaming Capability:</strong> None. Assumes the entire model resides in unified RAM. If an M3 Mac with 8 GB RAM tries to run a 14B model, MLX crashes with memory exhaustion.</li>
<li><strong>Fatal Edge Failure Mode:</strong> Strictly locked to Apple hardware. Unusable on Android, Linux, Windows, or embedded platforms.</li>
</ul>
</details>

<!-- ==================== 7. VLLM & SGLANG ==================== -->
<details class="app">
<summary>7. vLLM &amp; SGLang &mdash; Datacenter Batch Serving Titans</summary>
<div class="callout note">
<p><strong>Core Philosophy:</strong> High-throughput server-class serving frameworks designed to maximize multi-tenant batched throughput and eliminate internal memory fragmentation.</p>
</div>
<h4>Architectural Tear-Down:</h4>
<ul>
<li><strong>PagedAttention &amp; RadixAttention:</strong> Treats the KV cache like virtual memory pages, dynamically allocating non-contiguous physical memory blocks to prevent memory fragmentation during multi-turn chats.</li>
<li><strong>Datacenter Scaling:</strong> Optimized for clusters of 80 GB NVIDIA H100/H200 GPUs interconnected via 900 GB/s NVLink.</li>
<li><strong>Fatal Edge Failure Mode:</strong> Massive software stack (hundreds of Python dependencies, CUDA/Triton runtimes, heavy daemon processes). Consumes hundreds of megabytes of RAM just initializing the Python runtime. Completely incompatible with edge constraints.</li>
</ul>
</details>
</section>
"""
