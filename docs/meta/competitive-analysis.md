# Competitive analysis

Where SipLLM sits among CPU/edge and server LLM runtimes. The comparison is
**positioning, not scoring**: SipLLM optimizes one axis — provably bounded peak
RSS while running a model larger than RAM — that most runtimes were never built
to target. A gap in the table below means *"not a design goal for that project,"*
not a defect. Server runtimes like vLLM and TensorRT-LLM are excellent at what
they aim for (datacenter throughput on GPUs); they simply aim elsewhere.

> [!KEY] The one claim
> Grounded in `CHANGELOG.md` (Wave 6): **llama.cpp's mmap has no hard ceiling**,
> and **vLLM / TensorRT-LLM / MLX / MLC / ExecuTorch require the model to fit in
> RAM/VRAM.** SipLLM's differentiator is the hard `--ram-budget` ceiling that
> lets a model many times larger than RAM run at a peak RSS *you* choose.

## Positioning matrix

| Runtime | Streams layers | Hard RSS budget | Runs model &gt; RAM | Android / Flutter SDK | Live residency viz |
|:--|:--:|:--:|:--:|:--:|:--:|
| **SipLLM** | yes | yes (`--ram-budget`) | yes | yes (WIP, host-verified) | planned |
| llama.cpp | mmap only | no hard ceiling | partial (via mmap) | no first-party | no |
| vLLM | no | no | no (must fit VRAM/RAM) | no | no |
| TensorRT-LLM | no | no | no (must fit VRAM) | no | no |
| MLX | no | no | no (must fit unified mem) | no | no |
| MLC-LLM | no | no | no (must fit RAM/VRAM) | partial (mobile builds) | no |
| ExecuTorch | no | no | no (must fit RAM) | partial (on-device runtime) | no |
| Ollama | mmap (via llama.cpp) | no hard ceiling | partial (via mmap) | no | no |

## How to read each column

- **Streams layers** — loads and evicts transformer weights block-by-block so
  resident weights track layer width, not model size. SipLLM's core mechanism
  (`LayerLoader`, see [streaming loader](streaming-loader.html)). llama.cpp /
  Ollama approximate this only through `mmap` page-cache behavior, not an explicit
  streamed loader.
- **Hard RSS budget** — a *provable* peak-memory ceiling the operator sets.
  Unique to SipLLM (`--ram-budget`, guarded so `resident_bytes() ≤ budget`).
  mmap-based runtimes have soft, OS-decided residency with no ceiling.
- **Runs model &gt; RAM** — SipLLM does this by construction (Llama-2-13B,
  7.87&nbsp;GB Q4, in 317&nbsp;MB). mmap runtimes can *start* a larger-than-RAM
  model but the OS may still fault most of it in — "partial." Server/unified-memory
  runtimes require the whole model resident.
- **Android / Flutter SDK** — a first-party mobile binding. SipLLM ships a C ABI
  + Dart `SipllmRuntime` (Wave 8), **host-verified, not yet on-device**. MLC-LLM
  and ExecuTorch have on-device runtimes but not a Flutter-first SDK.
- **Live residency viz** — a real-time view of which layers are pinned vs
  streamed. **Planned** for SipLLM (the Flutter app); no runtime ships it today.

> [!NOTE] Why the mmap runtimes are "partial," not "no"
> `mmap` genuinely lets you *open* a model bigger than RAM, and under light
> pressure much of it stays paged out — a real and useful capability. It is
> marked "partial" only because there is no hard ceiling: the kernel, not the
> operator, decides residency, and it can fault the whole file in. SipLLM's
> difference is the contract, not the ability to open the file.

## Honest caveats about SipLLM's own row

Fairness cuts both ways. SipLLM is **not** competitive on raw decode throughput
against a GPU server runtime, and its streaming (exceeds-RAM) regime is
disk-bandwidth-bound (<1 tok/s at the bounded-memory extreme). The Android SDK is
host-verified only. GPU offload is stubbed (Vulkan is detection-only — see
[design decisions](design-decisions.html)). SipLLM wins on the memory axis and
trades speed for it, on a dial.

> [!CAUTION] This is a living, positioning snapshot
> Competitors evolve fast — flags, quantization, and platform support change
> release to release. Treat this table as SipLLM's *positioning thesis* (which
> axis it uniquely targets), not a benchmark leaderboard. Verify any specific
> competitor capability against its current docs before quoting it. See the
> [measured benchmarks](benchmarks.html) for SipLLM's own numbers, and
> [Why another runtime?](why-another-runtime.html) for the fuller argument.
