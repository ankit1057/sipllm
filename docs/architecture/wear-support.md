# Wear OS support & model transfer

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** Running an LLM *on your watch* is only feasible because SipLLM streams weights instead of resident-loading them; a watch never has the RAM to hold a model. This page covers getting the model *onto* the watch (`WearTransfer` over the Wearable Data Layer) and why the [streaming thesis](why-streaming.html) is the enabling precondition. The runtime that actually executes on the watch is the same one described in [flutter-runtime](flutter-runtime.html); the budget it runs under is the [memory planner](memory-planner.html)'s contract.

## Problem

A Wear OS watch is a ~2 GB-class device with no cellular model store and an unreliable, low-power radio. To run inference locally it needs the GGUF file — hundreds of MB — copied from the paired phone. Two hard constraints shape the transfer: (1) Wear OS **does not expose RFCOMM/SPP** raw Bluetooth sockets to apps, and (2) a dropped Bluetooth link or a killed process mid-copy must **never** mean restarting a 400 MB transfer from zero. And once the bytes land, the watch must run the model in a memory budget a fraction of the model's size — which is only possible because the engine streams.

## Design

### Transport: the Wearable Data Layer, not a socket

`WearTransfer` (`lib/src/wear/wear_transfer.dart`) moves the file over the Wearable **Data Layer** `com.google.android.gms.wearable.ChannelClient` — **not** a raw Bluetooth RFCOMM/SPP socket (which Wear OS deliberately does not offer). A `ChannelClient` channel is a bidirectional byte stream, exactly what a large GGUF copy needs, and the platform auto-negotiates the physical link: **Bluetooth for control/handshake**, and the **Wi-Fi High-Bandwidth** path for the bulk stream when both devices sit on the same network. That is far higher throughput than `MessageClient`/`DataClient`, which cap payloads at tens/hundreds of KB.

The devices must already be **paired** through the Wear OS companion app, both running this plugin's package under the same application id so the Data Layer can route the `/sipllm/model-transfer` channel. With no paired peer, `connectedNodes()` returns empty and `sendModel()` reports a `failed` progress event.

### Protocol: resumable JSON header + ack

The wire protocol is resumable. The sender writes a small **JSON header** — `{filename, totalSize, sha256, resumeOffset}` — and the receiver, if it already holds a partial file, **appends** to the destination and **acks with its current length** so the sender seeks forward instead of resending. `pause()` tears the channel down while keeping offsets; `resume()` reopens and re-negotiates the offset; `cancel()` discards the partial file. On completion the receiver verifies the `sha256` before marking the transfer done.

```text
phone (sender)                    Data Layer channel           watch (receiver)
--------------                    /sipllm/model-transfer        ----------------
connectedNodes() -> [watch]
sendModel(path) --- header{filename,totalSize,sha256,resumeOffset} -->
                 <----------------- ack{resumeOffset = current file length} ---
seek(resumeOffset); stream bytes ==(Wi-Fi High-Bandwidth)==> append to <dest>
   TransferProgress events (both ends) ....................... verify sha256 -> completed
```

### API surface and wire types

`WearTransfer` exposes `connectedNodes()`, `sendModel(path, {nodeId, resume = true})`, `listenIncoming()` (watch side — register the incoming-channel callback, idempotent), `pause()`, `resume()`, `cancel()`, and `events()` (a shared broadcast `Stream<TransferProgress>` over the `sipllm/wear/events` `EventChannel`). The plain-value wire types (`lib/src/wear/model_transfer.dart`) cross the `MethodChannel`/`EventChannel` boundary as codec maps holding no platform handles:

| Type | Role |
|:-----|:-----|
| `TransferDirection` | `send` (phone) / `receive` (watch); `fromWire` defaults to `send` on unknown |
| `TransferState` | `idle · connecting · transferring · paused · completed · failed · canceled`, mirrored 1:1 by Kotlin |
| `WearNode` | a discovered peer; `nearby` = Data Layer reachability (a BT/Wi-Fi hop exists), **not** a guarantee the Wi-Fi path is up |
| `TransferProgress` | one snapshot: `direction, nodeId, filename, sent, total, bytesPerSecond, state`, with `fraction` and `isTerminal` |

### Why a watch can run the model at all

`DeviceProfile.suggestedRamBudgetBytes` (`lib/src/device/arch.dart`) returns **~220 MB** (`220 * 1024 * 1024`) when `isWearOs` is true, versus `0` (unlimited streaming, user-raised) on a phone. That tight budget is the point: because the engine streams one transformer block at a time, peak RSS tracks *layer* width, not *model* size, so a model many times larger than the watch's free RAM still runs under a fixed ceiling. **Streaming is the precondition** — without it, a watch could hold no model worth running, and the transfer would be pointless.

> [!KEY] The streaming thesis is what makes on-watch inference feasible
> Getting a 400 MB GGUF onto the watch is only useful because the runtime never has to hold it resident. The ~220 MB watch budget streams the rest off flash, exactly as on the phone and desktop — the same code path, a smaller dial.

## Alternatives considered

| Approach | Why not |
|:---------|:--------|
| Raw Bluetooth RFCOMM/SPP socket | Not exposed to Wear OS apps; the Data Layer is the sanctioned paired-device channel and auto-selects Wi-Fi High-Bandwidth for bulk. |
| `MessageClient` / `DataClient` | Payload-capped at tens/hundreds of KB — unusable for a hundreds-of-MB GGUF. |
| Non-resumable copy | A dropped BT link mid-transfer would restart from zero; the JSON header + length-ack makes resume free. |
| Download the model directly on the watch | The watch radio and battery make a large HF pull impractical; the phone downloads once (see [flutter-runtime](flutter-runtime.html)'s `DownloadManager`) and relays over the fast local link. |
| Resident-load a tiny model on the watch | Defeats the point — streaming lets a watch run models it could never hold, under the ~220 MB budget. |

## Tradeoffs

- **Android-only.** The Wearable Data Layer is a Google Play Services API; `WearTransfer` is a facade over an Android `MethodChannel`/`EventChannel` bridge with no iOS/desktop equivalent.
- **Single-slot.** A `WearTransfer` instance drives **one transfer at a time** (the Android bridge is single-slot) — construct per-transfer or serialize calls.
- **`sendModel()` completes when the transfer *starts*, not finishes.** Its `Future` resolves once the channel is opened and streaming has begun; **observe `events()`** for actual progress and terminal state. Treating the returned `Future` as "transfer done" is a bug.
- **`nearby` is not a bandwidth promise.** It only says the node is addressable over *some* hop; the high-bandwidth Wi-Fi path may not be up.
- **No measured watch throughput.** Wave 8 is host-verified; on-watch transfer rates and inference speed are **N/A** until measured on hardware. No device numbers are fabricated here.

## Source files

| File | Role |
|:-----|:-----|
| `lib/src/wear/wear_transfer.dart` | `WearTransfer` facade: `connectedNodes`/`sendModel`/`listenIncoming`/`pause`/`resume`/`cancel`/`events`, transport + resumability docs |
| `lib/src/wear/model_transfer.dart` | wire types: `TransferDirection`, `TransferState`, `WearNode`, `TransferProgress` |
| `lib/src/device/arch.dart` | `DeviceProfile.isWearOs` detection and the ~220 MB `suggestedRamBudgetBytes` watch budget |
| `example/lib/features/wear/wear_screen.dart` | the Wear shell the phone shell routes to when a watch is detected |

## Future work

- **Multi-slot / queued transfers** so several models can be relayed without serializing at the call site.
- **iOS / cross-platform relay** — today the bridge is Android Data Layer only.
- **Adaptive segmenting** tuned to the negotiated link (Bluetooth-only vs Wi-Fi High-Bandwidth), and integrity beyond the final `sha256` (per-chunk verification for very long transfers).
- **On-watch benchmark capture** to replace the current N/A watch performance figures with measured ones.
