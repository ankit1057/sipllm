# Dart / Flutter reference

`package:sipllm_flutter` binds the streaming engine to Flutter through the [C ABI](api-c.html), and adds the surrounding machinery a mobile app needs: a resumable Hugging Face downloader, an on-device SQLite embedding store, device/arch detection, and phone&harr;Wear OS model transfer. The inference runtime is isolate-backed — one `SipllmRuntime` owns one worker isolate holding one native context. This page reproduces the public surface verbatim from `lib/src/**`.

> [!NOTE] Reference
> Signatures are copied from source. Types that cross the isolate/MethodChannel boundary are plain values (no pointers). The runtime wraps the [C ABI](api-c.html) one-to-one; see [Flutter runtime](flutter-runtime.html) for the isolate architecture and [Wear support](wear-support.html) for the transfer bridge.

## Inference runtime

### `SipllmRuntime`

A loaded model driven from a background isolate. Exposes streaming generation, cancellation, embeddings, and per-run stats.

```dart
static Future<SipllmRuntime> open(
  String modelPath, {
  SipllmParams params = const SipllmParams(),
  String? libraryPath,
  int logLevel = 1,
});                                        // open a GGUF/.llmw model on a worker isolate

Stream<SipllmToken> generate(
  String prompt, {
  int maxNew = 256,
  SipllmSampler sampler = const SipllmSampler(),
});                                        // stream pieces; populates lastStats on completion

Future<String> complete(
  String prompt, {
  int maxNew = 256,
  SipllmSampler sampler = const SipllmSampler(),
});                                        // collect the full completion as one string

void cancel();                             // stop the in-flight generation (thread-safe)

Future<Float32List> embed(
  String text, {
  Pooling pooling = Pooling.last,
});                                        // L2-normalized embedding for `text`

void reset();                              // clear KV cache / conversation state
Future<void> close();                      // free native context, shut down the isolate

final SipllmModelInfo modelInfo;           // static description of the loaded model
final int threads;                         // worker thread count the engine spun up
SipllmStats? get lastStats;                // metrics from the most recent generation
bool get isBusy;                           // a generation/embed is in flight
```

> [!CAUTION] `embed()` clears KV state
> Like the C ABI's `sipllm_embed`, `embed()` clears the KV cache. Use a dedicated `SipllmRuntime` for embeddings, not one mid-conversation. `generate()` and `embed()` also throw / error if the runtime `isBusy` — a runtime is single-context.

`cancel()` is the one method safe to call while the worker is mid-`generate()`: `SipllmRuntime.open` loads its own native handle in the main isolate so it can invoke the thread-safe native `sipllm_cancel` on the shared context (`bindings/flutter/sipllm_flutter/lib/src/runtime/sipllm_runtime.dart`).

### `SipllmParams`

Open-time configuration. Zero-arg construction mirrors the engine defaults (streaming, quantized residency, hardware threads).

```dart
const SipllmParams({
  this.ramBudgetBytes = 0,
  this.threads = 0,
  this.maxCtx = 0,
  this.nBuffers = 2,
  this.useMmap = false,
  this.asyncPrefetch = true,
  this.fastQuant = false,
  this.streamLmHead = false,
  this.residencyFp32 = false,
  this.forceBudget = false,
  this.schedulePolicy = SchedulePolicy.proportional2,
});

final int ramBudgetBytes;                  // hard peak-RSS ceiling (--ram-budget); 0 = unlimited
final int threads;                         // >0 fixed; 0 = hw concurrency; -1 = auto-tune + cache
final int maxCtx;
final int nBuffers;
final bool useMmap;
final bool asyncPrefetch;
final bool fastQuant;                      // opt-in int8 SDOT kernel for Q8_0 (--fast)
final bool streamLmHead;
final bool residencyFp32;
final bool forceBudget;
final SchedulePolicy schedulePolicy;

SipllmParams copyWithBudgetMiB(int mib);   // convenience: budget in MiB
SipllmParams copyWith({ /* per-field overrides */ });
```

### `SipllmSampler`

Token sampling. `temperature <= 0` selects greedy decoding.

```dart
const SipllmSampler({
  this.temperature = 0.8,
  this.topK = 40,
  this.topP = 0.95,
  this.repeatPenalty = 1.1,
  this.repeatLastN = 64,
  this.seed = 0x2545F4914F6CDD1D,
});

const SipllmSampler.greedy();              // temperature 0, topK 0, topP 1.0

final double temperature;
final int topK;
final double topP;
final double repeatPenalty;
final int repeatLastN;
final int seed;
```

### `SipllmModelInfo`

Static description of a loaded model (returned as `SipllmRuntime.modelInfo`).

```dart
const SipllmModelInfo({
  required this.arch,
  required this.nLayers,
  required this.nHeads,
  required this.nKvHeads,
  required this.dim,
  required this.vocabSize,
  required this.ctxLen,
  required this.tokenizerKind,
});

final String arch;
final int nLayers;
final int nHeads;
final int nKvHeads;
final int dim;
final int vocabSize;
final int ctxLen;
final TokenizerKind tokenizerKind;
```

### `SipllmStats`

Per-generation metrics — the numbers the SipLLM dashboard displays.

```dart
const SipllmStats({
  required this.ttftSeconds,
  required this.prefillTokensPerSecond,
  required this.decodeTokensPerSecond,
  required this.peakRssBytes,
  required this.weightsResidentBytes,
  required this.kvBytes,
  required this.bytesRead,
  required this.prefetchHits,
  required this.prefetchMisses,
  required this.pinnedLayers,
  required this.nLayers,
  required this.promptTokens,
  required this.genTokens,
  required this.ctxUsed,
  required this.ctxMax,
});

final double ttftSeconds;
final double prefillTokensPerSecond;
final double decodeTokensPerSecond;
final int peakRssBytes;
final int weightsResidentBytes;
final int kvBytes;
final int bytesRead;
final int prefetchHits;
final int prefetchMisses;
final int pinnedLayers;
final int nLayers;
final int promptTokens;
final int genTokens;
final int ctxUsed;
final int ctxMax;

double get peakRssMiB;
```

> [!NOTE] Dart stats are a subset
> `SipllmStats` is a *subset* of the C [`sipllm_stats`](api-c.html#sipllm-stats-per-generation-metrics): it drops `load_s` / `prefill_s` / `decode_s` (the per-phase seconds). It keeps TTFT, prefill/decode tok/s, and every byte/count field.

### `SipllmToken`

Emitted for each streamed piece during generation.

```dart
const SipllmToken(this.piece, this.id);
final String piece;
final int id;
```

### Runtime enums

```dart
enum SchedulePolicy {
  static$(kSchedStatic), fixed8(kSchedFixed8), fixed16(kSchedFixed16),
  fixed32(kSchedFixed32), proportional2(kSchedProportional2),
  proportional4(kSchedProportional4), adaptive(kSchedAdaptive);
  final int code;
}                                          // proportional2 is the engine default

enum Pooling { last(kPoolLast), mean(kPoolMean); final int code; }
enum TokenizerKind { sentencePiece, bpe, byte }
```

## Device / architecture

### `SipllmDevice`

Static engine + accelerator capabilities that need no model. `SipllmDevice.load({String? libraryPath})`.

```dart
String get engineVersion;                  // native version string
int get hardwareConcurrency;               // logical CPU count
bool get vulkanCompiled;                    // built with Vulkan backend
bool get vulkanAvailable;                   // usable Vulkan device found at runtime
String get vulkanInfo;                      // device name / fallback reason
int optimalThreads(String modelPath, {int ramBudgetBytes = 0});  // benchmark + cache profile
void setLogLevel(int level);
```

> [!CAUTION] Vulkan is detection-only
> `vulkanAvailable` may be true, but the compute path is stubbed and always falls back to CPU. Never treat it as working GPU acceleration.

### `ArchDetector`, `DeviceProfile`, `CpuArch`

Device/ABI profile + Wear OS detection.

```dart
class ArchDetector {
  ArchDetector([MethodChannel? channel]);
  Future<DeviceProfile> detect();          // native on Android; inferred from dart:io elsewhere
}

enum CpuArch {
  arm64('arm64-v8a', true), arm32('armeabi-v7a', false),
  x64('x86_64', true), x86('x86', false), unknown('unknown', false);
  final String androidAbi;
  final bool is64Bit;
  static CpuArch fromAbi(String abi);
}

class DeviceProfile {
  const DeviceProfile({
    required this.primaryArch, required this.supportedAbis, required this.isWearOs,
    required this.cores, required this.model, required this.androidSdkInt,
  });
  final CpuArch primaryArch;
  final List<CpuArch> supportedAbis;
  final bool isWearOs;
  final int cores;
  final String model;
  final int androidSdkInt;
  bool get is64Bit;
  int get suggestedRamBudgetBytes;         // conservative default per device class
  Map<String, Object> toMap();
}
```

## Hugging Face download manager

A resumable, multi-connection downloader on `dart:io` `HttpClient` alone (no Flutter), so it is host-VM unit-testable.

### `HuggingFaceRepo`

```dart
const HuggingFaceRepo(this.repo, {this.revision = 'main'});
final String repo;                         // "<owner>/<name>"
final String revision;                     // branch, tag, or commit sha
String resolveUrl(String filename);        // https://huggingface.co/<repo>/resolve/<rev>/<file>
```

### `DownloadManager`

Owns a shared `HttpClient` and a global connection budget, and mints `DownloadTask`s. The `maxConnections` cap is enforced across *all* tasks.

```dart
DownloadManager({int maxConnections = 4});

List<DownloadTask> get tasks;              // all tasks ever enqueued, newest last

DownloadTask enqueue({
  required String url,
  required String destPath,
  int? expectedSize,
  String? sha256,
  int? connections,
  Map<String, String>? headers,
});                                        // create + start a download; resumes from a sidecar

Future<void> disposeAll();                 // pause every task, flush offsets, close the client
```

### `DownloadTask`

A single resumable download.

> [!WARNING] Do not construct directly
> The `DownloadTask` constructor is `@internal`. Mint tasks via `DownloadManager.enqueue`; observe `progress`, await `done`, and steer with `pause`/`resume`/`cancel`.

```dart
final String url;
final String destPath;                     // final assembled output path
final int? expectedSize;                   // verified after assembly
final String? sha256Hex;                   // lowercase-hex sha256, verified after assembly

Stream<DownloadProgress> get progress;     // throttled (~4/sec) samples; terminal sample before done
Future<void> get done;                     // completes on completed/canceled; errors on failed
DownloadState get state;
Future<void> get settled;                  // resolves when the current run cycle unwinds

void start();                              // kick off the first run (no-op past queued)
void pause();                              // stop in-flight connections, flush offsets
Future<void> resume();                     // restart a paused/failed task from committed offsets
void cancel();                             // abort and delete every part file + sidecar
```

### `DownloadProgress`

```dart
const DownloadProgress({
  required this.received,
  required this.total,
  required this.bytesPerSecond,
  required this.activeConnections,
});
final int received;                        // bytes committed to disk across every segment
final int? total;                          // null when the server never disclosed a length
final double bytesPerSecond;               // ~2s moving-average throughput
final int activeConnections;
double? get fraction;                      // completion in [0,1], or null when total unknown
```

### `DownloadState`

```dart
enum DownloadState { queued, running, paused, completed, failed, canceled }
// completed/failed/canceled are terminal; paused is a resting state resume() revives.
```

## On-device embeddings

A local SQLite-backed vector store (`package:sqlite3`), host-VM testable. Engine embeddings are L2-normalized, so cosine similarity is a dot product; search is a full linear scan.

### `EmbeddingStore`

```dart
static EmbeddingStore open(String path, {required int dim});   // file-backed; dim must match
static EmbeddingStore inMemory({required int dim});            // private, ephemeral

final int dim;                             // fixed vector dimensionality
int get length;                            // number of embeddings stored

int add({
  required String text,
  required Float32List embedding,
  Map<String, Object?>? metadata,
});                                        // insert one; returns row id

void addBatch(List<EmbeddingRecord> records);   // insert many in one transaction

List<EmbeddingMatch> search(
  Float32List query, {
  int topK = 5,
  bool Function(Map<String, Object?>? meta)? where,
});                                        // top-K by descending cosine similarity

EmbeddingRecord? get(int id);              // null if absent
void delete(int id);                       // no-op if absent
void clear();                              // remove all, keep dim
void close();                              // dispose the database
```

### `EmbeddingRecord`, `EmbeddingMatch`

```dart
const EmbeddingRecord({
  required this.text,
  required this.embedding,                 // length must equal store dim; expected L2-normalized
  this.metadata,                           // JSON-encodable
});
final String text;
final Float32List embedding;
final Map<String, Object?>? metadata;

const EmbeddingMatch({
  required this.id,
  required this.text,
  required this.score,                     // cosine similarity in [-1, 1]; higher is closer
  this.metadata,
});
final int id;
final String text;
final double score;
final Map<String, Object?>? metadata;
```

## Phone &harr; Wear OS transfer

Copies a model file between a paired phone and watch over the Wearable Data Layer `ChannelClient`. See [Wear support](wear-support.html).

### `WearTransfer`

One transfer at a time (the Android bridge is single-slot).

```dart
WearTransfer({MethodChannel? methodChannel, EventChannel? eventChannel});

Future<List<WearNode>> connectedNodes();   // reachable Data Layer nodes (empty if unpaired)
Stream<TransferProgress> events();          // broadcast send + receive progress
Future<void> sendModel(String path, {String? nodeId, bool resume = true});  // start a send
Future<void> listenIncoming();              // watch side: accept an incoming channel (idempotent)
Future<void> pause();                       // close the channel, retain offsets
Future<void> resume();                      // reopen + re-negotiate the resume offset
Future<void> cancel();                      // abort, discard partial file
```

### `WearNode`

```dart
const WearNode({required this.id, required this.displayName, required this.nearby});
final String id;                            // opaque Data Layer node id
final String displayName;
final bool nearby;                          // Data Layer reachability signal
factory WearNode.fromMap(Map<Object?, Object?> m);
```

### `TransferProgress`

```dart
const TransferProgress({
  required this.direction,
  required this.nodeId,
  required this.filename,
  required this.sent,
  required this.total,
  required this.bytesPerSecond,
  required this.state,
});
final TransferDirection direction;
final String? nodeId;                       // null before a node is resolved
final String filename;                      // model file basename
final int sent;                             // bytes transferred so far
final int? total;                           // null until known
final double bytesPerSecond;                // 0 while connecting
final TransferState state;
double? get fraction;                       // [0,1], or null when total unknown
bool get isTerminal;                        // completed/failed/canceled
factory TransferProgress.fromMap(Map<Object?, Object?> m);
```

### `TransferDirection`, `TransferState`

```dart
enum TransferDirection {
  send, receive;
  static TransferDirection fromWire(String? s);   // defaults to send on unknown
}

enum TransferState {
  idle, connecting, transferring, paused, completed, failed, canceled;
  static TransferState fromWire(String? s);       // defaults to idle on unknown
}
```

## See also

- [C ABI reference](api-c.html) — the native surface this package binds.
- [Flutter runtime](flutter-runtime.html) — the isolate + FFI architecture.
- [Wear support](wear-support.html) — the Data Layer transfer design.
