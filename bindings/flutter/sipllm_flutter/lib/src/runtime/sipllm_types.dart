// sipllm_types.dart — ergonomic Dart types over the POD C structs. These cross
// the isolate boundary as plain values, so they hold no pointers.
import '../ffi/sipllm_bindings.dart';

/// How octa-core matmul work is distributed. `fixed16` (small dynamic chunks)
/// is the engine default; on big.LITTLE phones it keeps the slow cores from
/// holding up every matmul.
enum SchedulePolicy {
  static$(kSchedStatic),
  fixed8(kSchedFixed8),
  fixed16(kSchedFixed16),
  fixed32(kSchedFixed32),
  proportional2(kSchedProportional2),
  proportional4(kSchedProportional4),
  adaptive(kSchedAdaptive);

  const SchedulePolicy(this.code);
  final int code;
}

/// Embedding pooling. [last] is correct for causal decoders; [mean] averages
/// per-position hidden states when the engine reports them.
enum Pooling {
  last(kPoolLast),
  mean(kPoolMean);

  const Pooling(this.code);
  final int code;
}

enum TokenizerKind { sentencePiece, bpe, byte }

/// Open-time configuration. Zero-arg construction mirrors the engine defaults
/// (streaming, quantized residency, hardware threads).
class SipllmParams {
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
    this.schedulePolicy = SchedulePolicy.fixed16,
    this.specDraft = 0,
  });

  /// Hard peak-RSS ceiling in bytes (`--ram-budget`). 0 = unlimited streaming.
  final int ramBudgetBytes;

  /// >0 fixed worker count; 0 = hardware_concurrency; -1 = auto-tune + cache.
  final int threads;
  final int maxCtx;
  final int nBuffers;
  final bool useMmap;
  final bool asyncPrefetch;

  /// Opt-in int8 kernels for Q8_0, Q4_K and Q6_K (`--fast`). Quantizes the
  /// activation to int8 (same approximation as llama.cpp); the exact fp32 path
  /// stays the default.
  final bool fastQuant;
  final bool streamLmHead;
  final bool residencyFp32;
  final bool forceBudget;
  final SchedulePolicy schedulePolicy;

  /// Prompt-lookup speculative decoding: max draft tokens per verify pass
  /// (`--spec N`). 0 = off. Output is token-identical to plain decoding; the
  /// win is on copy-heavy output (edits, summaries) and largest when streaming.
  final int specDraft;

  /// Convenience: build a budget in MiB.
  SipllmParams copyWithBudgetMiB(int mib) =>
      copyWith(ramBudgetBytes: mib * 1024 * 1024);

  SipllmParams copyWith({
    int? ramBudgetBytes,
    int? threads,
    int? maxCtx,
    int? nBuffers,
    bool? useMmap,
    bool? asyncPrefetch,
    bool? fastQuant,
    bool? streamLmHead,
    bool? residencyFp32,
    bool? forceBudget,
    SchedulePolicy? schedulePolicy,
    int? specDraft,
  }) {
    return SipllmParams(
      ramBudgetBytes: ramBudgetBytes ?? this.ramBudgetBytes,
      threads: threads ?? this.threads,
      maxCtx: maxCtx ?? this.maxCtx,
      nBuffers: nBuffers ?? this.nBuffers,
      useMmap: useMmap ?? this.useMmap,
      asyncPrefetch: asyncPrefetch ?? this.asyncPrefetch,
      fastQuant: fastQuant ?? this.fastQuant,
      streamLmHead: streamLmHead ?? this.streamLmHead,
      residencyFp32: residencyFp32 ?? this.residencyFp32,
      forceBudget: forceBudget ?? this.forceBudget,
      schedulePolicy: schedulePolicy ?? this.schedulePolicy,
      specDraft: specDraft ?? this.specDraft,
    );
  }

  Map<String, Object> toMap() => {
        'ramBudgetBytes': ramBudgetBytes,
        'threads': threads,
        'maxCtx': maxCtx,
        'nBuffers': nBuffers,
        'useMmap': useMmap,
        'asyncPrefetch': asyncPrefetch,
        'fastQuant': fastQuant,
        'streamLmHead': streamLmHead,
        'residencyFp32': residencyFp32,
        'forceBudget': forceBudget,
        'schedulePolicy': schedulePolicy.index,
        'specDraft': specDraft,
      };

  static SipllmParams fromMap(Map<String, Object?> m) => SipllmParams(
        ramBudgetBytes: m['ramBudgetBytes']! as int,
        threads: m['threads']! as int,
        maxCtx: m['maxCtx']! as int,
        nBuffers: m['nBuffers']! as int,
        useMmap: m['useMmap']! as bool,
        asyncPrefetch: m['asyncPrefetch']! as bool,
        fastQuant: m['fastQuant']! as bool,
        streamLmHead: m['streamLmHead']! as bool,
        residencyFp32: m['residencyFp32']! as bool,
        forceBudget: m['forceBudget']! as bool,
        schedulePolicy: SchedulePolicy.values[m['schedulePolicy']! as int],
        specDraft: (m['specDraft'] as int?) ?? 0,
      );
}

/// Token sampling. `temperature <= 0` selects greedy decoding.
class SipllmSampler {
  const SipllmSampler({
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.repeatPenalty = 1.1,
    this.repeatLastN = 64,
    this.seed = 0x2545F4914F6CDD1D,
  });

  const SipllmSampler.greedy()
      : temperature = 0.0,
        topK = 0,
        topP = 1.0,
        repeatPenalty = 1.1,
        repeatLastN = 64,
        seed = 0x2545F4914F6CDD1D;

  final double temperature;
  final int topK;
  final double topP;
  final double repeatPenalty;
  final int repeatLastN;
  final int seed;

  Map<String, Object> toMap() => {
        'temperature': temperature,
        'topK': topK,
        'topP': topP,
        'repeatPenalty': repeatPenalty,
        'repeatLastN': repeatLastN,
        'seed': seed,
      };

  static SipllmSampler fromMap(Map<String, Object?> m) => SipllmSampler(
        temperature: (m['temperature']! as num).toDouble(),
        topK: m['topK']! as int,
        topP: (m['topP']! as num).toDouble(),
        repeatPenalty: (m['repeatPenalty']! as num).toDouble(),
        repeatLastN: m['repeatLastN']! as int,
        seed: m['seed']! as int,
      );
}

/// Static description of a loaded model.
class SipllmModelInfo {
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

  Map<String, Object> toMap() => {
        'arch': arch,
        'nLayers': nLayers,
        'nHeads': nHeads,
        'nKvHeads': nKvHeads,
        'dim': dim,
        'vocabSize': vocabSize,
        'ctxLen': ctxLen,
        'tokenizerKind': tokenizerKind.index,
      };

  static SipllmModelInfo fromMap(Map<String, Object?> m) => SipllmModelInfo(
        arch: m['arch']! as String,
        nLayers: m['nLayers']! as int,
        nHeads: m['nHeads']! as int,
        nKvHeads: m['nKvHeads']! as int,
        dim: m['dim']! as int,
        vocabSize: m['vocabSize']! as int,
        ctxLen: m['ctxLen']! as int,
        tokenizerKind: TokenizerKind.values[m['tokenizerKind']! as int],
      );
}

/// Per-generation metrics — the numbers the SipLLM dashboard displays.
class SipllmStats {
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

  double get peakRssMiB => peakRssBytes / (1024 * 1024);

  Map<String, Object> toMap() => {
        'ttftSeconds': ttftSeconds,
        'prefillTokensPerSecond': prefillTokensPerSecond,
        'decodeTokensPerSecond': decodeTokensPerSecond,
        'peakRssBytes': peakRssBytes,
        'weightsResidentBytes': weightsResidentBytes,
        'kvBytes': kvBytes,
        'bytesRead': bytesRead,
        'prefetchHits': prefetchHits,
        'prefetchMisses': prefetchMisses,
        'pinnedLayers': pinnedLayers,
        'nLayers': nLayers,
        'promptTokens': promptTokens,
        'genTokens': genTokens,
        'ctxUsed': ctxUsed,
        'ctxMax': ctxMax,
      };

  static SipllmStats fromMap(Map<String, Object?> m) => SipllmStats(
        ttftSeconds: (m['ttftSeconds']! as num).toDouble(),
        prefillTokensPerSecond:
            (m['prefillTokensPerSecond']! as num).toDouble(),
        decodeTokensPerSecond: (m['decodeTokensPerSecond']! as num).toDouble(),
        peakRssBytes: m['peakRssBytes']! as int,
        weightsResidentBytes: m['weightsResidentBytes']! as int,
        kvBytes: m['kvBytes']! as int,
        bytesRead: m['bytesRead']! as int,
        prefetchHits: m['prefetchHits']! as int,
        prefetchMisses: m['prefetchMisses']! as int,
        pinnedLayers: m['pinnedLayers']! as int,
        nLayers: m['nLayers']! as int,
        promptTokens: m['promptTokens']! as int,
        genTokens: m['genTokens']! as int,
        ctxUsed: m['ctxUsed']! as int,
        ctxMax: m['ctxMax']! as int,
      );
}

/// Emitted for each streamed piece during generation.
class SipllmToken {
  const SipllmToken(this.piece, this.id);
  final String piece;
  final int id;
}
