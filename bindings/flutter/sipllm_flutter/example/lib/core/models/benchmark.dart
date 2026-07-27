import 'dart:convert';

import 'package:sipllm_flutter/sipllm_flutter.dart';

/// A single on-device benchmark run: the SipLLM North Star metrics plus the
/// device/config context needed to compare against the desktop baseline.
class BenchmarkResult {
  BenchmarkResult({
    required this.timestamp,
    required this.modelName,
    required this.quant,
    required this.deviceModel,
    required this.arch,
    required this.cores,
    required this.threads,
    required this.schedulePolicy,
    required this.ramBudgetBytes,
    required this.vulkan,
    required this.engineVersion,
    required this.stats,
  });

  final DateTime timestamp;
  final String modelName;
  final String quant;
  final String deviceModel;
  final String arch;
  final int cores;
  final int threads;
  final String schedulePolicy;
  final int ramBudgetBytes;
  final String vulkan;
  final String engineVersion;
  final SipllmStats stats;

  double get expansionFactor =>
      stats.peakRssBytes > 0 ? stats.bytesRead / stats.peakRssBytes : 0;

  Map<String, Object?> toMap() => {
        'timestamp': timestamp.toIso8601String(),
        'model': modelName,
        'quant': quant,
        'device': deviceModel,
        'arch': arch,
        'cores': cores,
        'threads': threads,
        'schedule_policy': schedulePolicy,
        'ram_budget_bytes': ramBudgetBytes,
        'vulkan': vulkan,
        'engine_version': engineVersion,
        'ttft_s': stats.ttftSeconds,
        'prefill_tok_s': stats.prefillTokensPerSecond,
        'decode_tok_s': stats.decodeTokensPerSecond,
        'peak_rss_bytes': stats.peakRssBytes,
        'weights_resident_bytes': stats.weightsResidentBytes,
        'kv_bytes': stats.kvBytes,
        'bytes_read': stats.bytesRead,
        'pinned_layers': stats.pinnedLayers,
        'n_layers': stats.nLayers,
        'prompt_tokens': stats.promptTokens,
        'gen_tokens': stats.genTokens,
        'ctx_used': stats.ctxUsed,
        'ctx_max': stats.ctxMax,
      };

  String toJson() => const JsonEncoder.withIndent('  ').convert(toMap());

  /// A North Star scorecard block in the exact spirit of CHANGELOG.md.
  String toMarkdown() {
    String mib(int b) => '${(b / (1024 * 1024)).toStringAsFixed(1)} MB';
    final budget =
        ramBudgetBytes == 0 ? 'unlimited (stream)' : mib(ramBudgetBytes);
    return '''
## SipLLM mobile benchmark — $modelName ($quant)

- **Device:** $deviceModel · $arch · $cores cores
- **Config:** threads=$threads · schedule=$schedulePolicy · ram-budget=$budget · vulkan=$vulkan
- **Engine:** sipllm $engineVersion · measured ${timestamp.toIso8601String()}

```
SipLLM North Star (on-device)
=============================
TTFT:                ${stats.ttftSeconds.toStringAsFixed(3)} s
Prefill tok/s:       ${stats.prefillTokensPerSecond.toStringAsFixed(2)}
Decode tok/s:        ${stats.decodeTokensPerSecond.toStringAsFixed(2)}
Peak RSS:            ${mib(stats.peakRssBytes)}
Resident Weights:    ${mib(stats.weightsResidentBytes)}
KV cache:            ${mib(stats.kvBytes)}
Disk streamed:       ${mib(stats.bytesRead)}
Pinned layers:       ${stats.pinnedLayers} / ${stats.nLayers}
Context used:        ${stats.ctxUsed} / ${stats.ctxMax}
Expansion factor:    ${expansionFactor.toStringAsFixed(1)}x  (disk / peak RSS)
Prompt tokens:       ${stats.promptTokens}
Generated tokens:    ${stats.genTokens}
```
''';
  }
}
