import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/models/benchmark.dart';
import '../../core/theme/app_theme.dart';

/// The North Star scorecard screen.
///
/// Runs an on-device benchmark against the loaded model and renders the SipLLM
/// North Star metrics (TTFT, prefill/decode throughput, memory residency,
/// disk-streaming expansion factor, context usage) as a scorecard, keeps a
/// compact run history, and exports the latest result as JSON, Markdown, or via
/// the platform share sheet.
class BenchmarkScreen extends StatelessWidget {
  const BenchmarkScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<SipllmController>();
    final results = controller.benchmarks;
    final latest = results.isNotEmpty ? results.first : null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmark'),
        backgroundColor: AppTheme.background,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _RunControls(controller: controller),
          const SizedBox(height: 20),
          if (latest == null)
            _EmptyState(hasModel: controller.isModelLoaded)
          else ...[
            _Scorecard(result: latest),
            const SizedBox(height: 16),
            _ExportBar(result: latest),
            if (results.length > 1) ...[
              const SizedBox(height: 24),
              const _SectionLabel('History'),
              const SizedBox(height: 8),
              ...results
                  .skip(1)
                  .map((r) => _HistoryTile(result: r)),
            ],
          ],
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Run controls
// ---------------------------------------------------------------------------

class _RunControls extends StatelessWidget {
  const _RunControls({required this.controller});

  final SipllmController controller;

  @override
  Widget build(BuildContext context) {
    final busy = controller.benchmarking;
    final canRun = controller.isModelLoaded &&
        !controller.benchmarking &&
        !controller.generating;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ElevatedButton.icon(
          onPressed: canRun
              ? () => context.read<SipllmController>().runBenchmark()
              : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: AppTheme.primary,
            foregroundColor: Colors.black,
            disabledBackgroundColor: AppTheme.surface,
            disabledForegroundColor: AppTheme.textSec,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          icon: busy
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor:
                        AlwaysStoppedAnimation<Color>(AppTheme.textSec),
                  ),
                )
              : const Icon(LucideIcons.gauge),
          label: Text(busy ? 'Running…' : 'Run benchmark'),
        ),
        if (!controller.isModelLoaded)
          const Padding(
            padding: EdgeInsets.only(top: 8),
            child: Text(
              'Load a model first',
              textAlign: TextAlign.center,
              style: TextStyle(color: AppTheme.textSec, fontSize: 12),
            ),
          ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.hasModel});

  final bool hasModel;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          const Icon(LucideIcons.activity, size: 40, color: AppTheme.textSec),
          const SizedBox(height: 12),
          Text(
            hasModel
                ? 'No benchmark yet.\nRun one to see the North Star scorecard.'
                : 'Load a model, then run a benchmark.',
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppTheme.textSec),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Scorecard
// ---------------------------------------------------------------------------

class _Scorecard extends StatelessWidget {
  const _Scorecard({required this.result});

  final BenchmarkResult result;

  @override
  Widget build(BuildContext context) {
    final s = result.stats;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primary.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(LucideIcons.gauge, size: 18, color: AppTheme.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '${result.modelName} · ${result.quant}',
                  style: const TextStyle(
                    color: AppTheme.textMain,
                    fontWeight: FontWeight.bold,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          const Text(
            'SipLLM North Star (on-device)',
            style: TextStyle(color: AppTheme.textSec, fontSize: 12),
          ),
          const Divider(height: 24, color: Colors.white12),
          _Metric('TTFT', '${s.ttftSeconds.toStringAsFixed(3)} s'),
          _Metric('Prefill tok/s', s.prefillTokensPerSecond.toStringAsFixed(2)),
          _Metric(
            'Decode tok/s',
            s.decodeTokensPerSecond.toStringAsFixed(2),
            highlight: true,
          ),
          _Metric('Peak RSS', _mib(s.peakRssBytes)),
          _Metric('Resident weights', _mib(s.weightsResidentBytes)),
          _Metric('KV cache', _mib(s.kvBytes)),
          _Metric('Disk streamed', _mib(s.bytesRead)),
          _Metric('Pinned layers', '${s.pinnedLayers} / ${s.nLayers}'),
          _Metric('Context used', '${s.ctxUsed} / ${s.ctxMax}'),
          _Metric(
            'Expansion factor',
            '${result.expansionFactor.toStringAsFixed(1)}x',
          ),
          _Metric('Prompt tokens', '${s.promptTokens}'),
          _Metric('Generated tokens', '${s.genTokens}'),
          const Divider(height: 24, color: Colors.white12),
          _ContextBlock(result: result),
        ],
      ),
    );
  }
}

class _ContextBlock extends StatelessWidget {
  const _ContextBlock({required this.result});

  final BenchmarkResult result;

  @override
  Widget build(BuildContext context) {
    final budget = result.ramBudgetBytes == 0
        ? 'unlimited (stream)'
        : _mib(result.ramBudgetBytes);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _ContextLine(
          LucideIcons.smartphone,
          '${result.deviceModel} · ${result.arch} · ${result.cores} cores',
        ),
        _ContextLine(
          LucideIcons.settings2,
          'threads=${result.threads} · schedule=${result.schedulePolicy} · '
          'ram-budget=$budget · vulkan=${result.vulkan}',
        ),
        _ContextLine(
          LucideIcons.cpu,
          'sipllm ${result.engineVersion} · ${_timestamp(result.timestamp)}',
        ),
      ],
    );
  }
}

class _ContextLine extends StatelessWidget {
  const _ContextLine(this.icon, this.text);

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 14, color: AppTheme.textSec),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(color: AppTheme.textSec, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric(this.label, this.value, {this.highlight = false});

  final String label;
  final String value;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                color: AppTheme.textSec,
                fontSize: 13,
              ),
            ),
          ),
          Text(
            value,
            style: TextStyle(
              color: highlight ? AppTheme.primary : AppTheme.textMain,
              fontSize: 13,
              fontWeight: highlight ? FontWeight.bold : FontWeight.w500,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Export bar
// ---------------------------------------------------------------------------

class _ExportBar extends StatelessWidget {
  const _ExportBar({required this.result});

  final BenchmarkResult result;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _ExportButton(
          icon: LucideIcons.braces,
          label: 'Copy JSON',
          onTap: () => _copy(context, result.toJson(), 'JSON copied'),
        ),
        _ExportButton(
          icon: LucideIcons.fileText,
          label: 'Copy Markdown',
          onTap: () => _copy(context, result.toMarkdown(), 'Markdown copied'),
        ),
        _ExportButton(
          icon: LucideIcons.share2,
          label: 'Share',
          onTap: () => _share(context, result.toMarkdown()),
        ),
      ],
    );
  }

  Future<void> _copy(
    BuildContext context,
    String data,
    String message,
  ) async {
    await Clipboard.setData(ClipboardData(text: data));
    if (!context.mounted) return;
    _snack(context, message);
  }

  Future<void> _share(BuildContext context, String data) async {
    try {
      await Share.share(data, subject: 'SipLLM benchmark');
    } catch (_) {
      if (!context.mounted) return;
      await Clipboard.setData(ClipboardData(text: data));
      if (!context.mounted) return;
      _snack(context, 'Sharing unavailable — copied to clipboard');
    }
  }
}

class _ExportButton extends StatelessWidget {
  const _ExportButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: onTap,
      style: OutlinedButton.styleFrom(
        foregroundColor: AppTheme.textMain,
        side: BorderSide(color: Colors.white.withValues(alpha: 0.12)),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
      ),
      icon: Icon(icon, size: 16),
      label: Text(label),
    );
  }
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

class _HistoryTile extends StatelessWidget {
  const _HistoryTile({required this.result});

  final BenchmarkResult result;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const Icon(LucideIcons.history, size: 16, color: AppTheme.textSec),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              '${result.modelName} · '
              '${result.stats.decodeTokensPerSecond.toStringAsFixed(1)} tok/s · '
              '${_mib(result.stats.peakRssBytes)} · '
              '${_timestamp(result.timestamp)}',
              style: const TextStyle(
                color: AppTheme.textSec,
                fontSize: 12,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: const TextStyle(
        color: AppTheme.textMain,
        fontWeight: FontWeight.bold,
        fontSize: 15,
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

String _mib(int bytes) => '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';

String _timestamp(DateTime t) {
  String two(int v) => v.toString().padLeft(2, '0');
  return '${t.year}-${two(t.month)}-${two(t.day)} '
      '${two(t.hour)}:${two(t.minute)}';
}

void _snack(BuildContext context, String message) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Text(message),
      backgroundColor: AppTheme.surface,
      behavior: SnackBarBehavior.floating,
    ),
  );
}
