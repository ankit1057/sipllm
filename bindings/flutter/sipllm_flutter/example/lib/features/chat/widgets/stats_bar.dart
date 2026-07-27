import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import '../../../core/theme/app_theme.dart';

/// A compact, horizontally-scrolling strip of live inference telemetry
/// (decode throughput, TTFT, peak RSS, pinned layers, context, threads).
///
/// Rendered while a generation is running or once the most recent generation
/// has produced [SipllmStats]. Missing numbers are simply omitted.
class StatsBar extends StatelessWidget {
  final SipllmStats? stats;
  final bool generating;
  final int threads;

  const StatsBar({
    super.key,
    required this.stats,
    required this.generating,
    required this.threads,
  });

  @override
  Widget build(BuildContext context) {
    final s = stats;
    final chips = <Widget>[];

    if (s != null) {
      chips.add(_chip(
        LucideIcons.zap,
        '${s.decodeTokensPerSecond.toStringAsFixed(1)} tok/s',
        AppTheme.primary,
      ));
      chips.add(_chip(
        LucideIcons.timer,
        'TTFT ${_secs(s.ttftSeconds)}',
        AppTheme.accent,
      ));
      chips.add(_chip(
        LucideIcons.memoryStick,
        '${s.peakRssMiB} MB',
        AppTheme.textSec,
      ));
      chips.add(_chip(
        LucideIcons.layers,
        '${s.pinnedLayers}/${s.nLayers} pinned',
        AppTheme.textSec,
      ));
      chips.add(_chip(
        LucideIcons.scrollText,
        'ctx ${s.ctxUsed}/${s.ctxMax}',
        AppTheme.textSec,
      ));
    } else if (generating) {
      chips.add(_chip(LucideIcons.loader, 'generating…', AppTheme.primary));
    }

    chips.add(_chip(
      LucideIcons.cpu,
      threads == 0 ? 'auto threads' : '$threads threads',
      AppTheme.textSec,
    ));

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: AppTheme.surface,
        border: Border(
          bottom: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
        ),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: [
            for (final c in chips) Padding(padding: const EdgeInsets.only(right: 8), child: c),
          ],
        ),
      ),
    );
  }

  String _secs(double v) => v <= 0 ? '—' : '${(v * 1000).toStringAsFixed(0)} ms';

  Widget _chip(IconData icon, String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: AppTheme.textMain,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
