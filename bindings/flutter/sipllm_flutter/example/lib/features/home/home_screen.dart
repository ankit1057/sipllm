import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';
import '../../core/widgets/sipllm_logo.dart';

/// The landing dashboard. A pure, scrollable view over [SipllmController]
/// surfacing device + engine capabilities and a quick model/runtime status.
///
/// While the controller is still initializing, a centered progress indicator
/// is shown instead of the cards.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();

    if (!c.initialized) {
      return const Center(
        child: CircularProgressIndicator(color: AppTheme.primary),
      );
    }

    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 32),
      children: [
        _Header(isWearOs: c.isWearOs),
        const SizedBox(height: 20),
        _DeviceCard(controller: c),
        const SizedBox(height: 14),
        _EngineCard(controller: c),
        const SizedBox(height: 14),
        _StatusCard(controller: c),
      ],
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.isWearOs});

  final bool isWearOs;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const SipLlmLogo(size: 38),
            const SizedBox(width: 12),
            const Text(
              'SipLLM Studio',
              style: TextStyle(
                color: AppTheme.textMain,
                fontSize: 26,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.5,
              ),
            ),
            if (isWearOs) ...[
              const SizedBox(width: 10),
              const _Chip(icon: LucideIcons.watch, label: 'Wear OS'),
            ],
          ],
        ),
        const SizedBox(height: 6),
        Text(
          'Bigger-than-RAM LLMs, on your device',
          style: TextStyle(
            color: AppTheme.textSec,
            fontSize: 14,
            height: 1.3,
          ),
        ),
      ],
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    const tint = AppTheme.accent;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: tint.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: tint.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: tint),
          const SizedBox(width: 5),
          Text(
            label,
            style: TextStyle(
              color: tint,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

/// A titled surface card used for each dashboard section.
class _Card extends StatelessWidget {
  const _Card({
    required this.icon,
    required this.title,
    required this.children,
  });

  final IconData icon;
  final String title;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 18, color: AppTheme.primary),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  color: AppTheme.textMain,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ...children,
        ],
      ),
    );
  }
}

/// A single label / value row inside a card.
class _Row extends StatelessWidget {
  const _Row({
    required this.icon,
    required this.label,
    required this.value,
    this.valueColor,
    this.subtitle,
  });

  final IconData icon;
  final String label;
  final String value;
  final Color? valueColor;
  final String? subtitle;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: AppTheme.textSec),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(color: AppTheme.textSec, fontSize: 13),
            ),
          ),
          const SizedBox(width: 10),
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  value,
                  textAlign: TextAlign.end,
                  style: TextStyle(
                    color: valueColor ?? AppTheme.textMain,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (subtitle != null && subtitle!.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      subtitle!,
                      textAlign: TextAlign.end,
                      style: const TextStyle(
                        color: AppTheme.textSec,
                        fontSize: 11,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DeviceCard extends StatelessWidget {
  const _DeviceCard({required this.controller});

  final SipllmController controller;

  @override
  Widget build(BuildContext context) {
    final p = controller.deviceProfile;
    return _Card(
      icon: LucideIcons.smartphone,
      title: 'Device',
      children: [
        _Row(
          icon: LucideIcons.smartphone,
          label: 'Model',
          value: (p?.model.isNotEmpty ?? false) ? p!.model : 'unknown',
        ),
        _Row(
          icon: LucideIcons.cpu,
          label: 'Architecture',
          value: p == null
              ? 'unknown'
              : '${p.primaryArch.androidAbi} '
                  '(${p.is64Bit ? '64-bit' : '32-bit'})',
        ),
        _Row(
          icon: LucideIcons.cpu,
          label: 'CPU',
          value: p == null ? 'unknown' : '${p.cores} cores',
        ),
        _Row(
          icon: p?.isWearOs ?? false
              ? LucideIcons.watch
              : LucideIcons.smartphone,
          label: 'Android SDK',
          value: p == null ? 'unknown' : 'API ${p.androidSdkInt}',
        ),
      ],
    );
  }
}

class _EngineCard extends StatelessWidget {
  const _EngineCard({required this.controller});

  final SipllmController controller;

  @override
  Widget build(BuildContext context) {
    final c = controller;

    final String gpuLabel;
    final Color gpuColor;
    if (c.vulkanAvailable) {
      gpuLabel = 'active';
      gpuColor = AppTheme.primary;
    } else if (c.vulkanCompiled) {
      gpuLabel = 'compiled, CPU fallback';
      gpuColor = AppTheme.accent;
    } else {
      gpuLabel = 'CPU only';
      gpuColor = AppTheme.textSec;
    }

    return _Card(
      icon: LucideIcons.zap,
      title: 'Engine',
      children: [
        _Row(
          icon: LucideIcons.zap,
          label: 'Version',
          value: c.engineVersion.isNotEmpty ? c.engineVersion : 'unknown',
        ),
        _Row(
          icon: LucideIcons.cpu,
          label: 'Threads',
          value: '${c.hardwareConcurrency}',
        ),
        _Row(
          icon: LucideIcons.zap,
          label: 'GPU',
          value: gpuLabel,
          valueColor: gpuColor,
          subtitle: c.vulkanInfo,
        ),
      ],
    );
  }
}

class _StatusCard extends StatelessWidget {
  const _StatusCard({required this.controller});

  final SipllmController controller;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    final loaded = c.loadedModel;

    final String loadedValue;
    final String? loadedSubtitle;
    if (loaded == null) {
      loadedValue = 'none';
      loadedSubtitle = null;
    } else {
      loadedValue = loaded.name;
      final parts = <String>[
        if (loaded.quant.isNotEmpty) loaded.quant,
        if (loaded.params.isNotEmpty) loaded.params,
      ];
      loadedSubtitle = parts.join(' \u00B7 ');
    }

    final ram = c.ramBudgetMiB == 0
        ? 'unlimited (stream)'
        : '${c.ramBudgetMiB} MB';

    return _Card(
      icon: LucideIcons.database,
      title: 'Status',
      children: [
        _Row(
          icon: LucideIcons.package,
          label: 'Installed models',
          value: '${c.installedCount}',
        ),
        _Row(
          icon: LucideIcons.hardDrive,
          label: 'Storage used',
          value: _humanizeBytes(c.totalStorageBytes),
        ),
        _Row(
          icon: c.isModelLoaded ? LucideIcons.checkCircle : LucideIcons.circle,
          label: 'Loaded model',
          value: loadedValue,
          valueColor: c.isModelLoaded ? AppTheme.primary : AppTheme.textSec,
          subtitle: loadedSubtitle,
        ),
        _Row(
          icon: LucideIcons.gauge,
          label: 'RAM budget',
          value: ram,
        ),
      ],
    );
  }
}

/// Humanizes a byte count into a compact B / KB / MB / GB string.
String _humanizeBytes(int bytes) {
  if (bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  var value = bytes.toDouble();
  var unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  final str = value >= 100 || value == value.roundToDouble()
      ? value.toStringAsFixed(0)
      : value.toStringAsFixed(1);
  return '$str ${units[unit]}';
}
