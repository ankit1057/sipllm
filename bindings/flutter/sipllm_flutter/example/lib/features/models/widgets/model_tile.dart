import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../../core/controller/sipllm_controller.dart';
import '../../../core/models/managed_model.dart';
import '../../../core/theme/app_theme.dart';

/// Humanizes a byte count into a compact `B / KB / MB / GB` string.
String humanizeBytes(num bytes) {
  final b = bytes.toDouble();
  if (b < 1024) return '${b.toStringAsFixed(0)} B';
  const units = ['KB', 'MB', 'GB', 'TB'];
  var value = b / 1024;
  var i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return '${value.toStringAsFixed(value >= 100 ? 0 : 1)} ${units[i]}';
}

/// A single row in the Ollama-style model manager: metadata plus a
/// status-driven action area (download / pause / resume / load / retry).
class ModelTile extends StatelessWidget {
  final ManagedModel model;

  const ModelTile({super.key, required this.model});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<SipllmController>();
    final isLoaded = controller.loadedModel?.id == model.id;

    final sizeBytes = model.sizeBytes > 0
        ? model.sizeBytes
        : (model.catalog?.approxBytes ?? 0);

    final subtitleParts = <String>[
      if (model.params.isNotEmpty) model.params,
      model.quant,
      if (sizeBytes > 0) humanizeBytes(sizeBytes),
    ];

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isLoaded
              ? AppTheme.primary.withValues(alpha: 0.5)
              : Colors.white.withValues(alpha: 0.05),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.accent.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(
                  model.imported ? LucideIcons.fileBox : LucideIcons.box,
                  size: 18,
                  color: AppTheme.accent,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Flexible(
                          child: Text(
                            model.name,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: AppTheme.textMain,
                              fontWeight: FontWeight.w600,
                              fontSize: 15,
                            ),
                          ),
                        ),
                        if (isLoaded) ...[
                          const SizedBox(width: 8),
                          const _LoadedChip(),
                        ],
                      ],
                    ),
                    if (subtitleParts.isNotEmpty) ...[
                      const SizedBox(height: 2),
                      Text(
                        subtitleParts.join(' · '),
                        style: const TextStyle(
                          color: AppTheme.textSec,
                          fontSize: 12,
                        ),
                      ),
                    ],
                    if (model.duplicateOf != null) ...[
                      const SizedBox(height: 2),
                      Text(
                        'Second copy of ${model.duplicateOf} (same model, other file name)',
                        style: const TextStyle(color: Colors.amber, fontSize: 11),
                      ),
                    ],
                  ],
                ),
              ),
              _trailing(context, controller),
            ],
          ),
          ..._statusDetail(context),
        ],
      ),
    );
  }

  Widget _trailing(BuildContext context, SipllmController controller) {
    switch (model.status) {
      case ModelStatus.available:
        return _iconButton(
          icon: LucideIcons.download,
          tooltip: 'Download',
          onTap: () => context.read<SipllmController>().download(model),
        );
      case ModelStatus.downloading:
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _iconButton(
              icon: LucideIcons.pause,
              tooltip: 'Pause',
              onTap: () =>
                  context.read<SipllmController>().pauseDownload(model),
            ),
            _iconButton(
              icon: LucideIcons.x,
              tooltip: 'Cancel',
              onTap: () =>
                  context.read<SipllmController>().cancelDownload(model),
            ),
          ],
        );
      case ModelStatus.paused:
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _iconButton(
              icon: LucideIcons.play,
              tooltip: 'Resume',
              color: AppTheme.primary,
              onTap: () =>
                  context.read<SipllmController>().resumeDownload(model),
            ),
            _iconButton(
              icon: LucideIcons.x,
              tooltip: 'Cancel',
              onTap: () =>
                  context.read<SipllmController>().cancelDownload(model),
            ),
          ],
        );
      case ModelStatus.importing:
        return const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8),
          child: SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: AppTheme.accent,
            ),
          ),
        );
      case ModelStatus.installed:
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _loadButton(context, controller),
            _overflowMenu(context),
          ],
        );
      case ModelStatus.failed:
        return _iconButton(
          icon: LucideIcons.refreshCw,
          tooltip: 'Retry',
          color: AppTheme.primary,
          onTap: () => context.read<SipllmController>().download(model),
        );
    }
  }

  Widget _loadButton(BuildContext context, SipllmController controller) {
    final isLoaded = controller.loadedModel?.id == model.id;
    final busy = controller.loading || controller.unloading;
    if (isLoaded) {
      return TextButton.icon(
        onPressed: busy ? null : () => _unload(context),
        icon: const Icon(LucideIcons.power, size: 16),
        label: const Text('Unload'),
        style: TextButton.styleFrom(
          foregroundColor: AppTheme.primary,
          visualDensity: VisualDensity.compact,
        ),
      );
    }
    if (busy) {
      return const Padding(
        padding: EdgeInsets.symmetric(horizontal: 8),
        child: SizedBox(
          width: 18,
          height: 18,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            color: AppTheme.primary,
          ),
        ),
      );
    }
    return _iconButton(
      icon: LucideIcons.play,
      tooltip: 'Load',
      color: AppTheme.primary,
      onTap: () => _load(context),
    );
  }

  // Loading or unloading closes the current model, which stops whatever is
  // using it. Say so first instead of silently ending an agent run or reply.
  Future<bool> _confirmInterrupt(BuildContext context, String action) async {
    final c = context.read<SipllmController>();
    final what = c.agentRunning
        ? 'the agent run'
        : c.benchmarking
            ? 'the benchmark'
            : c.generating
                ? 'the reply being generated'
                : null;
    if (what == null) return true;
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppTheme.surface,
        title: Text('$action now?', style: const TextStyle(color: AppTheme.textMain)),
        content: Text('This stops $what.', style: const TextStyle(color: AppTheme.textSec)),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel', style: TextStyle(color: AppTheme.textSec)),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: Text(action, style: const TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    return ok == true;
  }

  // A failed load used to end as an unhandled Future error: nothing shown.
  Future<void> _load(BuildContext context) async {
    final controller = context.read<SipllmController>();
    final messenger = ScaffoldMessenger.maybeOf(context);
    if (!await _confirmInterrupt(context, 'Load')) return;
    try {
      await controller.loadModel(model);
    } catch (e) {
      messenger?.showSnackBar(
        SnackBar(content: Text('Could not load ${model.name}: $e')),
      );
    }
  }

  Future<void> _unload(BuildContext context) async {
    final controller = context.read<SipllmController>();
    final messenger = ScaffoldMessenger.maybeOf(context);
    if (!await _confirmInterrupt(context, 'Unload')) return;
    try {
      await controller.unloadModel();
      messenger?.showSnackBar(
        SnackBar(content: Text('Unloaded ${model.name}')),
      );
    } catch (e) {
      messenger?.showSnackBar(
        SnackBar(content: Text('Could not unload ${model.name}: $e')),
      );
    }
  }

  Widget _overflowMenu(BuildContext context) {
    final controller = context.read<SipllmController>();
    final canUnload = controller.loadedModel?.id == model.id &&
        !controller.loading &&
        !controller.unloading;
    return PopupMenuButton<String>(
      icon: const Icon(
        LucideIcons.moreVertical,
        size: 18,
        color: AppTheme.textSec,
      ),
      color: AppTheme.surface,
      onSelected: (value) => _onMenu(context, value),
      itemBuilder: (context) => [
        if (canUnload)
          const PopupMenuItem(
            value: 'unload',
            child: _MenuRow(icon: LucideIcons.power, label: 'Unload'),
          ),
        const PopupMenuItem(
          value: 'verify',
          child: _MenuRow(icon: LucideIcons.shieldCheck, label: 'Verify'),
        ),
        const PopupMenuItem(
          value: 'watch',
          child: _MenuRow(icon: LucideIcons.watch, label: 'Send to Watch'),
        ),
        const PopupMenuItem(
          value: 'delete',
          child: _MenuRow(
            icon: LucideIcons.trash2,
            label: 'Delete',
            danger: true,
          ),
        ),
      ],
    );
  }

  Future<void> _onMenu(BuildContext context, String value) async {
    final controller = context.read<SipllmController>();
    final messenger = ScaffoldMessenger.of(context);
    switch (value) {
      case 'unload':
        await _unload(context);
        break;
      case 'verify':
        final ok = await controller.verifyModel(model);
        messenger.showSnackBar(
          SnackBar(content: Text(ok ? 'Integrity OK' : 'Corrupt')),
        );
        break;
      case 'watch':
        final nodes = await controller.wear.connectedNodes();
        if (nodes.isEmpty) {
          messenger.showSnackBar(
            const SnackBar(content: Text('No paired watch found')),
          );
        } else {
          await controller.wear.sendModel(model.localPath);
          messenger.showSnackBar(
            SnackBar(
              content: Text('Sending ${model.name} to ${nodes.first.displayName}'),
            ),
          );
        }
        break;
      case 'delete':
        if (!context.mounted) return;
        final confirmed = await _confirmDelete(context);
        if (confirmed == true) await controller.deleteModel(model);
        break;
    }
  }

  Future<bool?> _confirmDelete(BuildContext context) {
    return showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppTheme.surface,
        title: const Text(
          'Delete model?',
          style: TextStyle(color: AppTheme.textMain),
        ),
        content: Text(
          'This removes "${model.name}" from local storage.',
          style: const TextStyle(color: AppTheme.textSec),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel', style: TextStyle(color: AppTheme.textSec)),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete', style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
  }

  List<Widget> _statusDetail(BuildContext context) {
    switch (model.status) {
      case ModelStatus.downloading:
      case ModelStatus.paused:
        return [
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: model.fraction > 0 ? model.fraction : null,
              minHeight: 6,
              backgroundColor: Colors.white.withValues(alpha: 0.06),
              valueColor: AlwaysStoppedAnimation(
                model.status == ModelStatus.paused
                    ? AppTheme.textSec
                    : AppTheme.primary,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            _progressText(),
            style: const TextStyle(color: AppTheme.textSec, fontSize: 11),
          ),
        ];
      case ModelStatus.failed:
        return [
          const SizedBox(height: 10),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(LucideIcons.alertTriangle,
                  size: 14, color: Colors.redAccent),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  model.error ?? 'Download failed',
                  style: const TextStyle(color: Colors.redAccent, fontSize: 12),
                ),
              ),
            ],
          ),
        ];
      default:
        return const [];
    }
  }

  String _progressText() {
    final prog = model.progress;
    if (prog == null) {
      return model.status == ModelStatus.paused ? 'Paused' : 'Starting…';
    }
    final received = humanizeBytes(prog.received);
    final total = prog.total != null ? humanizeBytes(prog.total!) : '?';
    final rate = '${humanizeBytes(prog.bytesPerSecond)}/s';
    final base = '$received / $total @ $rate, ${prog.activeConnections} conns';
    return model.status == ModelStatus.paused ? 'Paused · $base' : base;
  }

  Widget _iconButton({
    required IconData icon,
    required String tooltip,
    required VoidCallback? onTap,
    Color color = AppTheme.textSec,
  }) {
    return IconButton(
      icon: Icon(icon, size: 18),
      color: color,
      tooltip: tooltip,
      onPressed: onTap,
      splashRadius: 20,
      visualDensity: VisualDensity.compact,
    );
  }
}

class _LoadedChip extends StatelessWidget {
  const _LoadedChip();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: AppTheme.primary.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.primary.withValues(alpha: 0.4)),
      ),
      child: const Text(
        'Loaded',
        style: TextStyle(
          color: AppTheme.primary,
          fontSize: 10,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

class _MenuRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool danger;

  const _MenuRow({required this.icon, required this.label, this.danger = false});

  @override
  Widget build(BuildContext context) {
    final color = danger ? Colors.redAccent : AppTheme.textMain;
    return Row(
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 10),
        Text(label, style: TextStyle(color: color, fontSize: 13)),
      ],
    );
  }
}
