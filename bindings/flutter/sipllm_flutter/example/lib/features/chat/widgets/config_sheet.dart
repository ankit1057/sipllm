import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import '../../../core/controller/sipllm_controller.dart';
import '../../../core/theme/app_theme.dart';

/// Draggable bottom sheet that edits the controller's generation config.
///
/// Fields are plain public fields on [SipllmController]; the sheet mutates them
/// directly and calls [setState] for a local rebuild. Sampling values
/// (temperature/topP/maxTokens) and RAG take effect on the next `send`, while
/// ram-budget/threads/scheduler/fastQuant apply on the next model load.
class ConfigSheet extends StatefulWidget {
  final SipllmController controller;

  const ConfigSheet({super.key, required this.controller});

  @override
  State<ConfigSheet> createState() => _ConfigSheetState();
}

class _ConfigSheetState extends State<ConfigSheet> {
  SipllmController get c => widget.controller;

  @override
  Widget build(BuildContext context) {
    final cores = c.deviceProfile?.cores ?? c.hardwareConcurrency;
    final maxThreads = cores > 0 ? cores : 16;

    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.7,
      minChildSize: 0.4,
      maxChildSize: 0.95,
      builder: (context, scrollController) {
        return Container(
          decoration: const BoxDecoration(
            color: AppTheme.background,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: ListView(
            controller: scrollController,
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: AppTheme.textSec.withValues(alpha: 0.4),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              _header(LucideIcons.sliders, 'Generation'),
              _slider(
                label: 'Temperature',
                value: c.temperature,
                min: 0,
                max: 1.5,
                display: c.temperature.toStringAsFixed(2),
                onChanged: (v) => setState(() => c.temperature = v),
              ),
              _slider(
                label: 'Top-P',
                value: c.topP,
                min: 0,
                max: 1,
                display: c.topP.toStringAsFixed(2),
                onChanged: (v) => setState(() => c.topP = v),
              ),
              _slider(
                label: 'Max tokens',
                value: c.maxTokens.toDouble(),
                min: 16,
                max: 1024,
                divisions: (1024 - 16) ~/ 16,
                display: '${c.maxTokens}',
                onChanged: (v) => setState(() => c.maxTokens = v.round()),
              ),
              const SizedBox(height: 8),
              _header(LucideIcons.cpu, 'Runtime (applies on next model load)'),
              _slider(
                label: 'RAM budget',
                value: c.ramBudgetMiB.toDouble(),
                min: 0,
                max: 4096,
                divisions: 4096 ~/ 64,
                display: c.ramBudgetMiB == 0
                    ? 'unlimited / stream'
                    : '${c.ramBudgetMiB} MiB',
                onChanged: (v) => setState(() => c.ramBudgetMiB = v.round()),
              ),
              _slider(
                label: 'Threads',
                value: c.threads.toDouble().clamp(0, maxThreads.toDouble()),
                min: 0,
                max: maxThreads.toDouble(),
                divisions: maxThreads,
                display: c.threads == 0 ? 'auto' : '${c.threads}',
                onChanged: (v) => setState(() => c.threads = v.round()),
              ),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                activeThumbColor: AppTheme.primary,
                title: const Text('Fast quantization',
                    style: TextStyle(color: AppTheme.textMain)),
                subtitle: const Text('Faster dequant kernels',
                    style: TextStyle(color: AppTheme.textSec, fontSize: 12)),
                value: c.fastQuant,
                onChanged: (v) => setState(() => c.fastQuant = v),
              ),
              _scheduleDropdown(),
              const SizedBox(height: 8),
              _header(LucideIcons.database, 'Memory (RAG)'),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                activeThumbColor: AppTheme.accent,
                title: const Text('Use retrieval memory',
                    style: TextStyle(color: AppTheme.textMain)),
                subtitle: Text('${c.memorySize} item(s) stored',
                    style: const TextStyle(color: AppTheme.textSec, fontSize: 12)),
                value: c.useRag,
                onChanged: (v) => setState(() => c.useRag = v),
              ),
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: c.memorySize == 0
                      ? null
                      : () {
                          c.clearMemory();
                          setState(() {});
                        },
                  icon: const Icon(LucideIcons.trash2, size: 16),
                  label: const Text('Clear memory'),
                  style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _header(IconData icon, String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 4),
      child: Row(
        children: [
          Icon(icon, size: 16, color: AppTheme.primary),
          const SizedBox(width: 8),
          Text(
            title,
            style: const TextStyle(
              color: AppTheme.textSec,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _slider({
    required String label,
    required double value,
    required double min,
    required double max,
    required String display,
    required ValueChanged<double> onChanged,
    int? divisions,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(color: AppTheme.textMain)),
            Text(display,
                style: const TextStyle(color: AppTheme.primary, fontSize: 13)),
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: AppTheme.primary,
            thumbColor: AppTheme.primary,
            inactiveTrackColor: AppTheme.surface,
          ),
          child: Slider(
            value: value.clamp(min, max),
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }

  Widget _scheduleDropdown() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Text('Schedule policy',
              style: TextStyle(color: AppTheme.textMain)),
          DropdownButton<SchedulePolicy>(
            value: c.schedulePolicy,
            dropdownColor: AppTheme.surface,
            style: const TextStyle(color: AppTheme.textMain),
            underline: const SizedBox.shrink(),
            items: [
              for (final p in SchedulePolicy.values)
                DropdownMenuItem(value: p, child: Text(p.name)),
            ],
            onChanged: (v) {
              if (v != null) setState(() => c.schedulePolicy = v);
            },
          ),
        ],
      ),
    );
  }
}
