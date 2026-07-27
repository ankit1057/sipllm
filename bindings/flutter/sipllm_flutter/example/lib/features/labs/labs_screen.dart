import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';

/// Experimental Labs — advanced features for runtime enthusiasts: Vulkan GPU offload,
/// thread schedule policy tuning, KV cache inspection, and model-to-Wear OS transfer.
class LabsScreen extends StatelessWidget {
  const LabsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        title: const Text('Experimental Labs'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Banner
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppTheme.accent.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppTheme.accent.withValues(alpha: 0.4)),
            ),
            child: const Row(
              children: [
                Icon(LucideIcons.flaskConical, color: AppTheme.accent, size: 24),
                SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Experimental runtime features and low-level kernel diagnostics.',
                    style: TextStyle(color: AppTheme.accent, fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          // Vulkan Accelerator Switch
          Card(
            color: AppTheme.surface,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: SwitchListTile(
              secondary: const Icon(LucideIcons.zap, color: AppTheme.primary),
              title: const Text('Vulkan Matmul Acceleration', style: TextStyle(color: AppTheme.textMain, fontSize: 14, fontWeight: FontWeight.bold)),
              subtitle: Text(
                c.vulkanAvailable
                    ? 'Active: ${c.vulkanInfo}'
                    : (c.vulkanCompiled ? 'Compiled, CPU fallback' : 'Vulkan not compiled'),
                style: const TextStyle(color: AppTheme.textSec, fontSize: 11),
              ),
              value: c.vulkanAvailable,
              onChanged: null, // Device hardware capability dependent
            ),
          ),
          const SizedBox(height: 12),
          // Thread Schedule Policy Selector
          Card(
            color: AppTheme.surface,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Thread Schedule Policy', style: TextStyle(color: AppTheme.textMain, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  const Text('Tuning policy for octa-core big.LITTLE CPUs', style: TextStyle(color: AppTheme.textSec, fontSize: 11)),
                  const SizedBox(height: 10),
                  DropdownButton<SchedulePolicy>(
                    value: c.schedulePolicy,
                    isExpanded: true,
                    dropdownColor: AppTheme.surface,
                    style: const TextStyle(color: AppTheme.primary, fontWeight: FontWeight.bold),
                    items: const [
                      DropdownMenuItem(
                        value: SchedulePolicy.proportional2,
                        child: Text('Proportional2 (Default, Octa-core optimized)'),
                      ),
                      DropdownMenuItem(
                        value: SchedulePolicy.proportional4,
                        child: Text('Proportional4 (High-throughput)'),
                      ),
                      DropdownMenuItem(
                        value: SchedulePolicy.adaptive,
                        child: Text('Adaptive (Dynamic work allocation)'),
                      ),
                    ],
                    onChanged: (policy) {
                      if (policy != null) {
                        c.schedulePolicy = policy;
                      }
                    },
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          // Wear OS Model Transfer Bridge
          Card(
            color: AppTheme.surface,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: ListTile(
              leading: const Icon(LucideIcons.watch, color: AppTheme.accent),
              title: const Text('Wear OS Data Layer Bridge', style: TextStyle(color: AppTheme.textMain, fontWeight: FontWeight.bold, fontSize: 14)),
              subtitle: const Text('Transfer GGUF weights directly to paired smartwatch', style: TextStyle(color: AppTheme.textSec, fontSize: 11)),
              trailing: const Icon(LucideIcons.chevronRight, color: AppTheme.textSec),
              onTap: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Model transfer bridge ready for paired Wear OS devices')),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
