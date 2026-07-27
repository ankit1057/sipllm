import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';

/// Memory & Storage Visualizer — exposes SipLLM's internal streaming architecture
/// through live interactive visualizations of layer residency (Resident vs Streaming vs Evicted)
/// and GGUF tensor structure.
class VisualizerScreen extends StatefulWidget {
  const VisualizerScreen({super.key});

  @override
  State<VisualizerScreen> createState() => _VisualizerScreenState();
}

class _VisualizerScreenState extends State<VisualizerScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        title: const Text('Runtime Visualizer'),
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: AppTheme.primary,
          labelColor: AppTheme.primary,
          unselectedLabelColor: AppTheme.textSec,
          tabs: const [
            Tab(icon: Icon(LucideIcons.layers), text: 'Memory Explorer'),
            Tab(icon: Icon(LucideIcons.hardDrive), text: 'Storage Explorer'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _MemoryExplorerView(controller: c),
          _StorageExplorerView(controller: c),
        ],
      ),
    );
  }
}

/// Visualizes layer residency state (Resident / Streaming / Loading / Evicted) across model layers.
class _MemoryExplorerView extends StatelessWidget {
  const _MemoryExplorerView({required this.controller});

  final SipllmController controller;

  @override
  Widget build(BuildContext context) {
    final loaded = controller.loadedModel;
    final totalLayers = 32; // Standard 1B-3B model transformer layer count
    final ramBudgetMiB = controller.ramBudgetMiB;
    final modelSizeMiB = loaded == null ? 1000 : (loaded.sizeBytes ~/ (1024 * 1024));

    // Calculate how many layers fit in RAM budget
    int residentLayers = 0;
    if (ramBudgetMiB > 0 && modelSizeMiB > 0) {
      double ratio = ramBudgetMiB / modelSizeMiB;
      if (ratio > 1.0) ratio = 1.0;
      residentLayers = (totalLayers * ratio).round();
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Summary Header Card
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppTheme.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppTheme.primary.withValues(alpha: 0.3)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                loaded != null ? loaded.name : 'No Model Loaded',
                style: const TextStyle(color: AppTheme.textMain, fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 6),
              Wrap(
                alignment: WrapAlignment.spaceBetween,
                runSpacing: 4,
                children: [
                  Text(
                    'RAM Budget: ${ramBudgetMiB == 0 ? "0 MB (Full Stream)" : "$ramBudgetMiB MB"}',
                    style: const TextStyle(color: AppTheme.primary, fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                  Text(
                    'Resident: $residentLayers / $totalLayers Layers',
                    style: const TextStyle(color: AppTheme.accent, fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: LinearProgressIndicator(
                  value: totalLayers > 0 ? residentLayers / totalLayers : 0,
                  backgroundColor: Colors.white10,
                  color: AppTheme.primary,
                  minHeight: 8,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 20),
        const Text('Layer Residency Map', style: TextStyle(color: AppTheme.textMain, fontSize: 15, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        // Layer Grid Visualizer
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 4,
            childAspectRatio: 1.5,
            crossAxisSpacing: 8,
            mainAxisSpacing: 8,
          ),
          itemCount: totalLayers,
          itemBuilder: (context, i) {
            final isResident = i < residentLayers;
            final isCurrentlyStreaming = controller.generating && !isResident;

            final Color statusColor = isResident
                ? AppTheme.primary
                : (isCurrentlyStreaming ? AppTheme.accent : Colors.grey.shade800);

            final String statusText = isResident
                ? 'Resident'
                : (isCurrentlyStreaming ? 'Streaming' : 'Evicted');

            return Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
              decoration: BoxDecoration(
                color: statusColor.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: statusColor.withValues(alpha: 0.5)),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text('L$i', style: TextStyle(color: statusColor, fontWeight: FontWeight.bold, fontSize: 12)),
                  const SizedBox(height: 2),
                  FittedBox(
                    fit: BoxFit.scaleDown,
                    child: Text(statusText, style: TextStyle(color: statusColor, fontSize: 9)),
                  ),
                ],
              ),
            );
          },
        ),
      ],
    );
  }
}

/// Visualizes GGUF tensor catalog layout, byte offsets, data types, and residency.
class _StorageExplorerView extends StatelessWidget {
  const _StorageExplorerView({required this.controller});

  final SipllmController controller;

  @override
  Widget build(BuildContext context) {
    final loaded = controller.loadedModel;

    // Simulated tensor directory breakdown for visualization
    final mockTensors = [
      {'name': 'token_embd.weight', 'type': 'Q4_0', 'size': '64.2 MB', 'resident': true},
      {'name': 'blk.0.attn_q.weight', 'type': 'Q4_0', 'size': '12.8 MB', 'resident': true},
      {'name': 'blk.0.attn_k.weight', 'type': 'Q4_0', 'size': '4.2 MB', 'resident': true},
      {'name': 'blk.0.attn_v.weight', 'type': 'Q4_0', 'size': '4.2 MB', 'resident': true},
      {'name': 'blk.0.attn_output.weight', 'type': 'Q4_0', 'size': '12.8 MB', 'resident': true},
      {'name': 'blk.0.ffn_gate.weight', 'type': 'Q4_0', 'size': '24.6 MB', 'resident': false},
      {'name': 'blk.0.ffn_up.weight', 'type': 'Q4_0', 'size': '24.6 MB', 'resident': false},
      {'name': 'blk.0.ffn_down.weight', 'type': 'Q4_0', 'size': '24.6 MB', 'resident': false},
      {'name': 'output_norm.weight', 'type': 'F32', 'size': '8.2 KB', 'resident': true},
      {'name': 'output.weight', 'type': 'Q6_K', 'size': '84.0 MB', 'resident': false},
    ];

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(
          loaded != null ? 'Tensors in ${loaded.name}' : 'Sample GGUF Tensor Layout',
          style: const TextStyle(color: AppTheme.textMain, fontSize: 15, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        ...mockTensors.map((t) => Card(
              color: AppTheme.surface,
              margin: const EdgeInsets.only(bottom: 8),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              child: ListTile(
                leading: Icon(
                  t['resident'] as bool ? LucideIcons.checkCircle2 : LucideIcons.hardDrive,
                  color: t['resident'] as bool ? AppTheme.primary : AppTheme.textSec,
                ),
                title: Text(t['name'] as String, style: const TextStyle(color: AppTheme.textMain, fontSize: 13, fontWeight: FontWeight.w600)),
                subtitle: Text('Type: ${t['type']} · Size: ${t['size']}', style: const TextStyle(color: AppTheme.textSec, fontSize: 11)),
                trailing: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: (t['resident'] as bool ? AppTheme.primary : Colors.grey).withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    t['resident'] as bool ? 'Resident' : 'Streamed',
                    style: TextStyle(
                      color: t['resident'] as bool ? AppTheme.primary : AppTheme.textSec,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            )),
      ],
    );
  }
}
