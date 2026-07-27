import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';
import 'widgets/model_tile.dart';

/// The Ollama-style model manager: browse the catalog, download / import GGUFs,
/// verify or delete them, push one to a paired watch, and load the active model
/// into the inference runtime. A thin `Consumer` over [SipllmController].
class ModelsScreen extends StatelessWidget {
  const ModelsScreen({super.key});

  Future<void> _import(BuildContext context) async {
    final controller = context.read<SipllmController>();
    final messenger = ScaffoldMessenger.of(context);
    final error = await controller.importModel();
    messenger.showSnackBar(
      SnackBar(content: Text(error ?? 'Imported')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        title: const Text('Models'),
        actions: [
          IconButton(
            icon: const Icon(LucideIcons.filePlus),
            tooltip: 'Import GGUF',
            onPressed: () => _import(context),
          ),
        ],
      ),
      body: Consumer<SipllmController>(
        builder: (context, controller, _) {
          final models = controller.models;
          return Column(
            children: [
              _StorageHeader(
                count: controller.installedCount,
                totalBytes: controller.totalStorageBytes,
              ),
              Expanded(
                child: models.isEmpty
                    ? const _EmptyState()
                    : ListView.builder(
                        padding: const EdgeInsets.only(top: 4, bottom: 24),
                        itemCount: models.length,
                        itemBuilder: (context, i) =>
                            ModelTile(model: models[i]),
                      ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _StorageHeader extends StatelessWidget {
  final int count;
  final int totalBytes;

  const _StorageHeader({required this.count, required this.totalBytes});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Row(
        children: [
          const Icon(LucideIcons.hardDrive, size: 16, color: AppTheme.textSec),
          const SizedBox(width: 10),
          Text(
            '$count ${count == 1 ? 'model' : 'models'} · ${humanizeBytes(totalBytes)} used',
            style: const TextStyle(color: AppTheme.textSec, fontSize: 13),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            LucideIcons.boxes,
            size: 48,
            color: AppTheme.textSec.withValues(alpha: 0.5),
          ),
          const SizedBox(height: 16),
          const Text(
            'No models yet',
            style: TextStyle(
              color: AppTheme.textMain,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Download from the catalog or import a .gguf',
            style: TextStyle(color: AppTheme.textSec, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
