import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';
import 'widgets/model_tile.dart';

/// The Ollama-style model manager: browse the catalog, download / import GGUFs,
/// verify or delete them, push one to a paired watch, and load the active model
/// into the inference runtime. A thin `Consumer` over [SipllmController].
///
/// The models folder is re-read when this screen opens, when the app comes
/// back to the foreground, on pull-to-refresh and from the refresh button, so
/// a GGUF copied in while the app runs shows up without a restart.
class ModelsScreen extends StatefulWidget {
  const ModelsScreen({super.key});

  @override
  State<ModelsScreen> createState() => _ModelsScreenState();
}

class _ModelsScreenState extends State<ModelsScreen> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // After the first frame: a rescan notifies listeners, which must not
    // happen while this tree is being built.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) context.read<SipllmController>().rescanModels();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && mounted) {
      context.read<SipllmController>().rescanModels();
    }
  }

  Future<void> _import(BuildContext context) async {
    final controller = context.read<SipllmController>();
    final messenger = ScaffoldMessenger.of(context);
    try {
      final message = await controller.importModel();
      // null: the picker was cancelled.
      if (message != null) messenger.showSnackBar(SnackBar(content: Text(message)));
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Import failed: $e')));
    }
  }

  Future<void> _refresh(BuildContext context) async {
    final controller = context.read<SipllmController>();
    final messenger = ScaffoldMessenger.of(context);
    final before = controller.models.length;
    await controller.rescanModels();
    final added = controller.models.length - before;
    messenger.showSnackBar(SnackBar(
      duration: const Duration(seconds: 2),
      content: Text(added > 0
          ? 'Found $added new model${added == 1 ? '' : 's'}'
          : added < 0
              ? 'Models folder re-read: ${-added} removed'
              : 'Models folder re-read: no new models'),
    ));
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
            icon: const Icon(LucideIcons.refreshCw),
            tooltip: 'Rescan models folder',
            onPressed: () => _refresh(context),
          ),
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
                child: RefreshIndicator(
                  onRefresh: () => controller.rescanModels(),
                  child: models.isEmpty
                      // Scrollable, so pull-to-refresh works on an empty list.
                      ? ListView(
                          physics: const AlwaysScrollableScrollPhysics(),
                          children: const [
                            SizedBox(height: 120),
                            _EmptyState(),
                          ],
                        )
                      : ListView.builder(
                          physics: const AlwaysScrollableScrollPhysics(),
                          padding: const EdgeInsets.only(top: 4, bottom: 24),
                          itemCount: models.length,
                          itemBuilder: (context, i) =>
                              ModelTile(model: models[i]),
                        ),
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
