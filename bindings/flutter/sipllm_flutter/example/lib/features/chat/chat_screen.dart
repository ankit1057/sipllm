import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';
import 'widgets/chat_bubble.dart';
import 'widgets/chat_input.dart';
import 'widgets/config_sheet.dart';
import 'widgets/stats_bar.dart';

/// The streaming chat surface.
///
/// A thin view over [SipllmController]: it renders [SipllmController.messages]
/// as [ChatBubble]s, keeps the list pinned to the newest token via a
/// [ScrollController], surfaces live [StatsBar] telemetry, and hosts the
/// composer ([ChatInput]) plus the generation [ConfigSheet]. All mutations are
/// delegated to the controller; the widget owns only scroll state.
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _scrollController = ScrollController();

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _confirmReset(SipllmController controller) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppTheme.surface,
        title: const Text('Reset context?',
            style: TextStyle(color: AppTheme.textMain)),
        content: const Text(
          'This clears the conversation and the model\'s KV cache.',
          style: TextStyle(color: AppTheme.textSec),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel',
                style: TextStyle(color: AppTheme.textSec)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Reset',
                style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    if (ok == true) controller.resetChat();
  }

  void _openConfig(SipllmController controller) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ConfigSheet(controller: controller),
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<SipllmController>();
    final loaded = controller.isModelLoaded;
    _scrollToBottom();

    final showStats = controller.lastStats != null || controller.generating;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        elevation: 0,
        title: Text(
          controller.loadedModel?.name ?? 'Chat',
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(color: AppTheme.textMain, fontSize: 16),
        ),
        actions: [
          IconButton(
            tooltip: 'Reset context',
            icon: const Icon(LucideIcons.refreshCw, size: 20),
            color: AppTheme.textSec,
            onPressed: controller.messages.isEmpty
                ? null
                : () => _confirmReset(controller),
          ),
          IconButton(
            tooltip: 'Config',
            icon: const Icon(LucideIcons.sliders, size: 20),
            color: AppTheme.textSec,
            onPressed: () => _openConfig(controller),
          ),
        ],
      ),
      body: Column(
        children: [
          if (showStats)
            StatsBar(
              stats: controller.lastStats,
              generating: controller.generating,
              threads: controller.threads,
            ),
          Expanded(child: _messageArea(controller, loaded)),
          if (!loaded) _noModelBanner(),
          ChatInput(
            enabled: loaded,
            generating: controller.generating,
            onSend: (text) => context.read<SipllmController>().send(text),
            onStop: () => context.read<SipllmController>().cancelGeneration(),
          ),
        ],
      ),
    );
  }

  Widget _messageArea(SipllmController controller, bool loaded) {
    if (controller.messages.isEmpty) {
      return _emptyState(
        icon: loaded ? LucideIcons.messageSquare : LucideIcons.boxSelect,
        text: loaded
            ? 'Load a model, then say hello.'
            : 'No model loaded — open Models',
      );
    }
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(vertical: 8),
      itemCount: controller.messages.length,
      itemBuilder: (context, i) => ChatBubble(message: controller.messages[i]),
    );
  }

  Widget _emptyState({required IconData icon, required String text}) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 48, color: AppTheme.textSec.withValues(alpha: 0.5)),
          const SizedBox(height: 16),
          Text(
            text,
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppTheme.textSec, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _noModelBanner() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: AppTheme.accent.withValues(alpha: 0.12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(LucideIcons.info, size: 16, color: AppTheme.accent),
          const SizedBox(width: 8),
          Text(
            'No model loaded — open Models to load one',
            style: TextStyle(
              color: AppTheme.textMain.withValues(alpha: 0.9),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}
