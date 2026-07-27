import 'dart:async';

import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/models/managed_model.dart';
import '../../core/theme/app_theme.dart';
import '../../core/widgets/sipllm_logo.dart';

/// Compact Wear OS shell & Companion Pair App.
/// The smartwatch acts as the receiver/companion half of the SipLLM pair:
/// listens for a model pushed from the phone over the Data Layer, loads it under
/// a tight streaming RAM budget (250 MB), and offers a touch-optimized mini chat.
class WearScreen extends StatefulWidget {
  const WearScreen({super.key});

  @override
  State<WearScreen> createState() => _WearScreenState();
}

class _WearScreenState extends State<WearScreen> {
  final _input = TextEditingController();
  StreamSubscription<TransferProgress>? _sub;
  TransferProgress? _progress;

  @override
  void initState() {
    super.initState();
    // Start receiving as soon as the watch app opens.
    final wear = context.read<SipllmController>().wear;
    unawaited(wear.listenIncoming());
    _sub = wear.events().listen(
      (p) => setState(() => _progress = p),
      onError: (_) {},
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _input.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();
    final installed = c.models.where((m) => m.isInstalled).toList();

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
          children: [
            // Header with SipLlmLogo Art & Companion Badge
            const Center(
              child: Column(
                children: [
                  SipLlmLogo(size: 32),
                  SizedBox(height: 6),
                  Text(
                    'SipLLM Watch',
                    style: TextStyle(
                      color: AppTheme.primary,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 4),
            Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: AppTheme.primary.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  'Companion Pair · 250MB Budget',
                  style: TextStyle(color: AppTheme.primary.withValues(alpha: 0.9), fontSize: 10, fontWeight: FontWeight.w600),
                ),
              ),
            ),
            const SizedBox(height: 14),
            _transferCard(),
            const SizedBox(height: 12),
            if (c.isModelLoaded)
              _miniChat(c)
            else if (installed.isNotEmpty)
              _modelPicker(c, installed)
            else
              _empty(),
          ],
        ),
      ),
    );
  }

  Widget _card({required Widget child}) => Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppTheme.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white10),
        ),
        child: child,
      );

  Widget _transferCard() {
    final p = _progress;
    if (p == null) {
      return _card(
        child: const Row(children: [
          Icon(LucideIcons.radio, color: AppTheme.accent, size: 16),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              'Listening for model push from paired Phone…',
              style: TextStyle(color: AppTheme.textSec, fontSize: 11),
            ),
          ),
        ]),
      );
    }
    return _card(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(
          '${p.state.name} · ${p.filename}',
          style: const TextStyle(color: AppTheme.textMain, fontSize: 11, fontWeight: FontWeight.bold),
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 6),
        LinearProgressIndicator(
          value: p.fraction,
          backgroundColor: AppTheme.background,
          color: AppTheme.primary,
        ),
      ]),
    );
  }

  Widget _modelPicker(SipllmController c, List<ManagedModel> installed) {
    return Column(
      children: [
        for (final m in installed)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: _card(
              child: Row(children: [
                Expanded(
                  child: Text(
                    m.name,
                    style: const TextStyle(color: AppTheme.textMain, fontSize: 11, fontWeight: FontWeight.w600),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (c.loading)
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2, color: AppTheme.primary),
                  )
                else
                  IconButton(
                    icon: const Icon(LucideIcons.play, color: AppTheme.primary, size: 18),
                    onPressed: () => c.loadModel(m),
                  ),
              ]),
            ),
          ),
      ],
    );
  }

  Widget _empty() => _card(
        child: const Text(
          'No model loaded.\nPush one from paired Phone app.',
          textAlign: TextAlign.center,
          style: TextStyle(color: AppTheme.textSec, fontSize: 11),
        ),
      );

  Widget _miniChat(SipllmController c) {
    final last = c.messages.isNotEmpty ? c.messages.last : null;
    return Column(
      children: [
        _card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                c.loadedModel?.name ?? '',
                style: const TextStyle(color: AppTheme.primary, fontSize: 11, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 6),
              Text(
                last == null
                    ? 'Ask me something.'
                    : last.text.isEmpty
                        ? '…'
                        : last.text,
                style: const TextStyle(color: AppTheme.textMain, fontSize: 12),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Row(children: [
          Expanded(
            child: TextField(
              controller: _input,
              style: const TextStyle(color: Colors.white, fontSize: 12),
              decoration: const InputDecoration(hintText: 'Ask…', filled: true),
              onSubmitted: (_) => _send(c),
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            icon: Icon(
              c.generating ? LucideIcons.square : LucideIcons.send,
              color: AppTheme.primary,
              size: 20,
            ),
            onPressed: () => c.generating ? c.cancelGeneration() : _send(c),
          ),
        ]),
      ],
    );
  }

  void _send(SipllmController c) {
    final t = _input.text.trim();
    if (t.isEmpty) return;
    _input.clear();
    unawaited(c.send(t));
  }
}
