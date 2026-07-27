import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';

/// Interactive AI Playground — live inference parameter controls (RAM budget,
/// thread pool size, scheduler policy, temperature) paired with streaming output
/// and real-time telemetry metrics (Peak RSS, TTFT, tok/s, layer residency).
class PlaygroundScreen extends StatefulWidget {
  const PlaygroundScreen({super.key});

  @override
  State<PlaygroundScreen> createState() => _PlaygroundScreenState();
}

class _PlaygroundScreenState extends State<PlaygroundScreen> {
  final TextEditingController _promptController = TextEditingController(
    text: 'Explain quantum computing in three simple sentences.',
  );
  String _streamedText = '';
  double _tokPerSec = 0.0;
  int _ttftMs = 0;
  int _totalTokens = 0;

  @override
  void dispose() {
    _promptController.dispose();
    super.dispose();
  }

  void _runInference(SipllmController controller) {
    if (_promptController.text.trim().isEmpty || controller.generating) return;
    setState(() {
      _streamedText = '';
      _tokPerSec = 0.0;
      _ttftMs = 0;
      _totalTokens = 0;
    });

    final stopwatch = Stopwatch()..start();
    controller.generateStream(_promptController.text.trim()).listen(
      (token) {
        if (_ttftMs == 0) {
          _ttftMs = stopwatch.elapsedMilliseconds;
        }
        _totalTokens++;
        final elapsedSec = stopwatch.elapsedMilliseconds / 1000.0;
        if (elapsedSec > 0) {
          _tokPerSec = _totalTokens / elapsedSec;
        }
        setState(() {
          _streamedText += token.piece;
        });
      },
      onDone: () {
        stopwatch.stop();
        setState(() {});
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();
    final isWide = MediaQuery.of(context).size.width > 700;

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        title: const Text('AI Playground'),
        actions: [
          IconButton(
            icon: const Icon(LucideIcons.rotateCcw),
            tooltip: 'Clear Output',
            onPressed: () => setState(() => _streamedText = ''),
          ),
        ],
      ),
      body: isWide
          ? Row(
              children: [
                Expanded(
                  flex: 5,
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: _buildControls(context, c),
                  ),
                ),
                const VerticalDivider(width: 1, color: Colors.white10),
                Expanded(
                  flex: 6,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: _buildOutputArea(c, expandBox: true),
                  ),
                ),
              ],
            )
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildControls(context, c),
                const SizedBox(height: 16),
                _buildOutputArea(c, expandBox: false),
              ],
            ),
    );
  }

  Widget _buildControls(BuildContext context, SipllmController c) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionHeader(LucideIcons.sliders, 'Inference Parameters'),
        const SizedBox(height: 12),
        // RAM Budget Slider
        Card(
          color: AppTheme.surface,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('RAM Budget', style: TextStyle(color: AppTheme.textMain, fontWeight: FontWeight.w600)),
                    Text(
                      c.ramBudgetMiB == 0 ? 'Unlimited (Stream)' : '${c.ramBudgetMiB} MB',
                      style: const TextStyle(color: AppTheme.primary, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                Slider(
                  value: c.ramBudgetMiB.toDouble(),
                  min: 0,
                  max: 4096,
                  divisions: 32,
                  activeColor: AppTheme.primary,
                  onChanged: (v) => setState(() => c.ramBudgetMiB = v.round()),
                ),
                Text(
                  c.ramBudgetMiB == 0
                      ? 'Stream weights directly from disk into a minimal buffer'
                      : 'Pin up to ${c.ramBudgetMiB} MB of layer weights into RAM',
                  style: const TextStyle(color: AppTheme.textSec, fontSize: 11),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        // Thread Count Slider
        Card(
          color: AppTheme.surface,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('CPU Threads', style: TextStyle(color: AppTheme.textMain, fontWeight: FontWeight.w600)),
                    Text(
                      c.threads == 0 ? 'Auto (${c.hardwareConcurrency})' : '${c.threads}',
                      style: const TextStyle(color: AppTheme.accent, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                Slider(
                  value: c.threads.toDouble(),
                  min: 0,
                  max: (c.hardwareConcurrency > 0 ? c.hardwareConcurrency : 8).toDouble(),
                  divisions: c.hardwareConcurrency > 0 ? c.hardwareConcurrency : 8,
                  activeColor: AppTheme.accent,
                  onChanged: (v) => setState(() => c.threads = v.round()),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        // Prompt Input
        TextField(
          controller: _promptController,
          maxLines: 3,
          style: const TextStyle(color: AppTheme.textMain, fontSize: 14),
          decoration: InputDecoration(
            labelText: 'Prompt',
            labelStyle: const TextStyle(color: AppTheme.textSec),
            filled: true,
            fillColor: AppTheme.surface,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
          ),
        ),
        const SizedBox(height: 14),
        SizedBox(
          width: double.infinity,
          height: 46,
          child: ElevatedButton.icon(
            onPressed: c.generating || !c.isModelLoaded ? null : () => _runInference(c),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primary,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            icon: c.generating
                ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(LucideIcons.play),
            label: Text(
              c.generating ? 'Generating…' : (c.isModelLoaded ? 'Run Inference' : 'Load Model First'),
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildOutputArea(SipllmController c, {bool expandBox = false}) {
    final box = Container(
      width: double.infinity,
      height: expandBox ? null : 260,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: SingleChildScrollView(
        child: Text(
          _streamedText.isEmpty ? 'Output will appear here in real time…' : _streamedText,
          style: TextStyle(
            color: _streamedText.isEmpty ? AppTheme.textSec : AppTheme.textMain,
            fontSize: 14,
            height: 1.4,
            fontFamily: 'monospace',
          ),
        ),
      ),
    );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionHeader(LucideIcons.activity, 'Live Telemetry & Output'),
        const SizedBox(height: 10),
        // Telemetry Grid
        Row(
          children: [
            Expanded(child: _buildTelemetryTile('Speed', '${_tokPerSec.toStringAsFixed(1)} t/s', LucideIcons.zap, AppTheme.primary)),
            const SizedBox(width: 8),
            Expanded(child: _buildTelemetryTile('TTFT', '$_ttftMs ms', LucideIcons.timer, AppTheme.accent)),
            const SizedBox(width: 8),
            Expanded(child: _buildTelemetryTile('Tokens', '$_totalTokens', LucideIcons.hash, Colors.greenAccent)),
          ],
        ),
        const SizedBox(height: 12),
        expandBox ? Expanded(child: box) : box,
      ],
    );
  }

  Widget _buildSectionHeader(IconData icon, String title) {
    return Row(
      children: [
        Icon(icon, size: 18, color: AppTheme.primary),
        const SizedBox(width: 8),
        Text(title, style: const TextStyle(color: AppTheme.textMain, fontSize: 16, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _buildTelemetryTile(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Icon(icon, size: 14, color: color),
              const SizedBox(width: 4),
              Expanded(
                child: Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: AppTheme.textSec, fontSize: 11),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              value,
              style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }
}
