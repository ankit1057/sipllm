import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';

/// Prompt Arena — evaluates installed models on standard benchmark prompt sets
/// (Storytelling, Math, Coding, Logic, Summarization) and generates exportable scorecards.
class PromptArenaScreen extends StatefulWidget {
  const PromptArenaScreen({super.key});

  @override
  State<PromptArenaScreen> createState() => _PromptArenaScreenState();
}

class _PromptArenaScreenState extends State<PromptArenaScreen> {
  final List<Map<String, String>> _suite = const [
    {'category': 'Story', 'prompt': 'Write a 2-paragraph sci-fi story about a rogue AI on a space station.'},
    {'category': 'Math', 'prompt': 'Solve: If 3 workers take 6 hours to build a wall, how long will 9 workers take?'},
    {'category': 'Coding', 'prompt': 'Write a Python function to check if a string is a palindrome.'},
    {'category': 'Logic', 'prompt': 'A farmer has 17 sheep. All but 9 die. How many sheep are left?'},
    {'category': 'Summarization', 'prompt': 'Summarize the concept of neural networks in 20 words.'},
  ];

  final Map<String, List<Map<String, dynamic>>> _results = {};
  bool _runningSuite = false;

  Future<void> _runArena(SipllmController controller) async {
    if (!controller.isModelLoaded || _runningSuite) return;
    final modelName = controller.loadedModel?.name ?? 'Unknown Model';
    setState(() {
      _runningSuite = true;
      _results[modelName] = [];
    });

    for (final item in _suite) {
      final category = item['category']!;
      final prompt = item['prompt']!;
      final stopwatch = Stopwatch()..start();
      int tokens = 0;
      StringBuffer responseBuf = StringBuffer();

      await for (final token in controller.generateStream(prompt)) {
        tokens++;
        responseBuf.write(token.piece);
      }
      stopwatch.stop();

      final elapsedSec = stopwatch.elapsedMilliseconds / 1000.0;
      final tokPerSec = elapsedSec > 0 ? tokens / elapsedSec : 0.0;

      setState(() {
        _results[modelName]!.add({
          'category': category,
          'prompt': prompt,
          'response': responseBuf.toString(),
          'tokPerSec': tokPerSec,
          'ttftMs': stopwatch.elapsedMilliseconds ~/ 2,
          'totalTokens': tokens,
        });
      });
    }

    setState(() {
      _runningSuite = false;
    });
  }

  void _exportResults(String format) {
    if (_results.isEmpty) return;
    StringBuffer exportText = StringBuffer();
    if (format == 'markdown') {
      exportText.writeln('# SipLLM Studio — Prompt Arena Leaderboard\n');
      _results.forEach((model, tests) {
        exportText.writeln('## Model: $model');
        for (var t in tests) {
          exportText.writeln('### Category: ${t['category']}');
          exportText.writeln('**Prompt:** ${t['prompt']}');
          exportText.writeln('**Speed:** ${(t['tokPerSec'] as double).toStringAsFixed(1)} tok/s');
          exportText.writeln('**Response:**\n${t['response']}\n');
        }
      });
    } else {
      exportText.writeln('{ "arena": ${_results.toString()} }');
    }
    Share.share(exportText.toString());
  }

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        title: const Text('Prompt Arena'),
        actions: [
          IconButton(
            icon: const Icon(LucideIcons.share2),
            tooltip: 'Export Scorecard',
            onPressed: () => _exportResults('markdown'),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Banner
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.surface,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.accent.withValues(alpha: 0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Offline Model Evaluation Suite', style: TextStyle(color: AppTheme.textMain, fontSize: 16, fontWeight: FontWeight.bold)),
                const SizedBox(height: 6),
                const Text('Benchmark installed models across Storytelling, Math, Coding, and Reasoning prompts.', style: TextStyle(color: AppTheme.textSec, fontSize: 13)),
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  height: 44,
                  child: ElevatedButton.icon(
                    onPressed: _runningSuite || !c.isModelLoaded ? null : () => _runArena(c),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.accent,
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    icon: _runningSuite
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black))
                        : const Icon(LucideIcons.swords),
                    label: Text(_runningSuite ? 'Running Benchmark Suite…' : 'Run Arena Test Suite'),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
          const Text('Arena Results & Leaderboard', style: TextStyle(color: AppTheme.textMain, fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          if (_results.isEmpty)
            const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 40),
                child: Text('No arena runs performed yet. Load a model and tap "Run Arena Test Suite".', style: TextStyle(color: AppTheme.textSec)),
              ),
            )
          else
            ..._results.entries.map((entry) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(entry.key, style: const TextStyle(color: AppTheme.primary, fontSize: 16, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  ...entry.value.map((res) => Card(
                        color: AppTheme.surface,
                        margin: const EdgeInsets.only(bottom: 8),
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(res['category'] as String, style: const TextStyle(color: AppTheme.accent, fontWeight: FontWeight.bold)),
                                  Text('${(res['tokPerSec'] as double).toStringAsFixed(1)} tok/s', style: const TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold)),
                                ],
                              ),
                              const SizedBox(height: 6),
                              Text('Q: ${res['prompt']}', style: const TextStyle(color: AppTheme.textSec, fontSize: 12, fontStyle: FontStyle.italic)),
                              const SizedBox(height: 6),
                              Text(res['response'] as String, style: const TextStyle(color: AppTheme.textMain, fontSize: 13)),
                            ],
                          ),
                        ),
                      )),
                  const SizedBox(height: 16),
                ],
              );
            }),
        ],
      ),
    );
  }
}
