import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';

/// Offline Play Arcade Hub — launch self-contained HTML5 games into a push-downable,
/// persistent bottom sheet dialog that stays running while you chat or benchmark.
class PlayScreen extends StatelessWidget {
  const PlayScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();

    final games = [
      {'name': '2048', 'desc': 'Slide matching tiles to reach 2048', 'icon': LucideIcons.boxes, 'color': AppTheme.primary},
      {'name': 'Snake', 'desc': 'Classic arcade snake with touch controls', 'icon': LucideIcons.activity, 'color': Colors.greenAccent},
      {'name': 'Sudoku', 'desc': 'Logic puzzle grid with touch number pad', 'icon': LucideIcons.grid, 'color': AppTheme.accent},
      {'name': 'Minesweeper', 'desc': 'Mine detection board game', 'icon': LucideIcons.bomb, 'color': Colors.redAccent},
    ];

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        title: const Text('Offline Play Arcade'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Banner Card
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppTheme.primary.withValues(alpha: 0.3)),
            ),
            child: const Row(
              children: [
                Icon(LucideIcons.gamepad2, size: 32, color: AppTheme.primary),
                SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Push-Downable Arcade',
                        style: TextStyle(color: AppTheme.textMain, fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'Launch a game into a floating bottom sheet. Drag down at any time to minimize and resume later while Chat runs!',
                        style: TextStyle(color: AppTheme.textSec, fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
          const Text('Available Offline Games', style: TextStyle(color: AppTheme.textMain, fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          ...games.map((g) {
            final gameName = g['name'] as String;
            final isCurrentlyActive = c.activeGame == gameName;

            return Card(
              color: AppTheme.surface,
              margin: const EdgeInsets.only(bottom: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                leading: CircleAvatar(
                  backgroundColor: (g['color'] as Color).withValues(alpha: 0.15),
                  child: Icon(g['icon'] as IconData, color: g['color'] as Color, size: 20),
                ),
                title: Text(gameName, style: const TextStyle(color: AppTheme.textMain, fontWeight: FontWeight.bold, fontSize: 15)),
                subtitle: Text(g['desc'] as String, style: const TextStyle(color: AppTheme.textSec, fontSize: 12)),
                trailing: ElevatedButton.icon(
                  onPressed: () {
                    if (isCurrentlyActive && c.gameMinimized) {
                      c.resumeGame();
                    } else {
                      c.launchGame(gameName);
                    }
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isCurrentlyActive ? AppTheme.accent : AppTheme.primary,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  icon: Icon(isCurrentlyActive ? LucideIcons.play : LucideIcons.rocket, size: 14),
                  label: Text(
                    isCurrentlyActive ? (c.gameMinimized ? 'Resume' : 'Playing') : 'Launch',
                    style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            );
          }),
        ],
      ),
    );
  }
}
