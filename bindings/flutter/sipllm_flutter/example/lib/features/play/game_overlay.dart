import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';
import 'package:webview_flutter/webview_flutter.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/theme/app_theme.dart';

/// Persistent Floating Game Sheet Overlay — launches offline games in a draggable,
/// push-downable bottom sheet that can be minimized into a floating pill at any time.
/// Allows uninterrupted gameplay while Chat, Playground, and background tasks run.
class GameBottomSheetOverlay extends StatefulWidget {
  const GameBottomSheetOverlay({super.key});

  @override
  State<GameBottomSheetOverlay> createState() => _GameBottomSheetOverlayState();
}

class _GameBottomSheetOverlayState extends State<GameBottomSheetOverlay> {
  late WebViewController _webViewController;
  String? _loadedGame;

  final Map<String, String> _gameAssets = const {
    '2048': 'assets/games/2048/index.html',
    'Snake': 'assets/games/snake/index.html',
    'Sudoku': 'assets/games/sudoku/index.html',
    'Minesweeper': 'assets/games/minesweeper/index.html',
  };

  @override
  void initState() {
    super.initState();
    _webViewController = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF0D0F17));
  }

  Future<void> _ensureLoaded(String gameName) async {
    if (_loadedGame != gameName) {
      _loadedGame = gameName;
      final assetPath = _gameAssets[gameName];
      if (assetPath != null) {
        try {
          final htmlContent = await rootBundle.loadString(assetPath);
          await _webViewController.loadHtmlString(
            htmlContent,
            baseUrl: 'about:blank',
          );
        } catch (e) {
          debugPrint('Failed to load game HTML asset $assetPath: $e');
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();
    final activeGame = c.activeGame;

    if (activeGame == null) {
      return const SizedBox.shrink();
    }

    _ensureLoaded(activeGame);

    if (c.gameMinimized) {
      // Minimized Floating Pill (Bottom-Right)
      return Positioned(
        bottom: 80,
        right: 16,
        child: Material(
          elevation: 8,
          borderRadius: BorderRadius.circular(30),
          color: AppTheme.primary,
          child: InkWell(
            borderRadius: BorderRadius.circular(30),
            onTap: () => c.resumeGame(),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(LucideIcons.gamepad2, size: 18, color: Colors.white),
                  const SizedBox(width: 8),
                  Text(
                    'Playing $activeGame',
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.all(4),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(LucideIcons.maximize2, size: 12, color: Colors.white),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    // Expanded Draggable Bottom Sheet
    final screenHeight = MediaQuery.of(context).size.height;
    return Align(
      alignment: Alignment.bottomCenter,
      child: GestureDetector(
        onVerticalDragEnd: (details) {
          if (details.primaryVelocity != null && details.primaryVelocity! > 300) {
            c.minimizeGame();
          }
        },
        child: Container(
          height: screenHeight * 0.72,
          width: double.infinity,
          decoration: const BoxDecoration(
            color: Color(0xFF0D0F17),
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
            boxShadow: [
              BoxShadow(
                color: Colors.black54,
                blurRadius: 20,
                spreadRadius: 5,
              ),
            ],
          ),
          child: Column(
            children: [
              // Top Drag Handle & Title Bar
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: const BoxDecoration(
                  border: Border(bottom: BorderSide(color: Colors.white10)),
                ),
                child: Column(
                  children: [
                    Container(
                      width: 40,
                      height: 5,
                      margin: const EdgeInsets.only(bottom: 10),
                      decoration: BoxDecoration(
                        color: Colors.white30,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                    Row(
                      children: [
                        const Icon(LucideIcons.gamepad2, color: AppTheme.primary, size: 20),
                        const SizedBox(width: 8),
                        Text(
                          activeGame,
                          style: const TextStyle(color: AppTheme.textMain, fontWeight: FontWeight.bold, fontSize: 16),
                        ),
                        const Spacer(),
                        // Game Selector Chips
                        ..._gameAssets.keys.map((game) {
                          final isSelected = game == activeGame;
                          return GestureDetector(
                            onTap: () => c.launchGame(game),
                            child: Container(
                              margin: const EdgeInsets.only(right: 6),
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color: isSelected ? AppTheme.primary.withValues(alpha: 0.25) : AppTheme.surface,
                                borderRadius: BorderRadius.circular(6),
                                border: Border.all(color: isSelected ? AppTheme.primary : Colors.white10),
                              ),
                              child: Text(
                                game,
                                style: TextStyle(
                                  color: isSelected ? AppTheme.primary : AppTheme.textSec,
                                  fontSize: 10,
                                  fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                                ),
                              ),
                            ),
                          );
                        }),
                        IconButton(
                          icon: const Icon(LucideIcons.minus, color: AppTheme.textSec, size: 18),
                          tooltip: 'Minimize & Resume Later',
                          onPressed: () => c.minimizeGame(),
                        ),
                        IconButton(
                          icon: const Icon(LucideIcons.x, color: AppTheme.textSec, size: 18),
                          tooltip: 'Close Game',
                          onPressed: () => c.closeGame(),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              // WebView Touch Canvas Area
              Expanded(
                child: ClipRRect(
                  child: WebViewWidget(controller: _webViewController),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
