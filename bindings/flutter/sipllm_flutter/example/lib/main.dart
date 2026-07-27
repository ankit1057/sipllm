import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:provider/provider.dart';

import 'core/controller/sipllm_controller.dart';
import 'core/theme/app_theme.dart';
import 'core/widgets/sipllm_logo.dart';
import 'features/chat/chat_screen.dart';
import 'features/home/home_screen.dart';
import 'features/hub/studio_hub_screen.dart';
import 'features/play/game_overlay.dart';
import 'features/playground/playground_screen.dart';
import 'features/visualizer/visualizer_screen.dart';
import 'features/wear/wear_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      statusBarBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.transparent,
      systemNavigationBarDividerColor: Colors.transparent,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  runApp(
    ChangeNotifierProvider(
      create: (_) => SipllmController()..init(),
      child: const SipllmApp(),
    ),
  );
}

class SipllmApp extends StatelessWidget {
  const SipllmApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SipLLM Studio',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      home: const _Bootstrap(),
    );
  }
}

/// Shows a splash until the controller has probed the device + engine, then
/// routes to the phone shell or the Wear OS shell.
class _Bootstrap extends StatelessWidget {
  const _Bootstrap();

  @override
  Widget build(BuildContext context) {
    final c = context.watch<SipllmController>();
    if (!c.initialized) {
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SipLlmLogo(size: 72),
              SizedBox(height: 20),
              CircularProgressIndicator(color: AppTheme.primary),
              SizedBox(height: 16),
              Text('Waking SipLLM Studio…', style: TextStyle(color: AppTheme.textSec, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      );
    }
    return c.isWearOs ? const WearScreen() : const RootShell();
  }
}

/// Phone & Tablet shell for SipLLM Studio: responsive navigation backed by an IndexedStack,
/// overlaid with a persistent, push-downable GameBottomSheetOverlay.
class RootShell extends StatefulWidget {
  const RootShell({super.key});

  @override
  State<RootShell> createState() => _RootShellState();
}

class _RootShellState extends State<RootShell> {
  int _index = 0;

  static const _screens = [
    HomeScreen(),
    PlaygroundScreen(),
    VisualizerScreen(),
    ChatScreen(),
    StudioHubScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    final isTablet = MediaQuery.of(context).size.width >= 720;

    if (isTablet) {
      return Scaffold(
        body: Stack(
          children: [
            Row(
              children: [
                NavigationRail(
                  selectedIndex: _index,
                  onDestinationSelected: (i) => setState(() => _index = i),
                  backgroundColor: AppTheme.surface,
                  indicatorColor: AppTheme.primary.withValues(alpha: 0.25),
                  labelType: NavigationRailLabelType.all,
                  destinations: const [
                    NavigationRailDestination(icon: Icon(LucideIcons.home), label: Text('Home')),
                    NavigationRailDestination(icon: Icon(LucideIcons.sliders), label: Text('Playground')),
                    NavigationRailDestination(icon: Icon(LucideIcons.layers), label: Text('Visualizer')),
                    NavigationRailDestination(icon: Icon(LucideIcons.messageSquare), label: Text('Chat')),
                    NavigationRailDestination(icon: Icon(LucideIcons.grid), label: Text('Hub')),
                  ],
                ),
                const VerticalDivider(width: 1, color: Colors.white10),
                Expanded(child: IndexedStack(index: _index, children: _screens)),
              ],
            ),
            const GameBottomSheetOverlay(),
          ],
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          IndexedStack(index: _index, children: _screens),
          const GameBottomSheetOverlay(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        backgroundColor: AppTheme.surface,
        indicatorColor: AppTheme.primary.withValues(alpha: 0.2),
        destinations: const [
          NavigationDestination(icon: Icon(LucideIcons.home), label: 'Home'),
          NavigationDestination(icon: Icon(LucideIcons.sliders), label: 'Play'),
          NavigationDestination(icon: Icon(LucideIcons.layers), label: 'Layers'),
          NavigationDestination(icon: Icon(LucideIcons.messageSquare), label: 'Chat'),
          NavigationDestination(icon: Icon(LucideIcons.grid), label: 'Hub'),
        ],
      ),
    );
  }
}
