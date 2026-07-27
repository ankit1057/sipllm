import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:local_llm_chat/main.dart';
import 'package:lucide_icons/lucide_icons.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Comprehensive SipLLM Studio End-to-End Application Driving Test', (WidgetTester tester) async {
    // 1. Boot Application & Wait for Engine Probe
    await tester.pumpWidget(const SipllmApp());

    // Loop-pump until the initialization splash finishes and RootShell is rendered
    for (int i = 0; i < 20; i++) {
      await tester.pump(const Duration(milliseconds: 500));
      if (find.text('SipLLM Studio').evaluate().isNotEmpty) break;
    }

    // 2. Drive Home Screen Dashboard
    expect(find.text('SipLLM Studio'), findsWidgets);

    // 3. Drive Playground Screen
    final playgroundIcon = find.byIcon(LucideIcons.sliders);
    if (playgroundIcon.evaluate().isNotEmpty) {
      await tester.tap(playgroundIcon.first);
      await tester.pumpAndSettle();

      expect(find.text('RAM Budget'), findsOneWidget);
      expect(find.text('CPU Threads'), findsOneWidget);

      // Interact with RAM Budget slider
      final sliders = find.byType(Slider);
      if (sliders.evaluate().isNotEmpty) {
        await tester.drag(sliders.first, const Offset(50, 0));
        await tester.pumpAndSettle();
      }

      // Enter prompt into TextField
      final promptInput = find.byType(TextField);
      if (promptInput.evaluate().isNotEmpty) {
        await tester.enterText(promptInput.first, 'Why is SipLLM streaming fast?');
        await tester.pumpAndSettle();
      }
    }

    // 4. Drive Visualizer Screen
    final visualizerIcon = find.byIcon(LucideIcons.layers);
    if (visualizerIcon.evaluate().isNotEmpty) {
      await tester.tap(visualizerIcon.first);
      await tester.pumpAndSettle();

      expect(find.text('Layer Residency Map'), findsOneWidget);
      expect(find.text('Storage Explorer'), findsOneWidget);
    }

    // 5. Drive Local Chat Screen
    final chatIcon = find.byIcon(LucideIcons.messageSquare);
    if (chatIcon.evaluate().isNotEmpty) {
      await tester.tap(chatIcon.first);
      await tester.pumpAndSettle();

      expect(find.text('Ask SipLLM anything…'), findsOneWidget);
    }

    // 6. Drive Studio Hub (Models, Arena, Arcade, Labs)
    final hubIcon = find.byIcon(LucideIcons.grid);
    if (hubIcon.evaluate().isNotEmpty) {
      await tester.tap(hubIcon.first);
      await tester.pumpAndSettle();

      expect(find.text('Studio Hub'), findsOneWidget);
      expect(find.text('Models'), findsWidgets);
      expect(find.text('Arena'), findsOneWidget);
      expect(find.text('Play Arcade'), findsOneWidget);
      expect(find.text('Labs'), findsOneWidget);

      // Switch to Arena Sub-Tab
      await tester.tap(find.text('Arena'));
      await tester.pumpAndSettle();
      expect(find.text('Prompt Arena'), findsOneWidget);

      // Switch to Play Arcade Sub-Tab
      await tester.tap(find.text('Play Arcade'));
      await tester.pumpAndSettle();
      expect(find.text('Push-Downable Arcade'), findsOneWidget);

      // Launch 2048 Game into Bottom Sheet Overlay
      final launchBtn = find.text('Launch').first;
      await tester.tap(launchBtn);
      await tester.pumpAndSettle();

      // Verify Game Bottom Sheet Overlay is Active
      expect(find.text('2048'), findsWidgets);
      expect(find.byIcon(LucideIcons.minus), findsOneWidget);

      // Minimize Game Overlay into Floating Pill
      await tester.tap(find.byIcon(LucideIcons.minus));
      await tester.pumpAndSettle();

      // Verify Minimized Pill
      expect(find.text('Playing 2048'), findsOneWidget);

      // Tap Pill to Resume Game Overlay
      await tester.tap(find.text('Playing 2048'));
      await tester.pumpAndSettle();

      // Close Game Overlay
      await tester.tap(find.byIcon(LucideIcons.x));
      await tester.pumpAndSettle();

      // Switch to Labs Sub-Tab
      await tester.tap(find.text('Labs'));
      await tester.pumpAndSettle();
      expect(find.text('Vulkan Matmul Status'), findsOneWidget);
    }
  });
}
