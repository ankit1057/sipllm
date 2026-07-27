import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';

import '../../core/theme/app_theme.dart';
import '../arena/prompt_arena_screen.dart';
import '../labs/labs_screen.dart';
import '../models/models_screen.dart';
import '../play/play_screen.dart';

/// Studio Hub — unifies Models, Prompt Arena, Offline Play Arcade, and Experimental Labs
/// into a clean, segmented tab interface optimized for mobile and tablet screens.
class StudioHubScreen extends StatefulWidget {
  const StudioHubScreen({super.key});

  @override
  State<StudioHubScreen> createState() => _StudioHubScreenState();
}

class _StudioHubScreenState extends State<StudioHubScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        elevation: 0,
        title: const Text('Studio Hub'),
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true,
          indicatorColor: AppTheme.primary,
          labelColor: AppTheme.primary,
          unselectedLabelColor: AppTheme.textSec,
          tabs: const [
            Tab(icon: Icon(LucideIcons.boxes, size: 18), text: 'Models'),
            Tab(icon: Icon(LucideIcons.swords, size: 18), text: 'Arena'),
            Tab(icon: Icon(LucideIcons.gamepad2, size: 18), text: 'Play Arcade'),
            Tab(icon: Icon(LucideIcons.flaskConical, size: 18), text: 'Labs'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: const [
          ModelsScreen(),
          PromptArenaScreen(),
          PlayScreen(),
          LabsScreen(),
        ],
      ),
    );
  }
}
