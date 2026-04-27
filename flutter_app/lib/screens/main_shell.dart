import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/navigation_provider.dart';
import 'package:cognithor_ui/providers/pip_provider.dart';
import 'package:cognithor_ui/providers/theme_provider.dart';
import 'package:cognithor_ui/screens/admin_hub_screen.dart';
import 'package:cognithor_ui/screens/chat_screen.dart';
import 'package:cognithor_ui/screens/config_screen.dart';
import 'package:cognithor_ui/screens/dashboard_screen.dart';
import 'package:cognithor_ui/screens/identity_screen.dart';
import 'package:cognithor_ui/screens/kanban_screen.dart';
import 'package:cognithor_ui/providers/packs_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/screens/leads_screen.dart';
import 'package:cognithor_ui/screens/research_screen.dart';
import 'package:cognithor_ui/widgets/packs/pack_preview_overlay.dart';
import 'package:cognithor_ui/data/known_packs.dart';
import 'package:cognithor_ui/screens/skills_screen.dart';
import 'package:cognithor_ui/widgets/global_search_dialog.dart';
import 'package:cognithor_ui/widgets/responsive_scaffold.dart';
import 'package:cognithor_ui/widgets/connection_guard.dart';
import 'package:cognithor_ui/widgets/robot_office/pip_overlay.dart';
import 'package:cognithor_ui/screens/trace/trace_list_screen.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  static const List<Widget> _baseScreens = [
    ChatScreen(),
    DashboardScreen(),
    SkillsScreen(),
    AdminHubScreen(),
    IdentityScreen(),
    KanbanScreen(),
    TraceListScreen(),
  ];

  void _openSearch() {
    showDialog(
      context: context,
      builder: (_) => GlobalSearchDialog(
        onNavigate: (pageKey) {
          Navigator.of(context).push(
            MaterialPageRoute<void>(
              builder: (_) => ConfigScreen(initialPageKey: pageKey),
            ),
          );
        },
      ),
    );
  }

  void _navigateTab(int index) {
    final leadsOn = context.read<SourcesProvider>().sources.isNotEmpty;
    // base screens + optional leads + always-present research
    final maxIndex = leadsOn ? _baseScreens.length + 1 : _baseScreens.length;
    if (index >= 0 && index <= maxIndex) {
      context.read<NavigationProvider>().setTab(index);
    }
  }

  /// Wraps the scaffold with the Robot Office PiP overlay when visible.
  Widget _wrapWithPip(PipProvider pip, Widget scaffold) {
    if (pip.visible) {
      return RobotOfficePip(child: scaffold);
    }
    return scaffold;
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final themeProvider = context.watch<ThemeProvider>();
    final nav = context.watch<NavigationProvider>();
    final leadsEngineEnabled = context
        .watch<SourcesProvider>()
        .sources
        .isNotEmpty;
    final packsProvider = context.watch<PacksProvider>();
    final researchPackLoaded = packsProvider.hasPackLoaded(
      'cognithor-official/deep-research-analyst',
    );

    final deepResearchPack = findKnownPackBySourceId('research')!;

    final screens = <Widget>[
      ..._baseScreens,
      if (leadsEngineEnabled) const LeadsScreen(),
      if (researchPackLoaded)
        const ResearchScreen()
      else
        PackPreviewOverlay(
          pack: deepResearchPack,
          child: const ResearchScreen(),
        ),
    ];

    final navItems = <NavItem>[
      NavItem(
        icon: Icons.chat_bubble_outline,
        selectedIcon: Icons.chat_bubble,
        label: l.chat,
        shortcut: '^1',
      ),
      NavItem(
        icon: Icons.dashboard_outlined,
        selectedIcon: Icons.dashboard,
        label: l.dashboard,
        shortcut: '^2',
      ),
      NavItem(
        icon: Icons.extension_outlined,
        selectedIcon: Icons.extension,
        label: l.skills,
        shortcut: '^3',
      ),
      NavItem(
        icon: Icons.admin_panel_settings_outlined,
        selectedIcon: Icons.admin_panel_settings,
        label: l.adminTitle,
        shortcut: '^4',
      ),
      NavItem(
        icon: Icons.psychology_outlined,
        selectedIcon: Icons.psychology,
        label: l.identity,
        shortcut: '^5',
      ),
      NavItem(
        icon: Icons.view_kanban_outlined,
        selectedIcon: Icons.view_kanban,
        label: l.kanban,
        shortcut: '^6',
      ),
      const NavItem(
        icon: Icons.timeline_outlined,
        selectedIcon: Icons.timeline,
        label: 'Traces',
        shortcut: '^7',
      ),
      if (leadsEngineEnabled)
        NavItem(
          icon: Icons.track_changes_outlined,
          selectedIcon: Icons.track_changes,
          label: l.redditLeads,
          shortcut: '^8',
        ),
      NavItem(
        icon: Icons.biotech_outlined,
        selectedIcon: Icons.biotech,
        label: l.research,
        shortcut: leadsEngineEnabled ? '^9' : '^8',
      ),
    ];

    // Clamp current tab if leads was disabled while on that tab.
    final safeIndex = nav.currentTab >= screens.length ? 0 : nav.currentTab;

    final pipProvider = context.watch<PipProvider>();

    return ConnectionGuard(
      child: CallbackShortcuts(
        bindings: {
          const SingleActivator(LogicalKeyboardKey.keyK, control: true):
              _openSearch,
          const SingleActivator(LogicalKeyboardKey.digit1, control: true): () =>
              _navigateTab(0),
          const SingleActivator(LogicalKeyboardKey.digit2, control: true): () =>
              _navigateTab(1),
          const SingleActivator(LogicalKeyboardKey.digit3, control: true): () =>
              _navigateTab(2),
          const SingleActivator(LogicalKeyboardKey.digit4, control: true): () =>
              _navigateTab(3),
          const SingleActivator(LogicalKeyboardKey.digit5, control: true): () =>
              _navigateTab(4),
          const SingleActivator(LogicalKeyboardKey.digit6, control: true): () =>
              _navigateTab(5),
          const SingleActivator(LogicalKeyboardKey.digit7, control: true): () =>
              _navigateTab(6),
          if (leadsEngineEnabled)
            const SingleActivator(
              LogicalKeyboardKey.digit8,
              control: true,
            ): () =>
                _navigateTab(7),
          const SingleActivator(LogicalKeyboardKey.digit9, control: true): () =>
              _navigateTab(leadsEngineEnabled ? 8 : 7),
        },
        child: Focus(
          autofocus: true,
          child: _wrapWithPip(
            pipProvider,
            ResponsiveScaffold(
              screens: screens,
              navItems: navItems,
              currentIndex: safeIndex,
              onIndexChanged: _navigateTab,
              onSearchTap: _openSearch,
              onThemeToggle: () => themeProvider.toggle(),
              isDark: themeProvider.isDark,
            ),
          ),
        ),
      ),
    );
  }
}
