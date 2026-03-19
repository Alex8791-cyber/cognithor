import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/providers/theme_provider.dart';
import 'package:jarvis_ui/screens/admin_hub_screen.dart';
import 'package:jarvis_ui/screens/chat_screen.dart';
import 'package:jarvis_ui/screens/config_screen.dart';
import 'package:jarvis_ui/screens/dashboard_screen.dart';
import 'package:jarvis_ui/screens/identity_screen.dart';
import 'package:jarvis_ui/screens/skills_screen.dart';
import 'package:jarvis_ui/widgets/global_search_dialog.dart';
import 'package:jarvis_ui/widgets/responsive_scaffold.dart';
import 'package:jarvis_ui/widgets/robot_office/pip_overlay.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    ChatScreen(),
    DashboardScreen(),
    SkillsScreen(),
    AdminHubScreen(),
    IdentityScreen(),
  ];

  void _openSearch() {
    showDialog(
      context: context,
      builder: (_) => GlobalSearchDialog(
        onNavigate: (pageIndex) {
          Navigator.of(context).push(
            MaterialPageRoute<void>(
              builder: (_) => const ConfigScreen(),
            ),
          );
        },
      ),
    );
  }

  void _navigateTab(int index) {
    if (index >= 0 && index < _screens.length) {
      setState(() => _currentIndex = index);
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

    final navItems = [
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
    ];

    final pipProvider = context.watch<PipProvider>();

    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.keyK, control: true):
            _openSearch,
        const SingleActivator(LogicalKeyboardKey.digit1, control: true):
            () => _navigateTab(0),
        const SingleActivator(LogicalKeyboardKey.digit2, control: true):
            () => _navigateTab(1),
        const SingleActivator(LogicalKeyboardKey.digit3, control: true):
            () => _navigateTab(2),
        const SingleActivator(LogicalKeyboardKey.digit4, control: true):
            () => _navigateTab(3),
        const SingleActivator(LogicalKeyboardKey.digit5, control: true):
            () => _navigateTab(4),
      },
      child: Focus(
        autofocus: true,
        child: _wrapWithPip(
          pipProvider,
          ResponsiveScaffold(
            screens: _screens,
            navItems: navItems,
            currentIndex: _currentIndex,
            onIndexChanged: _navigateTab,
            onSearchTap: _openSearch,
            onThemeToggle: () => themeProvider.toggle(),
            isDark: themeProvider.isDark,
          ),
        ),
      ),
    );
  }
}
