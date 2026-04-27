import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/navigation_provider.dart';
import 'package:cognithor_ui/providers/pip_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/animated_indexed_stack.dart';
import 'package:cognithor_ui/widgets/command_bar.dart';

// ── Data model ─────────────────────────────────────────────────────────────

/// Describes a single navigation destination.
class NavItem {
  const NavItem({
    required this.icon,
    required this.selectedIcon,
    required this.label,
    this.shortcut,
  });

  final IconData icon;
  final IconData selectedIcon;
  final String label;
  final String? shortcut;
}

// ── Breakpoints ────────────────────────────────────────────────────────────

enum _Layout { mobile, tablet, desktop }

_Layout _layoutFor(double width) {
  if (width >= 800) return _Layout.desktop;
  if (width >= 500) return _Layout.tablet;
  return _Layout.mobile;
}

// ── Constants ──────────────────────────────────────────────────────────────

const Duration _railAnimDuration = Duration(milliseconds: 300);
const Curve _railAnimCurve = Curves.easeOutQuart;

// ── ResponsiveScaffold ─────────────────────────────────────────────────────

class ResponsiveScaffold extends StatefulWidget {
  const ResponsiveScaffold({
    super.key,
    required this.screens,
    required this.navItems,
    required this.currentIndex,
    required this.onIndexChanged,
    this.actions,
    this.title,
    this.onSearchTap,
    this.onThemeToggle,
    this.isDark = true,
  });

  final List<Widget> screens;
  final List<NavItem> navItems;
  final int currentIndex;
  final ValueChanged<int> onIndexChanged;
  final List<Widget>? actions;
  final String? title;
  final VoidCallback? onSearchTap;
  final VoidCallback? onThemeToggle;
  final bool isDark;

  @override
  State<ResponsiveScaffold> createState() => _ResponsiveScaffoldState();
}

class _ResponsiveScaffoldState extends State<ResponsiveScaffold> {
  /// Tablet hover expansion.
  bool _tabletHovered = false;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final layout = _layoutFor(constraints.maxWidth);
        return switch (layout) {
          _Layout.mobile => _buildMobile(context),
          _Layout.tablet => _buildSideLayout(context, expanded: _tabletHovered),
          _Layout.desktop => _buildSideLayout(context, expanded: true),
        };
      },
    );
  }

  // ── Mobile (bottom nav) ─────────────────────────────────────────────────

  Widget _buildMobile(BuildContext context) {
    final isDark = widget.isDark;

    return Scaffold(
      body: Column(
        children: [
          CommandBar(onSearchTap: widget.onSearchTap),
          Expanded(
            child: AnimatedIndexedStack(
              index: widget.currentIndex,
              children: widget.screens,
            ),
          ),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          border: Border(
            top: BorderSide(
              color: isDark
                  ? Theme.of(context).dividerColor
                  : const Color(0xFFE0E0E8),
            ),
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
            child: Row(
              children: [
                ...List.generate(widget.navItems.length, (i) {
                  final item = widget.navItems[i];
                  final selected = i == widget.currentIndex;
                  final sectionColor = CognithorTheme.sectionColorFor(i);
                  return Expanded(
                    child: InkWell(
                      onTap: () => widget.onIndexChanged(i),
                      borderRadius: BorderRadius.circular(12),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(vertical: 6),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              selected ? item.selectedIcon : item.icon,
                              size: 22,
                              color: selected
                                  ? sectionColor
                                  : Theme.of(context).iconTheme.color,
                            ),
                            const SizedBox(height: 2),
                            Text(
                              item.label,
                              style: TextStyle(
                                fontSize:
                                    MediaQuery.of(context).size.width > 400
                                    ? 12
                                    : 10,
                                fontWeight: selected
                                    ? FontWeight.w600
                                    : FontWeight.normal,
                                color: selected
                                    ? sectionColor
                                    : Theme.of(
                                        context,
                                      ).textTheme.bodySmall?.color,
                              ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                            if (item.shortcut != null) ...[
                              const SizedBox(height: 1),
                              Text(
                                item.shortcut!,
                                style: TextStyle(
                                  fontSize: 8,
                                  color: CognithorTheme.textTertiary,
                                  fontFamily: 'monospace',
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  );
                }),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Side rail layout (tablet + desktop) ─────────────────────────────────

  Widget _buildSideLayout(BuildContext context, {required bool expanded}) {
    final isDark = widget.isDark;
    final l = AppLocalizations.of(context);
    final nav = context.watch<NavigationProvider>();

    // On desktop, sidebar width morphs based on the active section
    final railWidth = expanded ? nav.sidebarWidth : 64.0;

    final railBg = isDark ? CognithorTheme.surface : const Color(0xFFF8F8FC);
    final borderColor = isDark
        ? CognithorTheme.border
        : const Color(0xFFE0E0E8);

    return Scaffold(
      body: Row(
        children: [
          // ── Side Rail ──
          MouseRegion(
            onEnter: (_) {
              if (!expanded) {
                setState(() => _tabletHovered = true);
              }
            },
            onExit: (_) {
              setState(() => _tabletHovered = false);
            },
            child: AnimatedContainer(
              duration: _railAnimDuration,
              curve: _railAnimCurve,
              width: railWidth,
              decoration: BoxDecoration(
                color: railBg,
                border: Border(right: BorderSide(color: borderColor)),
              ),
              child: Column(
                children: [
                  // ── Logo / brand area ──
                  _BreathingLogo(expanded: expanded && nav.sidebarWidth > 80),

                  const SizedBox(height: 8),

                  // ── Nav items ──
                  Expanded(
                    child: ListView(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      children: List.generate(widget.navItems.length, (i) {
                        return _RailNavItem(
                          item: widget.navItems[i],
                          tabIndex: i,
                          selected: i == widget.currentIndex,
                          expanded: expanded && nav.sidebarWidth > 80,
                          onTap: () => widget.onIndexChanged(i),
                        );
                      }),
                    ),
                  ),

                  // ── Bottom actions ──
                  Divider(height: 1, thickness: 1, color: borderColor),
                  Padding(
                    padding: const EdgeInsets.all(8),
                    child: Column(
                      children: [
                        _RailActionButton(
                          icon: Icons.search,
                          label: l.searchLabel,
                          expanded: expanded && nav.sidebarWidth > 80,
                          glowColor: nav.sectionColor,
                          onTap: widget.onSearchTap ?? () {},
                        ),
                        const SizedBox(height: 4),
                        _RailActionButton(
                          icon: isDark ? Icons.light_mode : Icons.dark_mode,
                          label: isDark ? l.lightMode : l.darkMode,
                          expanded: expanded && nav.sidebarWidth > 80,
                          glowColor: CognithorTheme.orange,
                          onTap: widget.onThemeToggle ?? () {},
                        ),
                        const SizedBox(height: 4),
                        Consumer<PipProvider>(
                          builder: (context, pip, _) {
                            return _RailActionButton(
                              icon: Icons.smart_toy,
                              label: pip.visible
                                  ? l.hideRobotOffice
                                  : l.robotOffice,
                              expanded: expanded && nav.sidebarWidth > 80,
                              glowColor: CognithorTheme.purple,
                              onTap: () => pip.toggle(),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // ── Main content area ──
          Expanded(
            child: Column(
              children: [
                // Command bar replaces the old content header
                CommandBar(onSearchTap: widget.onSearchTap),
                // Screen content
                Expanded(
                  child: AnimatedIndexedStack(
                    index: widget.currentIndex,
                    children: widget.screens,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── Breathing Logo (scale 0.98–1.02, 4s loop) ────────────────────────────

class _BreathingLogo extends StatefulWidget {
  const _BreathingLogo({required this.expanded});

  final bool expanded;

  @override
  State<_BreathingLogo> createState() => _BreathingLogoState();
}

class _BreathingLogoState extends State<_BreathingLogo>
    with SingleTickerProviderStateMixin {
  late final AnimationController _breathCtrl;
  late final Animation<double> _scaleAnim;

  @override
  void initState() {
    super.initState();
    _breathCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat(reverse: true);
    _scaleAnim = Tween<double>(
      begin: 0.98,
      end: 1.02,
    ).animate(CurvedAnimation(parent: _breathCtrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _breathCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 16, 12, 0),
      child: Row(
        mainAxisAlignment: widget.expanded
            ? MainAxisAlignment.start
            : MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          // Animated logo
          AnimatedBuilder(
            animation: _scaleAnim,
            builder: (context, child) {
              return Transform.scale(scale: _scaleAnim.value, child: child);
            },
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.asset(
                'assets/logo.png',
                width: 36,
                height: 36,
                fit: BoxFit.cover,
                errorBuilder: (_, e, s) => Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        CognithorTheme.accent,
                        CognithorTheme.accent.withValues(alpha: 0.7),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: CognithorTheme.accent.withValues(alpha: 0.3),
                        blurRadius: 8,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: const Center(
                    child: Icon(
                      Icons.psychology,
                      color: Colors.white,
                      size: 22,
                    ),
                  ),
                ),
              ),
            ),
          ),
          if (widget.expanded) ...[
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Cognithor',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: isDark
                      ? CognithorTheme.textPrimary
                      : const Color(0xFF1A1A2E),
                  letterSpacing: -0.3,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Rail nav item (section-colored active/hover) ─────────────────────────

class _RailNavItem extends StatefulWidget {
  const _RailNavItem({
    required this.item,
    required this.tabIndex,
    required this.selected,
    required this.expanded,
    required this.onTap,
  });

  final NavItem item;
  final int tabIndex;
  final bool selected;
  final bool expanded;
  final VoidCallback onTap;

  @override
  State<_RailNavItem> createState() => _RailNavItemState();
}

class _RailNavItemState extends State<_RailNavItem> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final sectionColor = CognithorTheme.sectionColorFor(widget.tabIndex);
    final selected = widget.selected;

    // Active: Neon pill at 20% opacity
    final bgColor = selected
        ? sectionColor.withValues(alpha: 0.28)
        : _hovered
        ? sectionColor.withValues(alpha: 0.14)
        : Colors.transparent;

    final iconColor = selected
        ? sectionColor
        : (isDark ? CognithorTheme.textSecondary : const Color(0xFF6B6B80));

    final labelColor = selected
        ? sectionColor
        : (isDark ? CognithorTheme.textPrimary : const Color(0xFF1A1A2E));

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: MouseRegion(
        onEnter: (_) => setState(() => _hovered = true),
        onExit: (_) => setState(() => _hovered = false),
        child: GestureDetector(
          onTap: widget.onTap,
          child: AnimatedScale(
            scale: _hovered && !selected ? 1.05 : 1.0,
            duration: CognithorTheme.animDurationFast,
            child: AnimatedContainer(
              duration: CognithorTheme.animDurationFast,
              curve: CognithorTheme.animCurve,
              padding: EdgeInsets.symmetric(
                horizontal: widget.expanded ? 12 : 0,
                vertical: 10,
              ),
              decoration: BoxDecoration(
                color: bgColor,
                borderRadius: BorderRadius.circular(10),
                boxShadow: _hovered && !selected
                    ? [
                        BoxShadow(
                          color: sectionColor.withValues(alpha: 0.25),
                          blurRadius: 18,
                          spreadRadius: -2,
                        ),
                      ]
                    : null,
              ),
              child: Row(
                mainAxisAlignment: widget.expanded
                    ? MainAxisAlignment.start
                    : MainAxisAlignment.center,
                children: [
                  // Thin vertical accent line on left when selected
                  if (selected)
                    Container(
                      width: 3,
                      height: 22,
                      margin: const EdgeInsets.only(right: 8),
                      decoration: BoxDecoration(
                        color: sectionColor,
                        borderRadius: BorderRadius.circular(2),
                        boxShadow: [
                          BoxShadow(
                            color: sectionColor.withValues(alpha: 0.6),
                            blurRadius: 10,
                          ),
                        ],
                      ),
                    ),
                  Icon(
                    selected ? widget.item.selectedIcon : widget.item.icon,
                    size: 22,
                    color: iconColor,
                  ),
                  if (widget.expanded) ...[
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        widget.item.label,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: selected
                              ? FontWeight.w600
                              : FontWeight.w500,
                          color: labelColor,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (widget.item.shortcut != null)
                      Text(
                        widget.item.shortcut!,
                        style: TextStyle(
                          fontSize: 10,
                          color: CognithorTheme.textTertiary,
                          fontFamily: 'monospace',
                        ),
                      ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ── Rail bottom action button (neon glow styling) ────────────────────────

class _RailActionButton extends StatefulWidget {
  const _RailActionButton({
    required this.icon,
    required this.label,
    required this.expanded,
    required this.onTap,
    this.glowColor,
  });

  final IconData icon;
  final String label;
  final bool expanded;
  final VoidCallback onTap;
  final Color? glowColor;

  @override
  State<_RailActionButton> createState() => _RailActionButtonState();
}

class _RailActionButtonState extends State<_RailActionButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final glow = widget.glowColor ?? CognithorTheme.accent;
    final hoverBg = glow.withValues(alpha: 0.10);
    final iconColor = _hovered
        ? glow
        : (isDark ? CognithorTheme.textSecondary : const Color(0xFF6B6B80));

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: CognithorTheme.animDurationFast,
          curve: CognithorTheme.animCurve,
          padding: EdgeInsets.symmetric(
            horizontal: widget.expanded ? 12 : 0,
            vertical: 8,
          ),
          decoration: BoxDecoration(
            color: _hovered ? hoverBg : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
            boxShadow: _hovered
                ? [
                    BoxShadow(
                      color: glow.withValues(alpha: 0.20),
                      blurRadius: 10,
                      spreadRadius: -2,
                    ),
                  ]
                : null,
          ),
          child: Row(
            mainAxisAlignment: widget.expanded
                ? MainAxisAlignment.start
                : MainAxisAlignment.center,
            children: [
              Icon(widget.icon, size: 20, color: iconColor),
              if (widget.expanded) ...[
                const SizedBox(width: 12),
                Text(
                  widget.label,
                  style: TextStyle(fontSize: 12, color: iconColor),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
