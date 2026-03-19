import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/animated_indexed_stack.dart';

// ─── Data model ─────────────────────────────────────────────────────────────

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

// ─── Breakpoints ────────────────────────────────────────────────────────────

enum _Layout { mobile, tablet, desktop }

_Layout _layoutFor(double width) {
  if (width >= 1024) return _Layout.desktop;
  if (width >= 600) return _Layout.tablet;
  return _Layout.mobile;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const double _railExpandedWidth = 220;
const double _railCollapsedWidth = 64;
const Duration _railAnimDuration = Duration(milliseconds: 200);
const Curve _railAnimCurve = Curves.easeOutQuart;

// ─── ResponsiveScaffold ─────────────────────────────────────────────────────

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
  /// Desktop rail expanded / collapsed state.
  bool _desktopExpanded = true;

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
          _Layout.desktop =>
            _buildSideLayout(context, expanded: _desktopExpanded),
        };
      },
    );
  }

  // ── Mobile (bottom nav) ─────────────────────────────────────────────────

  Widget _buildMobile(BuildContext context) {
    final isDark = widget.isDark;
    return Scaffold(
      body: AnimatedIndexedStack(
        index: widget.currentIndex,
        children: widget.screens,
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
                                  ? JarvisTheme.accent
                                  : Theme.of(context).iconTheme.color,
                            ),
                            const SizedBox(height: 2),
                            Text(
                              item.label,
                              style: TextStyle(
                                fontSize: 10,
                                fontWeight: selected
                                    ? FontWeight.w600
                                    : FontWeight.normal,
                                color: selected
                                    ? JarvisTheme.accent
                                    : Theme.of(context)
                                        .textTheme
                                        .bodySmall
                                        ?.color,
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
                                  color: JarvisTheme.textTertiary,
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
                const SizedBox(width: 4),
                Container(
                  width: 1,
                  height: 32,
                  color: isDark
                      ? Theme.of(context).dividerColor
                      : const Color(0xFFE0E0E8),
                ),
                const SizedBox(width: 4),
                _BottomBarAction(
                  icon: Icons.search,
                  label: 'Search',
                  color: JarvisTheme.accent,
                  onTap: widget.onSearchTap ?? () {},
                ),
                _BottomBarAction(
                  icon: isDark ? Icons.light_mode : Icons.dark_mode,
                  label: isDark ? 'Light' : 'Dark',
                  color: JarvisTheme.orange,
                  onTap: widget.onThemeToggle ?? () {},
                ),
                Consumer<PipProvider>(
                  builder: (context, pip, _) {
                    return _BottomBarAction(
                      icon: Icons.smart_toy,
                      label: pip.visible ? 'Hide' : 'Office',
                      color: pip.visible
                          ? JarvisTheme.accent
                          : JarvisTheme.purple,
                      onTap: () => pip.toggle(),
                    );
                  },
                ),
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
    final railWidth = expanded ? _railExpandedWidth : _railCollapsedWidth;

    final railBg = isDark ? JarvisTheme.surface : const Color(0xFFF8F8FC);
    final borderColor =
        isDark ? JarvisTheme.border : const Color(0xFFE0E0E8);

    return Scaffold(
      body: Row(
        children: [
          // ── Side Rail ──
          MouseRegion(
            onEnter: (_) {
              if (!_desktopExpanded) {
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
                  _RailHeader(
                    expanded: expanded,
                    onToggle: () =>
                        setState(() => _desktopExpanded = !_desktopExpanded),
                  ),

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
                          selected: i == widget.currentIndex,
                          expanded: expanded,
                          onTap: () => widget.onIndexChanged(i),
                        );
                      }),
                    ),
                  ),

                  // ── Bottom actions ──
                  const Divider(height: 1),
                  Padding(
                    padding: const EdgeInsets.all(8),
                    child: Column(
                      children: [
                        _RailActionButton(
                          icon: Icons.search,
                          label: 'Search',
                          expanded: expanded,
                          onTap: widget.onSearchTap ?? () {},
                        ),
                        const SizedBox(height: 4),
                        _RailActionButton(
                          icon:
                              isDark ? Icons.light_mode : Icons.dark_mode,
                          label: isDark ? 'Light' : 'Dark',
                          expanded: expanded,
                          onTap: widget.onThemeToggle ?? () {},
                        ),
                        const SizedBox(height: 4),
                        Consumer<PipProvider>(
                          builder: (context, pip, _) {
                            return _RailActionButton(
                              icon: Icons.smart_toy,
                              label:
                                  pip.visible ? 'Hide Office' : 'Robot Office',
                              expanded: expanded,
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
                // Breadcrumb header
                if (widget.title != null ||
                    widget.currentIndex < widget.navItems.length)
                  _ContentHeader(
                    title: widget.title ??
                        widget.navItems[widget.currentIndex].label,
                    actions: widget.actions,
                  ),
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

// ─── Rail header (logo + collapse toggle) ───────────────────────────────────

class _RailHeader extends StatelessWidget {
  const _RailHeader({required this.expanded, required this.onToggle});

  final bool expanded;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 16, 12, 0),
      child: Row(
        children: [
          // Logo
          ClipRRect(
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
                  color: JarvisTheme.accent,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Center(
                  child: Text('C', style: TextStyle(
                    color: Colors.white, fontWeight: FontWeight.w800, fontSize: 18,
                  )),
                ),
              ),
            ),
          ),
          if (expanded) ...[
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Cognithor',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: isDark ? JarvisTheme.textPrimary : const Color(0xFF1A1A2E),
                  letterSpacing: -0.3,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
          if (expanded)
            _HoverIconButton(
              icon: Icons.menu_open,
              size: 18,
              onTap: onToggle,
              tooltip: 'Collapse sidebar',
            )
          else
            _HoverIconButton(
              icon: Icons.menu,
              size: 18,
              onTap: onToggle,
              tooltip: 'Expand sidebar',
            ),
        ],
      ),
    );
  }
}

// ─── Rail nav item ──────────────────────────────────────────────────────────

class _RailNavItem extends StatefulWidget {
  const _RailNavItem({
    required this.item,
    required this.selected,
    required this.expanded,
    required this.onTap,
  });

  final NavItem item;
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
    final accent = isDark ? JarvisTheme.accent : const Color(0xFF0077CC);
    final selected = widget.selected;

    final bgColor = selected
        ? accent.withValues(alpha: 0.12)
        : _hovered
            ? (isDark ? JarvisTheme.surfaceHover : const Color(0xFFF0F0F4))
            : Colors.transparent;

    final iconColor = selected
        ? accent
        : (isDark ? JarvisTheme.textSecondary : const Color(0xFF6B6B80));

    final labelColor = selected
        ? accent
        : (isDark ? JarvisTheme.textPrimary : const Color(0xFF1A1A2E));

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: MouseRegion(
        onEnter: (_) => setState(() => _hovered = true),
        onExit: (_) => setState(() => _hovered = false),
        child: GestureDetector(
          onTap: widget.onTap,
          child: AnimatedContainer(
            duration: JarvisTheme.animDurationFast,
            curve: JarvisTheme.animCurve,
            padding: EdgeInsets.symmetric(
              horizontal: widget.expanded ? 12 : 0,
              vertical: 10,
            ),
            decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              mainAxisAlignment: widget.expanded
                  ? MainAxisAlignment.start
                  : MainAxisAlignment.center,
              children: [
                AnimatedScale(
                  scale: _hovered && !selected ? 1.1 : 1.0,
                  duration: JarvisTheme.animDurationFast,
                  child: Icon(
                    selected ? widget.item.selectedIcon : widget.item.icon,
                    size: 22,
                    color: iconColor,
                  ),
                ),
                if (widget.expanded) ...[
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      widget.item.label,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight:
                            selected ? FontWeight.w600 : FontWeight.w500,
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
                        color: JarvisTheme.textTertiary,
                        fontFamily: 'monospace',
                      ),
                    ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─── Rail bottom action button ──────────────────────────────────────────────

class _RailActionButton extends StatefulWidget {
  const _RailActionButton({
    required this.icon,
    required this.label,
    required this.expanded,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final bool expanded;
  final VoidCallback onTap;

  @override
  State<_RailActionButton> createState() => _RailActionButtonState();
}

class _RailActionButtonState extends State<_RailActionButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final hoverBg =
        isDark ? JarvisTheme.surfaceHover : const Color(0xFFF0F0F4);
    final iconColor =
        isDark ? JarvisTheme.textSecondary : const Color(0xFF6B6B80);

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: JarvisTheme.animDurationFast,
          curve: JarvisTheme.animCurve,
          padding: EdgeInsets.symmetric(
            horizontal: widget.expanded ? 12 : 0,
            vertical: 8,
          ),
          decoration: BoxDecoration(
            color: _hovered ? hoverBg : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
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

// ─── Content header / breadcrumb ────────────────────────────────────────────

class _ContentHeader extends StatelessWidget {
  const _ContentHeader({required this.title, this.actions});

  final String title;
  final List<Widget>? actions;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      height: 48,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      decoration: BoxDecoration(
        color: isDark ? JarvisTheme.surface : Colors.white,
        border: Border(
          bottom: BorderSide(
            color: isDark ? JarvisTheme.border : const Color(0xFFE0E0E8),
          ),
        ),
      ),
      child: Row(
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: isDark
                  ? JarvisTheme.textPrimary
                  : const Color(0xFF1A1A2E),
            ),
          ),
          const Spacer(),
          if (actions != null) ...actions!,
        ],
      ),
    );
  }
}

// ─── Hover icon button helper ───────────────────────────────────────────────

class _HoverIconButton extends StatefulWidget {
  const _HoverIconButton({
    required this.icon,
    required this.size,
    required this.onTap,
    this.tooltip,
  });

  final IconData icon;
  final double size;
  final VoidCallback onTap;
  final String? tooltip;

  @override
  State<_HoverIconButton> createState() => _HoverIconButtonState();
}

class _HoverIconButtonState extends State<_HoverIconButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final color = _hovered
        ? (isDark ? JarvisTheme.textPrimary : const Color(0xFF1A1A2E))
        : (isDark ? JarvisTheme.textTertiary : const Color(0xFF9999AA));

    Widget icon = MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: Padding(
          padding: const EdgeInsets.all(4),
          child: Icon(widget.icon, size: widget.size, color: color),
        ),
      ),
    );

    if (widget.tooltip != null) {
      icon = Tooltip(message: widget.tooltip!, child: icon);
    }
    return icon;
  }
}

// ─── Bottom bar action (reused from original main_shell for mobile) ─────────

class _BottomBarAction extends StatelessWidget {
  const _BottomBarAction({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20, color: color),
            const SizedBox(height: 2),
            Text(label, style: TextStyle(fontSize: 9, color: color)),
          ],
        ),
      ),
    );
  }
}
