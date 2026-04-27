import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// Lightweight alternative to GlassPanel for use in list items and screens.
/// Uses solid semi-transparent backgrounds instead of BackdropFilter,
/// which can cause rendering issues on Flutter web.
class NeonCard extends StatefulWidget {
  const NeonCard({
    super.key,
    required this.child,
    this.tint,
    this.borderRadius = 16,
    this.padding,
    this.glowOnHover = false,
    this.onTap,
  });

  final Widget child;
  final Color? tint;
  final double borderRadius;
  final EdgeInsets? padding;
  final bool glowOnHover;
  final VoidCallback? onTap;

  @override
  State<NeonCard> createState() => _NeonCardState();
}

class _NeonCardState extends State<NeonCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final color = widget.tint ?? CognithorTheme.accent;
    final isHovered = _hovered && widget.glowOnHover;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final bgColor = isDark
        ? color.withValues(alpha: isHovered ? 0.14 : 0.08)
        : isHovered
        ? color.withValues(alpha: 0.08)
        : Theme.of(context).cardColor;

    final borderColor = isDark
        ? color.withValues(alpha: isHovered ? 0.50 : 0.22)
        : isHovered
        ? color.withValues(alpha: 0.40)
        : Theme.of(context).dividerColor;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(widget.borderRadius),
          clipBehavior: Clip.antiAlias,
          child: AnimatedContainer(
            duration: CognithorTheme.animDuration,
            curve: CognithorTheme.animCurve,
            decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(widget.borderRadius),
              border: Border.all(color: borderColor, width: 1.0),
              boxShadow: isHovered
                  ? [
                      BoxShadow(
                        color: color.withValues(alpha: isDark ? 0.25 : 0.12),
                        blurRadius: 28,
                        spreadRadius: -2,
                      ),
                    ]
                  : null,
            ),
            padding: widget.padding ?? const EdgeInsets.all(16),
            child: widget.child,
          ),
        ),
      ),
    );
  }
}
