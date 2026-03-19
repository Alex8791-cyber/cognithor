import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

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
    final color = widget.tint ?? JarvisTheme.accent;
    final isHovered = _hovered && widget.glowOnHover;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: JarvisTheme.animDuration,
          curve: JarvisTheme.animCurve,
          decoration: BoxDecoration(
            color: color.withValues(alpha: isHovered ? 0.08 : 0.04),
            borderRadius: BorderRadius.circular(widget.borderRadius),
            border: Border.all(
              color: color.withValues(alpha: isHovered ? 0.35 : 0.12),
            ),
            boxShadow: isHovered
                ? [
                    BoxShadow(
                      color: color.withValues(alpha: 0.15),
                      blurRadius: 20,
                      spreadRadius: -2,
                    ),
                  ]
                : null,
          ),
          padding: widget.padding ?? const EdgeInsets.all(16),
          child: widget.child,
        ),
      ),
    );
  }
}
