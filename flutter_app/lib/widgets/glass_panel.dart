import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

/// Glassmorphism panel — frosted glass effect with neon-tinted border.
///
/// Use this as the primary card/container widget throughout the app.
/// Replaces JarvisCard for the new Sci-Fi aesthetic.
class GlassPanel extends StatefulWidget {
  const GlassPanel({
    super.key,
    required this.child,
    this.tint, // section color tint (default: accent/violet)
    this.borderRadius = 16,
    this.blur = 16,
    this.padding,
    this.glowOnHover = false,
    this.onTap,
  });

  final Widget child;
  final Color? tint;
  final double borderRadius;
  final double blur;
  final EdgeInsets? padding;
  final bool glowOnHover;
  final VoidCallback? onTap;

  @override
  State<GlassPanel> createState() => _GlassPanelState();
}

class _GlassPanelState extends State<GlassPanel> {
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
          child: ClipRRect(
            borderRadius: BorderRadius.circular(widget.borderRadius),
            child: BackdropFilter(
              filter: ImageFilter.blur(
                sigmaX: widget.blur,
                sigmaY: widget.blur,
              ),
              child: Container(
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.04),
                  borderRadius: BorderRadius.circular(widget.borderRadius),
                ),
                padding: widget.padding ?? const EdgeInsets.all(16),
                child: widget.child,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
