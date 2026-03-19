import 'package:flutter/material.dart';

/// Wraps a child widget with a neon glow effect.
///
/// Use for buttons, icons, active indicators, and interactive elements
/// to give them the Sci-Fi neon aesthetic.
class NeonGlow extends StatefulWidget {
  const NeonGlow({
    super.key,
    required this.child,
    required this.color,
    this.intensity = 0.3,
    this.pulse = false, // breathing animation
    this.blurRadius = 12,
    this.spreadRadius = 0,
  });

  final Widget child;
  final Color color;
  final double intensity;
  final bool pulse;
  final double blurRadius;
  final double spreadRadius;

  @override
  State<NeonGlow> createState() => _NeonGlowState();
}

class _NeonGlowState extends State<NeonGlow>
    with SingleTickerProviderStateMixin {
  AnimationController? _pulseCtrl;

  @override
  void initState() {
    super.initState();
    if (widget.pulse) {
      _pulseCtrl = AnimationController(
        vsync: this,
        duration: const Duration(seconds: 2),
      )..repeat(reverse: true);
    }
  }

  @override
  void dispose() {
    _pulseCtrl?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_pulseCtrl != null) {
      return ListenableBuilder(
        listenable: _pulseCtrl!,
        builder: (context, child) => _buildGlow(_pulseCtrl!.value),
        child: widget.child,
      );
    }
    return _buildGlow(1.0);
  }

  Widget _buildGlow(double factor) {
    final effectiveIntensity = widget.intensity * (0.5 + factor * 0.5);
    return Container(
      decoration: BoxDecoration(
        boxShadow: [
          BoxShadow(
            color: widget.color.withValues(alpha: effectiveIntensity),
            blurRadius: widget.blurRadius,
            spreadRadius: widget.spreadRadius,
          ),
        ],
      ),
      child: widget.child,
    );
  }
}
