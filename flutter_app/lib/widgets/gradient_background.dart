import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// A subtle animated gradient background placed behind [child].
///
/// In dark mode two radial accent-colored glows rotate slowly (60 s full
/// rotation) at very low opacity. In light mode a faint blue tint is used
/// instead.
///
/// An optional [particleColor] enables a layer of slowly drifting dots
/// rendered at 5 % opacity (max 30 particles, 1-3 px radius).
class GradientBackground extends StatefulWidget {
  const GradientBackground({
    super.key,
    required this.child,
    this.particleColor,
  });

  final Widget child;

  /// If non-null, subtle floating particles will be drawn in this color at
  /// 5 % opacity.
  final Color? particleColor;

  @override
  State<GradientBackground> createState() => _GradientBackgroundState();
}

class _GradientBackgroundState extends State<GradientBackground>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 60),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return ListenableBuilder(
      listenable: _controller,
      builder: (context, child) {
        return Stack(
          fit: StackFit.expand,
          children: [
            // Background gradient layer
            CustomPaint(
              painter: _GradientPainter(
                angle: _controller.value * 2 * math.pi,
                isDark: isDark,
              ),
            ),
            // Particle layer (only when a color is provided)
            if (widget.particleColor != null)
              CustomPaint(
                painter: _ParticlePainter(
                  time: _controller.value * 60, // seconds elapsed
                  color: widget.particleColor!,
                ),
              ),
            // Main content
            child!,
          ],
        );
      },
      child: widget.child,
    );
  }
}

// ── Gradient Painter ──────────────────────────────────────────────────────

class _GradientPainter extends CustomPainter {
  _GradientPainter({required this.angle, required this.isDark});

  final double angle;
  final bool isDark;

  @override
  void paint(Canvas canvas, Size size) {
    if (isDark) {
      _paintDark(canvas, size);
    } else {
      _paintLight(canvas, size);
    }
  }

  void _paintDark(Canvas canvas, Size size) {
    final accentColor = CognithorTheme.accent;

    // Top-right glow — slowly orbits.
    final center1 = Offset(
      size.width * 0.75 + math.cos(angle) * size.width * 0.1,
      size.height * 0.2 + math.sin(angle) * size.height * 0.05,
    );
    final paint1 = Paint()
      ..shader =
          RadialGradient(
            colors: [
              accentColor.withValues(alpha: 0.05),
              accentColor.withValues(alpha: 0.0),
            ],
          ).createShader(
            Rect.fromCircle(center: center1, radius: size.width * 0.6),
          );
    canvas.drawRect(Offset.zero & size, paint1);

    // Bottom-left glow — counter-orbits.
    final center2 = Offset(
      size.width * 0.25 + math.cos(angle + math.pi) * size.width * 0.1,
      size.height * 0.8 + math.sin(angle + math.pi) * size.height * 0.05,
    );
    final paint2 = Paint()
      ..shader =
          RadialGradient(
            colors: [
              accentColor.withValues(alpha: 0.03),
              accentColor.withValues(alpha: 0.0),
            ],
          ).createShader(
            Rect.fromCircle(center: center2, radius: size.width * 0.5),
          );
    canvas.drawRect(Offset.zero & size, paint2);
  }

  void _paintLight(Canvas canvas, Size size) {
    // Very subtle blue tint that drifts.
    const blue = Color(0xFF448AFF);
    final center = Offset(
      size.width * 0.5 + math.cos(angle) * size.width * 0.15,
      size.height * 0.3 + math.sin(angle) * size.height * 0.08,
    );
    final paint = Paint()
      ..shader = RadialGradient(
        colors: [blue.withValues(alpha: 0.04), blue.withValues(alpha: 0.0)],
      ).createShader(Rect.fromCircle(center: center, radius: size.width * 0.7));
    canvas.drawRect(Offset.zero & size, paint);
  }

  @override
  bool shouldRepaint(_GradientPainter oldDelegate) =>
      angle != oldDelegate.angle || isDark != oldDelegate.isDark;
}

// ── Particle Painter ──────────────────────────────────────────────────────

/// Seed data for a single particle.
class _Particle {
  _Particle({
    required this.baseX,
    required this.baseY,
    required this.radius,
    required this.phaseX,
    required this.phaseY,
    required this.speedX,
    required this.speedY,
    required this.driftX,
    required this.driftY,
  });

  /// Normalized base position (0..1).
  final double baseX;
  final double baseY;

  /// Dot radius in logical pixels (1-3).
  final double radius;

  /// Phase offsets for sin/cos so particles don't move in sync.
  final double phaseX;
  final double phaseY;

  /// Speed multiplier for oscillation.
  final double speedX;
  final double speedY;

  /// Amplitude of drift in normalized coordinates.
  final double driftX;
  final double driftY;
}

class _ParticlePainter extends CustomPainter {
  _ParticlePainter({required this.time, required this.color});

  /// Elapsed time in seconds.
  final double time;

  /// Base color — will be drawn at 5 % opacity.
  final Color color;

  static const _count = 30;
  static List<_Particle>? _particles;

  static List<_Particle> _generate() {
    final rng = math.Random(123);
    return List.generate(_count, (_) {
      return _Particle(
        baseX: rng.nextDouble(),
        baseY: rng.nextDouble(),
        radius: 1.0 + rng.nextDouble() * 2.0, // 1-3 px
        phaseX: rng.nextDouble() * 2 * math.pi,
        phaseY: rng.nextDouble() * 2 * math.pi,
        speedX: 0.02 + rng.nextDouble() * 0.04, // slow
        speedY: 0.02 + rng.nextDouble() * 0.04,
        driftX: 0.01 + rng.nextDouble() * 0.03, // small amplitude
        driftY: 0.01 + rng.nextDouble() * 0.03,
      );
    });
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;
    _particles ??= _generate();

    final paint = Paint()..color = color.withValues(alpha: 0.05);

    for (final p in _particles!) {
      final x =
          (p.baseX + math.sin(time * p.speedX + p.phaseX) * p.driftX) *
          size.width;
      final y =
          (p.baseY + math.cos(time * p.speedY + p.phaseY) * p.driftY) *
          size.height;
      canvas.drawCircle(Offset(x, y), p.radius, paint);
    }
  }

  @override
  bool shouldRepaint(_ParticlePainter oldDelegate) =>
      oldDelegate.time != time || oldDelegate.color != color;
}
