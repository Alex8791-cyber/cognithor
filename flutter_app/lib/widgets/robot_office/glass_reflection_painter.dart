import 'package:flutter/material.dart';

/// Draws a subtle glass reflection overlay for the "looking through glass" effect.
class GlassReflectionPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final reflectionPaint = Paint()
      ..shader = LinearGradient(
        begin: const Alignment(-0.8, -0.8),
        end: const Alignment(0.8, 0.8),
        colors: [
          Colors.white.withValues(alpha: 0.0),
          Colors.white.withValues(alpha: 0.06),
          Colors.white.withValues(alpha: 0.0),
        ],
        stops: const [0.3, 0.5, 0.7],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      reflectionPaint,
    );

    final highlightPaint = Paint()
      ..shader = RadialGradient(
        center: const Alignment(-0.9, -0.9),
        radius: 0.6,
        colors: [
          Colors.white.withValues(alpha: 0.08),
          Colors.white.withValues(alpha: 0.0),
        ],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width * 0.4, size.height * 0.4),
      highlightPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
