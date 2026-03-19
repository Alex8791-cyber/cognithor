import 'dart:math';

import 'package:flutter/material.dart';

import 'package:jarvis_ui/widgets/robot_office/furniture.dart';
import 'package:jarvis_ui/widgets/robot_office/robot.dart';

/// CustomPainter that draws the isometric office scene with furniture and
/// animated robots.
class RobotOfficePainter extends CustomPainter {
  RobotOfficePainter({
    required this.robots,
    required this.furniture,
    required this.elapsed,
  });

  final List<Robot> robots;
  final List<Furniture> furniture;
  final double elapsed;

  @override
  void paint(Canvas canvas, Size size) {
    _drawFloor(canvas, size);
    _drawGrid(canvas, size);

    // Draw furniture
    for (final f in furniture) {
      _drawFurniture(canvas, size, f);
    }

    // Sort robots by y so overlapping looks correct
    final sorted = List<Robot>.from(robots)..sort((a, b) => a.y.compareTo(b.y));
    for (final r in sorted) {
      _drawRobot(canvas, size, r);
    }
  }

  // ── Floor ───────────────────────────────────────────────────

  void _drawFloor(Canvas canvas, Size size) {
    final paint = Paint()
      ..shader = const LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [Color(0xFF0e0e1a), Color(0xFF141428)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), paint);
  }

  void _drawGrid(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF1a1a30)
      ..strokeWidth = 0.5;

    const step = 30.0;
    for (var x = 0.0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (var y = 0.0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  // ── Furniture ───────────────────────────────────────────────

  void _drawFurniture(Canvas canvas, Size size, Furniture f) {
    final rect = Rect.fromLTWH(
      f.x * size.width,
      f.y * size.height,
      f.w * size.width,
      f.h * size.height,
    );

    switch (f.type) {
      case 'desk':
        _drawDesk(canvas, rect);
      case 'server':
        _drawServer(canvas, rect);
      case 'board':
        _drawBoard(canvas, rect);
      case 'plant':
        _drawPlant(canvas, rect);
      case 'coffee':
        _drawCoffee(canvas, rect);
    }
  }

  void _drawDesk(Canvas canvas, Rect rect) {
    // Desktop surface
    final topPaint = Paint()..color = const Color(0xFF2a2a45);
    final borderPaint = Paint()
      ..color = const Color(0xFF3a3a58)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(3)),
      topPaint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(3)),
      borderPaint,
    );

    // Monitor on desk
    final monW = rect.width * 0.35;
    final monH = rect.height * 0.5;
    final monRect = Rect.fromCenter(
      center: Offset(rect.center.dx, rect.top + monH * 0.4),
      width: monW,
      height: monH,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(monRect, const Radius.circular(2)),
      Paint()..color = const Color(0xFF1a1a2e),
    );
    // Screen glow
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        monRect.deflate(2),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFF00d4ff).withValues(alpha: 0.15),
    );
  }

  void _drawServer(Canvas canvas, Rect rect) {
    final paint = Paint()..color = const Color(0xFF1e1e38);
    final borderPaint = Paint()
      ..color = const Color(0xFF3a3a58)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(4)),
      paint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(4)),
      borderPaint,
    );

    // Blinking lights
    const lightCount = 4;
    for (var i = 0; i < lightCount; i++) {
      final ly = rect.top + rect.height * (0.2 + 0.2 * i);
      final phase = elapsed * 2.0 + i * 1.3;
      final on = sin(phase) > 0;
      canvas.drawCircle(
        Offset(rect.center.dx, ly),
        2.5,
        Paint()
          ..color = on
              ? const Color(0xFF00e676).withValues(alpha: 0.9)
              : const Color(0xFF333350),
      );
    }
  }

  void _drawBoard(Canvas canvas, Rect rect) {
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(3)),
      Paint()..color = const Color(0xFF1a2a3a),
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(3)),
      Paint()
        ..color = const Color(0xFF00d4ff).withValues(alpha: 0.2)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1,
    );
    // Text lines
    final linePaint = Paint()
      ..color = const Color(0xFF00d4ff).withValues(alpha: 0.25)
      ..strokeWidth = 1.5;
    for (var i = 0; i < 3; i++) {
      final ly = rect.top + rect.height * (0.3 + 0.2 * i);
      final lx1 = rect.left + rect.width * 0.15;
      final lx2 = rect.left + rect.width * (0.6 + i * 0.1);
      canvas.drawLine(Offset(lx1, ly), Offset(lx2, ly), linePaint);
    }
  }

  void _drawPlant(Canvas canvas, Rect rect) {
    // Pot
    final potRect = Rect.fromLTWH(
      rect.left + rect.width * 0.15,
      rect.top + rect.height * 0.5,
      rect.width * 0.7,
      rect.height * 0.5,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(potRect, const Radius.circular(2)),
      Paint()..color = const Color(0xFF5a3a2a),
    );
    // Leaves
    final cx = rect.center.dx;
    final cy = rect.top + rect.height * 0.35;
    for (var i = -1; i <= 1; i++) {
      final leafPath = Path()
        ..moveTo(cx, cy + 5)
        ..quadraticBezierTo(cx + i * 8, cy - 8, cx + i * 3, cy - 12);
      canvas.drawPath(
        leafPath,
        Paint()
          ..color = const Color(0xFF10b981).withValues(alpha: 0.7)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..strokeCap = StrokeCap.round,
      );
    }
  }

  void _drawCoffee(Canvas canvas, Rect rect) {
    // Machine body
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(3)),
      Paint()..color = const Color(0xFF2a2538),
    );
    // Steam
    final phase = elapsed * 1.5;
    final steamPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.08 + 0.05 * sin(phase))
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    final cx = rect.center.dx;
    for (var i = 0; i < 2; i++) {
      final sy = rect.top - 3 - i * 5;
      final sx = cx + sin(phase + i) * 3;
      canvas.drawLine(
        Offset(sx, sy),
        Offset(sx + sin(phase + i + 1) * 2, sy - 5),
        steamPaint,
      );
    }
  }

  // ── Robots ──────────────────────────────────────────────────

  void _drawRobot(Canvas canvas, Size size, Robot r) {
    final cx = r.x * size.width;
    final bobOffset = sin(r.bobPhase) * (r.state == RobotState.walking ? 3.0 : 1.0);
    final cy = r.y * size.height + bobOffset;
    final scale = size.height / 300; // base scale

    canvas.save();
    canvas.translate(cx, cy);
    canvas.scale(scale * r.facing.toDouble(), scale);

    // Shadow
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(0, 18), width: 24, height: 8),
      Paint()..color = Colors.black.withValues(alpha: 0.3),
    );

    // Body
    const bodyRect = Rect.fromLTWH(-10, -8, 20, 24);
    canvas.drawRRect(
      RRect.fromRectAndRadius(bodyRect, const Radius.circular(6)),
      Paint()..color = r.color,
    );
    // Body highlight
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        const Rect.fromLTWH(-8, -6, 8, 16),
        const Radius.circular(4),
      ),
      Paint()..color = Colors.white.withValues(alpha: 0.12),
    );

    // Head
    const headRect = Rect.fromLTWH(-8, -22, 16, 14);
    canvas.drawRRect(
      RRect.fromRectAndRadius(headRect, const Radius.circular(5)),
      Paint()..color = r.color,
    );

    // Eyes
    final blinking = r.blinkTimer < 0.12;
    final eyeH = blinking ? 1.0 : 4.0;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: const Offset(-3, -16), width: 4, height: eyeH),
        const Radius.circular(2),
      ),
      Paint()..color = r.eyeColor,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: const Offset(3, -16), width: 4, height: eyeH),
        const Radius.circular(2),
      ),
      Paint()..color = r.eyeColor,
    );

    // Antenna
    if (r.hasAntenna) {
      final antennaWobble = sin(r.bobPhase * 2) * 2;
      canvas.drawLine(
        const Offset(0, -22),
        Offset(antennaWobble, -30),
        Paint()
          ..color = r.color
          ..strokeWidth = 1.5
          ..strokeCap = StrokeCap.round,
      );
      canvas.drawCircle(
        Offset(antennaWobble, -31),
        2.5,
        Paint()..color = r.eyeColor.withValues(alpha: 0.8 + 0.2 * sin(elapsed * 3)),
      );
    }

    // Arms
    final armWave = r.typing ? sin(elapsed * 12) * 4 : 0.0;
    // Left arm
    canvas.drawLine(
      const Offset(-10, -2),
      Offset(-16, 8 + armWave),
      Paint()
        ..color = r.color
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );
    // Right arm
    canvas.drawLine(
      const Offset(10, -2),
      Offset(16, 8 - armWave),
      Paint()
        ..color = r.color
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );

    // Legs
    final legPhase = r.state == RobotState.walking ? sin(r.bobPhase * 2) * 3 : 0.0;
    canvas.drawLine(
      const Offset(-4, 16),
      Offset(-5 + legPhase, 24),
      Paint()
        ..color = r.color.withValues(alpha: 0.8)
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );
    canvas.drawLine(
      const Offset(4, 16),
      Offset(5 - legPhase, 24),
      Paint()
        ..color = r.color.withValues(alpha: 0.8)
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );

    // Carrying indicator (small box)
    if (r.carrying) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          const Rect.fromLTWH(12, -6, 8, 8),
          const Radius.circular(2),
        ),
        Paint()..color = const Color(0xFFf59e0b),
      );
    }

    canvas.restore();

    // Task message bubble (drawn un-flipped)
    if (r.msgTimer > 0) {
      _drawMsgBubble(canvas, cx, cy - 36 * scale, r.taskMsg, r.color, scale);
    }

    // Emoji pop
    if (r.emojiTimer > 0) {
      final ey = cy - 42 * scale - (1.5 - r.emojiTimer) * 10;
      final opacity = (r.emojiTimer / 1.5).clamp(0.0, 1.0);
      final tp = TextPainter(
        text: TextSpan(
          text: r.emoji,
          style: TextStyle(
            fontSize: 14 * scale,
            color: Colors.white.withValues(alpha: opacity),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(cx - tp.width / 2, ey));
    }
  }

  void _drawMsgBubble(
    Canvas canvas,
    double cx,
    double cy,
    String msg,
    Color accent,
    double scale,
  ) {
    final tp = TextPainter(
      text: TextSpan(
        text: msg,
        style: TextStyle(
          fontSize: 9 * scale,
          color: Colors.white.withValues(alpha: 0.9),
          fontWeight: FontWeight.w500,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final pw = tp.width + 10 * scale;
    final ph = tp.height + 6 * scale;
    final bubbleRect = RRect.fromRectAndRadius(
      Rect.fromCenter(center: Offset(cx, cy), width: pw, height: ph),
      Radius.circular(4 * scale),
    );

    canvas.drawRRect(
      bubbleRect,
      Paint()..color = accent.withValues(alpha: 0.75),
    );
    tp.paint(canvas, Offset(cx - tp.width / 2, cy - tp.height / 2));
  }

  // ── Repaint ─────────────────────────────────────────────────

  @override
  bool shouldRepaint(covariant RobotOfficePainter oldDelegate) => true;
}
