import 'dart:math';

import 'package:flutter/material.dart';

import 'package:cognithor_ui/widgets/robot_office/furniture.dart';
import 'package:cognithor_ui/widgets/robot_office/robot.dart';
import 'package:cognithor_ui/widgets/robot_office/robot_office_widget.dart';

/// CustomPainter that draws the isometric office scene with furniture,
/// animated robots, pets, and particle effects.
class RobotOfficePainter extends CustomPainter {
  RobotOfficePainter({
    required this.robots,
    required this.furniture,
    required this.elapsed,
    required this.dog,
    required this.cat,
    required this.particles,
  });

  final List<Robot> robots;
  final List<Furniture> furniture;
  final double elapsed;
  final OfficePet dog;
  final OfficePet cat;
  final ParticleSystem particles;

  @override
  void paint(Canvas canvas, Size size) {
    // Background (floor, walls, window) drawn by OfficePainter underneath.
    // Only draw interactive elements here.

    // Draw paw prints first (under everything)
    _drawPawPrints(canvas, size);

    // Furniture drawn by OfficePainter layer underneath.

    // Collect all drawable entities with their Y positions for depth sorting
    final List<_DrawableEntity> entities = [];

    for (final r in robots) {
      entities.add(
        _DrawableEntity(y: r.y, draw: () => _drawRobot(canvas, size, r)),
      );
    }
    entities.add(
      _DrawableEntity(y: dog.y, draw: () => _drawPet(canvas, size, dog)),
    );
    entities.add(
      _DrawableEntity(y: cat.y, draw: () => _drawPet(canvas, size, cat)),
    );

    entities.sort((a, b) => a.y.compareTo(b.y));
    for (final e in entities) {
      e.draw();
    }

    // Draw particles on top
    _drawParticles(canvas, size);
  }

  // ── Robots ──────────────────────────────────────────────────

  void _drawRobot(Canvas canvas, Size size, Robot r) {
    final cx = r.x * size.width;
    final bobAmount = _bobAmountForState(r);
    final bobOffset = sin(r.bobPhase) * bobAmount;
    final cy = r.y * size.height + bobOffset;
    final scale = size.height / 300;

    // Extra animations per state
    double extraXOffset = 0;
    double extraYOffset = 0;

    if (r.state == RobotState.dancing) {
      extraXOffset = sin(r.dancePhase) * 4 * scale;
      extraYOffset = -((sin(r.dancePhase * 2).abs()) * 3 * scale);
    }
    if (r.state == RobotState.celebrating) {
      // Jump up and down
      extraYOffset = -((sin(r.celebratePhase * 8).abs()) * 5 * scale);
    }
    if (r.state == RobotState.stretching) {
      // Slight upward stretch
      extraYOffset = -sin(elapsed * 2) * 3 * scale;
    }

    final drawX = cx + extraXOffset;
    final drawY = cy + extraYOffset;

    canvas.save();
    canvas.translate(drawX, drawY);
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
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        const Rect.fromLTWH(-8, -6, 8, 16),
        const Radius.circular(4),
      ),
      Paint()..color = Colors.white.withValues(alpha: 0.12),
    );

    // System glow (PGE Trinity robots: Planner, Executor, Gatekeeper)
    if (r.isSystem) {
      final glowAlpha = 0.15 + 0.05 * sin(elapsed * 2);
      final glowPaint = Paint()
        ..color = r.color.withValues(alpha: glowAlpha)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
      canvas.drawCircle(const Offset(0, -12), 18, glowPaint);
    }

    // Head
    const headRect = Rect.fromLTWH(-8, -22, 16, 14);
    canvas.drawRRect(
      RRect.fromRectAndRadius(headRect, const Radius.circular(5)),
      Paint()..color = r.color,
    );

    // Eyes
    final blinking = r.blinkTimer < 0.12;
    double eyeH = blinking ? 1.0 : 4.0;

    // Napping: eyes closed
    if (r.state == RobotState.napping) {
      eyeH = 1.0;
    }

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

    // Mouth expressions
    _drawMouth(canvas, r);

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
        Paint()
          ..color = r.eyeColor.withValues(alpha: 0.8 + 0.2 * sin(elapsed * 3)),
      );
    }

    // Arms
    _drawArms(canvas, r, scale);

    // Legs
    _drawLegs(canvas, r);

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
      _drawMsgBubble(
        canvas,
        drawX,
        drawY - 36 * scale,
        r.taskMsg,
        r.color,
        scale,
      );
    }

    // Chat bubble (different style)
    if (r.chatBubbleTimer > 0 && r.chatBubble.isNotEmpty) {
      _drawChatBubble(canvas, drawX, drawY - 40 * scale, r.chatBubble, scale);
    }

    // Emoji pop
    if (r.emojiTimer > 0) {
      final ey = drawY - 42 * scale - (1.5 - r.emojiTimer) * 10;
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
      tp.paint(canvas, Offset(drawX - tp.width / 2, ey));
    }
  }

  double _bobAmountForState(Robot r) {
    switch (r.state) {
      case RobotState.walking:
      case RobotState.carrying:
      case RobotState.coffeeBreak:
      case RobotState.playing:
        return 3.0;
      case RobotState.dancing:
        return 2.0;
      case RobotState.celebrating:
        return 0.5;
      case RobotState.napping:
        return 0.5; // gentle breathing
      default:
        return 1.0;
    }
  }

  void _drawMouth(Canvas canvas, Robot r) {
    switch (r.state) {
      case RobotState.napping:
        // Small O for sleeping
        canvas.drawOval(
          Rect.fromCenter(center: const Offset(0, -11), width: 4, height: 3),
          Paint()
            ..color = r.eyeColor.withValues(alpha: 0.4)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 0.8,
        );
      case RobotState.celebrating:
      case RobotState.dancing:
      case RobotState.highFive:
        // Big smile
        final smilePath = Path()
          ..moveTo(-4, -11)
          ..quadraticBezierTo(0, -7, 4, -11);
        canvas.drawPath(
          smilePath,
          Paint()
            ..color = r.eyeColor.withValues(alpha: 0.6)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.2
            ..strokeCap = StrokeCap.round,
        );
      case RobotState.thinking:
        // Slight frown
        final frownPath = Path()
          ..moveTo(-3, -10)
          ..quadraticBezierTo(0, -12, 3, -10);
        canvas.drawPath(
          frownPath,
          Paint()
            ..color = r.eyeColor.withValues(alpha: 0.4)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1
            ..strokeCap = StrokeCap.round,
        );
      case RobotState.pranking:
        if (r.isPranker) {
          // Mischievous grin
          final grinPath = Path()
            ..moveTo(-5, -12)
            ..quadraticBezierTo(0, -8, 5, -12);
          canvas.drawPath(
            grinPath,
            Paint()
              ..color = r.eyeColor.withValues(alpha: 0.5)
              ..style = PaintingStyle.stroke
              ..strokeWidth = 1.2
              ..strokeCap = StrokeCap.round,
          );
        }
      default:
        // Neutral
        canvas.drawLine(
          const Offset(-3, -11),
          const Offset(3, -11),
          Paint()
            ..color = r.eyeColor.withValues(alpha: 0.3)
            ..strokeWidth = 1
            ..strokeCap = StrokeCap.round,
        );
    }
  }

  void _drawArms(Canvas canvas, Robot r, double scale) {
    final armWave = r.typing ? sin(elapsed * 12) * 4 : 0.0;

    double leftArmEndX = -16;
    double leftArmEndY = 8 + armWave;
    double rightArmEndX = 16;
    double rightArmEndY = 8 - armWave;

    if (r.state == RobotState.stretching) {
      // Arms up
      leftArmEndX = -14;
      leftArmEndY = -20 + sin(elapsed * 2) * 3;
      rightArmEndX = 14;
      rightArmEndY = -20 + sin(elapsed * 2 + 1) * 3;
    } else if (r.state == RobotState.highFive) {
      // One arm up for high five
      rightArmEndX = 18;
      rightArmEndY = -15;
    } else if (r.state == RobotState.dancing) {
      leftArmEndX = -16 + sin(r.dancePhase) * 6;
      leftArmEndY = -5 + sin(r.dancePhase * 1.5) * 8;
      rightArmEndX = 16 + sin(r.dancePhase + 2) * 6;
      rightArmEndY = -5 + sin(r.dancePhase * 1.5 + 2) * 8;
    } else if (r.state == RobotState.coffeeBreak && r.x == r.targetX) {
      // Holding coffee cup
      rightArmEndX = 14;
      rightArmEndY = 0 + sin(elapsed * 1.5) * 2;
    }

    // Left arm
    canvas.drawLine(
      const Offset(-10, -2),
      Offset(leftArmEndX, leftArmEndY),
      Paint()
        ..color = r.color
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );
    // Right arm
    canvas.drawLine(
      const Offset(10, -2),
      Offset(rightArmEndX, rightArmEndY),
      Paint()
        ..color = r.color
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );
  }

  void _drawLegs(Canvas canvas, Robot r) {
    double legPhase = 0;
    if (r.state == RobotState.walking ||
        r.state == RobotState.carrying ||
        r.state == RobotState.playing ||
        r.state == RobotState.coffeeBreak) {
      legPhase = sin(r.bobPhase * 2) * 3;
    }
    if (r.state == RobotState.dancing) {
      legPhase = sin(r.dancePhase * 1.3) * 4;
    }

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

  void _drawChatBubble(
    Canvas canvas,
    double cx,
    double cy,
    String msg,
    double scale,
  ) {
    final tp = TextPainter(
      text: TextSpan(
        text: msg,
        style: TextStyle(
          fontSize: 7 * scale,
          color: Colors.white.withValues(alpha: 0.95),
          fontWeight: FontWeight.w400,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final pw = tp.width + 12 * scale;
    final ph = tp.height + 8 * scale;
    final bubbleRect = RRect.fromRectAndRadius(
      Rect.fromCenter(center: Offset(cx, cy), width: pw, height: ph),
      Radius.circular(6 * scale),
    );

    // White bubble with border
    canvas.drawRRect(
      bubbleRect,
      Paint()..color = const Color(0xFF2a2a50).withValues(alpha: 0.9),
    );
    canvas.drawRRect(
      bubbleRect,
      Paint()
        ..color = const Color(0xFF5a5a80)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 0.8,
    );
    // Pointer triangle
    final triPath = Path()
      ..moveTo(cx - 3 * scale, cy + ph / 2)
      ..lineTo(cx, cy + ph / 2 + 4 * scale)
      ..lineTo(cx + 3 * scale, cy + ph / 2)
      ..close();
    canvas.drawPath(
      triPath,
      Paint()..color = const Color(0xFF2a2a50).withValues(alpha: 0.9),
    );

    tp.paint(canvas, Offset(cx - tp.width / 2, cy - tp.height / 2));
  }

  // ── Pets ────────────────────────────────────────────────────

  void _drawPet(Canvas canvas, Size size, OfficePet pet) {
    final px = pet.x * size.width;
    final py = pet.y * size.height;
    final scale = size.height / 400;

    canvas.save();
    canvas.translate(px, py);
    canvas.scale(scale * pet.facing.toDouble(), scale);

    if (pet.type == PetType.dog) {
      _drawDog(canvas, pet);
    } else {
      _drawCat(canvas, pet);
    }

    canvas.restore();

    // Pet sleeping Z's
    if (pet.petState == PetState.sleeping) {
      final zPhase = (elapsed * 1.5) % 3.0;
      final zy = py - 10 * scale - zPhase * 6;
      final zAlpha = (1.0 - zPhase / 3.0).clamp(0.0, 1.0);
      final zSize = 6 + zPhase * 2;
      final tp = TextPainter(
        text: TextSpan(
          text: 'z',
          style: TextStyle(
            fontSize: zSize * scale,
            color: Colors.white.withValues(alpha: zAlpha * 0.6),
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(px + sin(elapsed * 2) * 3, zy));
    }
  }

  void _drawDog(Canvas canvas, OfficePet pet) {
    // Body (oval)
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(0, 0), width: 20, height: 12),
      Paint()..color = pet.color,
    );
    // Body highlight
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(-2, -2), width: 10, height: 6),
      Paint()..color = Colors.white.withValues(alpha: 0.15),
    );
    // Head
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(10, -5), width: 12, height: 10),
      Paint()..color = pet.color,
    );
    // Eyes
    canvas.drawCircle(const Offset(12, -7), 1.5, Paint()..color = Colors.black);
    // Nose
    canvas.drawCircle(
      const Offset(15, -5),
      1.5,
      Paint()..color = const Color(0xFF3E2723),
    );
    // Floppy ears
    final earPath = Path()
      ..moveTo(7, -9)
      ..quadraticBezierTo(3, -14, 5, -4);
    canvas.drawPath(
      earPath,
      Paint()
        ..color = const Color(0xFF6D4C41)
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );
    // Tail (wagging)
    final tailAngle = sin(pet.animPhase * pet.tailWagSpeed / 3) * 0.5;
    final tailPath = Path()
      ..moveTo(-10, -2)
      ..quadraticBezierTo(
        -16 + sin(tailAngle) * 4,
        -10 + cos(tailAngle) * 3,
        -14 + sin(tailAngle) * 6,
        -14,
      );
    canvas.drawPath(
      tailPath,
      Paint()
        ..color = pet.color
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );
    // Legs
    for (final lx in [-6.0, -2.0, 2.0, 6.0]) {
      final legWobble = pet.petState != PetState.sleeping
          ? sin(pet.animPhase * 2 + lx) * 1.5
          : 0.0;
      canvas.drawLine(
        Offset(lx, 5),
        Offset(lx + legWobble, 10),
        Paint()
          ..color = pet.color.withValues(alpha: 0.8)
          ..strokeWidth = 2
          ..strokeCap = StrokeCap.round,
      );
    }

    // Ball when playing
    if (pet.petState == PetState.playing) {
      canvas.drawCircle(
        Offset(18 + sin(pet.animPhase) * 3, 2),
        3,
        Paint()..color = const Color(0xFFE53935),
      );
    }
  }

  void _drawCat(Canvas canvas, OfficePet pet) {
    // Body (sleeker)
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(0, 0), width: 18, height: 10),
      Paint()..color = pet.color,
    );
    // White belly
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(0, 2), width: 10, height: 5),
      Paint()..color = Colors.white.withValues(alpha: 0.3),
    );
    // Head
    canvas.drawOval(
      Rect.fromCenter(center: const Offset(9, -4), width: 10, height: 9),
      Paint()..color = pet.color,
    );
    // Pointed ears (triangles)
    final leftEar = Path()
      ..moveTo(5, -8)
      ..lineTo(3, -15)
      ..lineTo(8, -9)
      ..close();
    canvas.drawPath(leftEar, Paint()..color = pet.color);
    // Inner ear
    final leftEarInner = Path()
      ..moveTo(5.5, -8.5)
      ..lineTo(4, -13)
      ..lineTo(7.5, -9)
      ..close();
    canvas.drawPath(leftEarInner, Paint()..color = const Color(0xFFE8BBD0));

    final rightEar = Path()
      ..moveTo(12, -8)
      ..lineTo(14, -15)
      ..lineTo(10, -9)
      ..close();
    canvas.drawPath(rightEar, Paint()..color = pet.color);
    final rightEarInner = Path()
      ..moveTo(11.5, -8.5)
      ..lineTo(13.5, -13)
      ..lineTo(10.5, -9)
      ..close();
    canvas.drawPath(rightEarInner, Paint()..color = const Color(0xFFE8BBD0));

    // Eyes
    if (pet.petState == PetState.sleeping) {
      // Closed eyes
      canvas.drawLine(
        const Offset(7, -5),
        const Offset(9, -5),
        Paint()
          ..color = Colors.black
          ..strokeWidth = 1
          ..strokeCap = StrokeCap.round,
      );
      canvas.drawLine(
        const Offset(10, -5),
        const Offset(12, -5),
        Paint()
          ..color = Colors.black
          ..strokeWidth = 1
          ..strokeCap = StrokeCap.round,
      );
    } else {
      // Cat eyes (slitted)
      canvas.drawOval(
        Rect.fromCenter(center: const Offset(7.5, -5), width: 3, height: 4),
        Paint()..color = const Color(0xFF4CAF50),
      );
      canvas.drawOval(
        Rect.fromCenter(center: const Offset(7.5, -5), width: 1, height: 3.5),
        Paint()..color = Colors.black,
      );
      canvas.drawOval(
        Rect.fromCenter(center: const Offset(11.5, -5), width: 3, height: 4),
        Paint()..color = const Color(0xFF4CAF50),
      );
      canvas.drawOval(
        Rect.fromCenter(center: const Offset(11.5, -5), width: 1, height: 3.5),
        Paint()..color = Colors.black,
      );
    }
    // Nose
    canvas.drawCircle(
      const Offset(9.5, -3),
      1,
      Paint()..color = const Color(0xFFE8BBD0),
    );
    // Whiskers
    for (final side in [-1.0, 1.0]) {
      canvas.drawLine(
        Offset(9.5 + side * 2, -3),
        Offset(9.5 + side * 8, -4),
        Paint()
          ..color = Colors.white.withValues(alpha: 0.5)
          ..strokeWidth = 0.5,
      );
      canvas.drawLine(
        Offset(9.5 + side * 2, -2.5),
        Offset(9.5 + side * 7, -1.5),
        Paint()
          ..color = Colors.white.withValues(alpha: 0.5)
          ..strokeWidth = 0.5,
      );
    }

    // Curved tail
    final tailCurve = sin(pet.animPhase * 1.5) * 0.3;
    final tailPath = Path()
      ..moveTo(-9, -1)
      ..cubicTo(
        -14,
        -3 + tailCurve * 5,
        -16,
        -8 + tailCurve * 8,
        -13,
        -12 + tailCurve * 4,
      );
    canvas.drawPath(
      tailPath,
      Paint()
        ..color = pet.color
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );

    // Legs
    for (final lx in [-4.0, 0.0, 4.0, 7.0]) {
      final legWobble = pet.petState != PetState.sleeping
          ? sin(pet.animPhase * 2 + lx) * 1
          : 0.0;
      canvas.drawLine(
        Offset(lx, 4),
        Offset(lx + legWobble, 9),
        Paint()
          ..color = pet.color.withValues(alpha: 0.8)
          ..strokeWidth = 1.8
          ..strokeCap = StrokeCap.round,
      );
    }

    // Washing face animation
    if (pet.petState == PetState.washingFace) {
      final pawX = 9 + sin(pet.animPhase * 4) * 3;
      final pawY = -5 + cos(pet.animPhase * 4) * 2;
      canvas.drawCircle(Offset(pawX, pawY), 2.5, Paint()..color = pet.color);
    }
  }

  // ── Paw Prints ──────────────────────────────────────────────

  void _drawPawPrints(Canvas canvas, Size size) {
    for (final p in particles.particles) {
      if (p.type != ParticleType.pawPrint) continue;
      final px = p.x * size.width;
      final py = p.y * size.height;
      final alpha = (p.life / p.maxLife).clamp(0.0, 1.0) * 0.3;
      final paint = Paint()..color = p.color.withValues(alpha: alpha);
      // Main pad
      canvas.drawOval(
        Rect.fromCenter(center: Offset(px, py), width: 4, height: 3),
        paint,
      );
      // Toe beans
      for (int t = 0; t < 3; t++) {
        final tx = px - 2 + t * 2;
        final ty = py - 2.5;
        canvas.drawCircle(Offset(tx, ty), 1, paint);
      }
    }
  }

  // ── Particles ───────────────────────────────────────────────

  void _drawParticles(Canvas canvas, Size size) {
    for (final p in particles.particles) {
      if (p.type == ParticleType.pawPrint) continue; // already drawn

      final px = p.x * size.width;
      final py = p.y * size.height;
      final alpha = (p.life / p.maxLife).clamp(0.0, 1.0);

      switch (p.type) {
        case ParticleType.spark:
          canvas.drawCircle(
            Offset(px, py),
            p.size * alpha,
            Paint()..color = p.color.withValues(alpha: alpha * 0.8),
          );
          // Glow
          canvas.drawCircle(
            Offset(px, py),
            p.size * alpha * 2,
            Paint()
              ..color = p.color.withValues(alpha: alpha * 0.2)
              ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
          );
        case ParticleType.confetti:
          canvas.save();
          canvas.translate(px, py);
          canvas.rotate(p.rotation);
          canvas.drawRect(
            Rect.fromCenter(
              center: Offset.zero,
              width: p.size,
              height: p.size * 0.6,
            ),
            Paint()..color = p.color.withValues(alpha: alpha * 0.9),
          );
          canvas.restore();
        case ParticleType.text:
          final tp = TextPainter(
            text: TextSpan(
              text: p.text ?? '?',
              style: TextStyle(
                fontSize: p.size,
                color: p.color.withValues(alpha: alpha * 0.8),
                fontWeight: FontWeight.bold,
              ),
            ),
            textDirection: TextDirection.ltr,
          )..layout();
          tp.paint(canvas, Offset(px - tp.width / 2, py - tp.height / 2));
        case ParticleType.dataPacket:
          // Glowing dot
          canvas.drawCircle(
            Offset(px, py),
            p.size,
            Paint()..color = p.color.withValues(alpha: alpha * 0.9),
          );
          // Trail glow
          canvas.drawCircle(
            Offset(px, py),
            p.size * 2.5,
            Paint()
              ..color = p.color.withValues(alpha: alpha * 0.15)
              ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4),
          );
          // Small trail dots
          for (int t = 1; t <= 4; t++) {
            final tt = (p.progress - t * 0.03).clamp(0.0, 1.0);
            final midX = (p.startX + p.endX) / 2;
            final midY = min(p.startY, p.endY) - 0.08;
            final trailX =
                (1 - tt) * (1 - tt) * p.startX +
                2 * (1 - tt) * tt * midX +
                tt * tt * p.endX;
            final trailY =
                (1 - tt) * (1 - tt) * p.startY +
                2 * (1 - tt) * tt * midY +
                tt * tt * p.endY;
            final trailAlpha = alpha * (1.0 - t / 5.0);
            canvas.drawCircle(
              Offset(trailX * size.width, trailY * size.height),
              p.size * 0.5,
              Paint()..color = p.color.withValues(alpha: trailAlpha * 0.5),
            );
          }
        case ParticleType.fallingItem:
          canvas.save();
          canvas.translate(px, py);
          canvas.rotate(p.rotation);
          canvas.drawRect(
            Rect.fromCenter(
              center: Offset.zero,
              width: p.size,
              height: p.size * 0.7,
            ),
            Paint()..color = p.color.withValues(alpha: alpha * 0.7),
          );
          canvas.restore();
        case ParticleType.pawPrint:
          break; // handled separately
      }
    }
  }

  // ── Repaint ─────────────────────────────────────────────────

  @override
  bool shouldRepaint(covariant RobotOfficePainter oldDelegate) => true;
}

/// Helper class for depth-sorted drawing.
class _DrawableEntity {
  _DrawableEntity({required this.y, required this.draw});
  final double y;
  final VoidCallback draw;
}
