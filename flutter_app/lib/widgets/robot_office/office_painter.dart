import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

// ──────────────────────────────────────────────────────────────────────
// Robot Office – Animated CustomPainter (centerpiece visualization)
//
// Draws a fully procedural office scene at 60 fps with 4 AI-agent robots,
// desks, a server rack, kanban board, coffee machine, plants, window
// that reflects the real time-of-day, data-stream particles, and more.
// ──────────────────────────────────────────────────────────────────────

// ── Data Models ──────────────────────────────────────────────────────

class RobotType {
  const RobotType({
    required this.id,
    required this.name,
    required this.color,
    required this.eyeColor,
    required this.role,
    this.antenna = false,
  });

  final String id;
  final String name;
  final Color color;
  final Color eyeColor;
  final String role;
  final bool antenna;
}

enum RobotState { idle, walking, working, carrying }

class Robot {
  Robot({
    required this.type,
    required this.x,
    required this.y,
    this.state = RobotState.idle,
    this.targetX = 0,
    this.targetY = 0,
    this.stateTimer = 0,
    this.nextSwitch = 3.0,
    this.blinkTimer = 0,
    this.speechText = '',
    this.speechTimer = 0,
    this.floatingEmoji = '',
    this.emojiTimer = 0,
  });

  final RobotType type;
  double x;
  double y;
  RobotState state;
  double targetX;
  double targetY;
  double stateTimer;
  double nextSwitch;
  double blinkTimer;
  String speechText;
  double speechTimer;
  String floatingEmoji;
  double emojiTimer;
}

// ── Predefined Robot Types ───────────────────────────────────────────

const _planner = RobotType(
  id: 'planner',
  name: 'Planner',
  color: Color(0xFF6366f1),
  eyeColor: Color(0xFFa5b4fc),
  role: 'Strategieplanung',
  antenna: true,
);

const _executor = RobotType(
  id: 'executor',
  name: 'Executor',
  color: Color(0xFF10b981),
  eyeColor: Color(0xFF6ee7b7),
  role: 'Aufgabenausführung',
);

const _researcher = RobotType(
  id: 'researcher',
  name: 'Researcher',
  color: Color(0xFFf59e0b),
  eyeColor: Color(0xFFfcd34d),
  role: 'Wissensrecherche',
  antenna: true,
);

const _gatekeeper = RobotType(
  id: 'gatekeeper',
  name: 'Gatekeeper',
  color: Color(0xFFef4444),
  eyeColor: Color(0xFFfca5a5),
  role: 'Sicherheitsprüfung',
);

// ── Office Landmark Positions (normalized 0..1) ──────────────────────

const _desk1 = Offset(0.18, 0.55);
const _desk2 = Offset(0.45, 0.52);
const _desk3 = Offset(0.72, 0.55);
const _server = Offset(0.90, 0.40);
const _kanban = Offset(0.06, 0.35);
const _coffee = Offset(0.55, 0.35);

// ── Office Painter ───────────────────────────────────────────────────

class OfficePainter extends CustomPainter {
  OfficePainter({
    required this.robots,
    required this.time,
    required this.isRunning,
    required this.brightness,
  });

  final List<Robot> robots;
  final double time; // elapsed seconds
  final bool isRunning;
  final Brightness brightness;

  // ── Entry Point ────────────────────────────────────────────────────

  @override
  void paint(Canvas canvas, Size size) {
    _drawCeiling(canvas, size);
    _drawBackWall(canvas, size);
    _drawWindow(canvas, size);
    _drawWallSign(canvas, size);
    _drawWhiteboard(canvas, size);
    _drawKanbanBoard(canvas, size);
    _drawCeilingLights(canvas, size);
    _drawFloor(canvas, size);
    _drawLightBeams(canvas, size);
    _drawCables(canvas, size);
    _drawDesks(canvas, size);
    _drawServerRack(canvas, size);
    _drawCoffeeMachine(canvas, size);
    _drawPlants(canvas, size);

    // Sort robots by Y for depth ordering
    final sorted = [...robots]..sort((a, b) => a.y.compareTo(b.y));
    for (final robot in sorted) {
      _drawRobot(canvas, robot, size);
    }
    if (isRunning) _drawDataStreams(canvas, size);
  }

  @override
  bool shouldRepaint(covariant OfficePainter old) => true;

  // ── Helpers ────────────────────────────────────────────────────────

  bool get _isDark => brightness == Brightness.dark;

  Color _wallColor() => _isDark ? const Color(0xFF1a1a2e) : const Color(0xFFe8e8f0);
  Color _floorBase() => _isDark ? const Color(0xFF22223a) : const Color(0xFFd0d0dc);
  Color _floorAlt() => _isDark ? const Color(0xFF2a2a44) : const Color(0xFFdcdce8);
  Color _furnitureDark() => _isDark ? const Color(0xFF2c2c44) : const Color(0xFFb8b8c8);
  Color _furnitureLight() => _isDark ? const Color(0xFF3a3a56) : const Color(0xFFcacad8);
  Color _textCol() => _isDark ? JarvisTheme.textPrimary : const Color(0xFF1A1A2E);

  double _osc(double speed, [double phase = 0]) => sin(time * speed + phase);

  // ── Ceiling ────────────────────────────────────────────────────────

  void _drawCeiling(Canvas canvas, Size s) {
    final rect = Rect.fromLTWH(0, 0, s.width, s.height * 0.18);
    final paint = Paint()
      ..shader = ui.Gradient.linear(
        rect.topCenter,
        rect.bottomCenter,
        [
          _isDark ? const Color(0xFF12122a) : const Color(0xFFd8d8e8),
          _wallColor(),
        ],
      );
    canvas.drawRect(rect, paint);
  }

  // ── Back Wall ──────────────────────────────────────────────────────

  void _drawBackWall(Canvas canvas, Size s) {
    final wallTop = s.height * 0.18;
    final wallBottom = s.height * 0.45;
    final rect = Rect.fromLTWH(0, wallTop, s.width, wallBottom - wallTop);
    final paint = Paint()
      ..shader = ui.Gradient.linear(
        rect.topCenter,
        rect.bottomCenter,
        [_wallColor(), _wallColor().withValues(alpha: 0.95)],
      );
    canvas.drawRect(rect, paint);

    // subtle baseboard
    final baseboard = Rect.fromLTWH(0, wallBottom - 4, s.width, 4);
    canvas.drawRect(
      baseboard,
      Paint()..color = _furnitureDark().withValues(alpha: 0.6),
    );
  }

  // ── Window (time-of-day aware) ─────────────────────────────────────

  void _drawWindow(Canvas canvas, Size s) {
    final hour = DateTime.now().hour + DateTime.now().minute / 60.0;
    final wx = s.width * 0.32;
    final wy = s.height * 0.06;
    final ww = s.width * 0.36;
    final wh = s.height * 0.30;
    final windowRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(wx, wy, ww, wh),
      const Radius.circular(4),
    );

    // Frame
    final framePaint = Paint()
      ..color = _isDark ? const Color(0xFF3a3a56) : const Color(0xFF8888a0)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;

    // Sky gradient based on time of day
    List<Color> skyColors;
    if (hour >= 6 && hour < 12) {
      // Morning
      final t = (hour - 6) / 6;
      skyColors = [
        Color.lerp(const Color(0xFFff9a56), const Color(0xFF56b4ff), t)!,
        Color.lerp(const Color(0xFFffce56), const Color(0xFF87ceeb), t)!,
      ];
    } else if (hour >= 12 && hour < 18) {
      // Afternoon
      skyColors = [const Color(0xFF4a90d9), const Color(0xFF87ceeb)];
    } else if (hour >= 18 && hour < 22) {
      // Sunset
      final t = (hour - 18) / 4;
      skyColors = [
        Color.lerp(const Color(0xFFff6b35), const Color(0xFF1a1a4e), t)!,
        Color.lerp(const Color(0xFFff9a56), const Color(0xFF2a2a5e), t)!,
      ];
    } else {
      // Night (22-6)
      skyColors = [const Color(0xFF0a0a2e), const Color(0xFF1a1a4e)];
    }

    final skyPaint = Paint()
      ..shader = ui.Gradient.linear(
        Offset(wx, wy),
        Offset(wx, wy + wh),
        skyColors,
      );

    canvas.save();
    canvas.clipRRect(windowRect);
    canvas.drawRect(Rect.fromLTWH(wx, wy, ww, wh), skyPaint);

    // Sun or moon
    if (hour >= 6 && hour < 22) {
      _drawSun(canvas, wx, wy, ww, wh, hour);
    } else {
      _drawMoon(canvas, wx, wy, ww, wh, hour);
      _drawStars(canvas, wx, wy, ww, wh);
    }
    canvas.restore();

    // Window frame
    canvas.drawRRect(windowRect, framePaint);

    // Cross dividers
    final divPaint = Paint()
      ..color = framePaint.color
      ..strokeWidth = 2;
    canvas.drawLine(
      Offset(wx + ww / 2, wy),
      Offset(wx + ww / 2, wy + wh),
      divPaint,
    );
    canvas.drawLine(
      Offset(wx, wy + wh / 2),
      Offset(wx + ww, wy + wh / 2),
      divPaint,
    );
  }

  void _drawSun(Canvas canvas, double wx, double wy, double ww, double wh, double hour) {
    // Sun position — arc across window
    double t;
    if (hour < 12) {
      t = (hour - 6) / 12; // 0 at 6am, 0.5 at 12
    } else {
      t = (hour - 6) / 16; // continues toward 1 at 22
    }
    final sunX = wx + ww * 0.1 + ww * 0.8 * t;
    final sunY = wy + wh * 0.8 - sin(t * pi) * wh * 0.6;
    final sunR = ww * 0.06;

    // Glow
    final glowPaint = Paint()
      ..shader = ui.Gradient.radial(
        Offset(sunX, sunY),
        sunR * 4,
        [
          const Color(0x66FFD700),
          const Color(0x00FFD700),
        ],
      );
    canvas.drawCircle(Offset(sunX, sunY), sunR * 4, glowPaint);

    // Sun body
    canvas.drawCircle(
      Offset(sunX, sunY),
      sunR,
      Paint()..color = const Color(0xFFFFD700),
    );

    // Rays
    final rayPaint = Paint()
      ..color = const Color(0x44FFD700)
      ..strokeWidth = 1.5;
    for (int i = 0; i < 8; i++) {
      final angle = i * pi / 4 + time * 0.3;
      final inner = sunR * 1.3;
      final outer = sunR * 2.0 + _osc(2, i.toDouble()) * sunR * 0.3;
      canvas.drawLine(
        Offset(sunX + cos(angle) * inner, sunY + sin(angle) * inner),
        Offset(sunX + cos(angle) * outer, sunY + sin(angle) * outer),
        rayPaint,
      );
    }
  }

  void _drawMoon(Canvas canvas, double wx, double wy, double ww, double wh, double hour) {
    final moonX = wx + ww * 0.65;
    final moonY = wy + wh * 0.30;
    final moonR = ww * 0.055;

    // Moon glow
    final glowPaint = Paint()
      ..shader = ui.Gradient.radial(
        Offset(moonX, moonY),
        moonR * 5,
        [
          const Color(0x33C0C0FF),
          const Color(0x00C0C0FF),
        ],
      );
    canvas.drawCircle(Offset(moonX, moonY), moonR * 5, glowPaint);

    // Moon body (crescent via clipping with offset circle)
    canvas.drawCircle(
      Offset(moonX, moonY),
      moonR,
      Paint()..color = const Color(0xFFE8E8F0),
    );
    // Darken a portion for crescent
    canvas.drawCircle(
      Offset(moonX + moonR * 0.35, moonY - moonR * 0.15),
      moonR * 0.82,
      Paint()..color = const Color(0xFF1a1a4e),
    );
  }

  void _drawStars(Canvas canvas, double wx, double wy, double ww, double wh) {
    final starPaint = Paint()..color = Colors.white;
    final starRng = Random(77);
    for (int i = 0; i < 25; i++) {
      final sx = wx + starRng.nextDouble() * ww;
      final sy = wy + starRng.nextDouble() * wh * 0.7;
      final twinkle = 0.4 + 0.6 * (0.5 + 0.5 * sin(time * 2 + i * 1.7));
      starPaint.color = Colors.white.withValues(alpha: twinkle);
      canvas.drawCircle(Offset(sx, sy), 1.0 + starRng.nextDouble(), starPaint);
    }
  }

  // ── Light Beams from Window ────────────────────────────────────────

  void _drawLightBeams(Canvas canvas, Size s) {
    final hour = DateTime.now().hour;
    if (hour < 6 || hour >= 22) return; // no beams at night

    final wx = s.width * 0.32;
    final wy = s.height * 0.06;
    final ww = s.width * 0.36;
    final wh = s.height * 0.30;

    final beamAlpha = _isDark ? 0.06 : 0.04;
    final beamPaint = Paint()
      ..color = const Color(0xFFFFD700).withValues(alpha: beamAlpha)
      ..style = PaintingStyle.fill;

    // Two light beam trapezoids
    final path = Path();
    path.moveTo(wx + ww * 0.1, wy + wh);
    path.lineTo(wx + ww * 0.45, wy + wh);
    path.lineTo(wx - s.width * 0.05, s.height * 0.85);
    path.lineTo(wx - s.width * 0.15, s.height * 0.85);
    path.close();
    canvas.drawPath(path, beamPaint);

    final path2 = Path();
    path2.moveTo(wx + ww * 0.55, wy + wh);
    path2.lineTo(wx + ww * 0.9, wy + wh);
    path2.lineTo(wx + ww + s.width * 0.15, s.height * 0.85);
    path2.lineTo(wx + ww + s.width * 0.05, s.height * 0.85);
    path2.close();
    canvas.drawPath(path2, beamPaint);
  }

  // ── Wall Sign ("COGNITHOR HQ") ─────────────────────────────────────

  void _drawWallSign(Canvas canvas, Size s) {
    final sx = s.width * 0.76;
    final sy = s.height * 0.20;
    const text = 'COGNITHOR HQ';

    // Neon glow
    final glowPaint = Paint()
      ..color = JarvisTheme.accent.withValues(alpha: 0.15 + 0.05 * _osc(1.5))
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 12);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(sx, sy), width: s.width * 0.18, height: 22),
        const Radius.circular(4),
      ),
      glowPaint,
    );

    // Sign background
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(sx, sy), width: s.width * 0.18, height: 22),
        const Radius.circular(4),
      ),
      Paint()..color = (_isDark ? const Color(0xFF1a1a2e) : const Color(0xFFd0d0dc)),
    );

    // Text
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: JarvisTheme.accent,
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 2,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, Offset(sx - tp.width / 2, sy - tp.height / 2));
  }

  // ── Whiteboard ─────────────────────────────────────────────────────

  void _drawWhiteboard(Canvas canvas, Size s) {
    final bx = s.width * 0.12;
    final by = s.height * 0.19;
    final bw = s.width * 0.12;
    final bh = s.height * 0.13;

    // Board
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(bx, by, bw, bh), const Radius.circular(3)),
      Paint()..color = Colors.white.withValues(alpha: _isDark ? 0.9 : 1.0),
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(bx, by, bw, bh), const Radius.circular(3)),
      Paint()
        ..color = const Color(0xFF888888)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Drawn diagrams (boxes + lines)
    final pen = Paint()
      ..color = const Color(0xFF3366CC)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    canvas.drawRect(Rect.fromLTWH(bx + bw * 0.1, by + bh * 0.15, bw * 0.25, bh * 0.25), pen);
    canvas.drawRect(Rect.fromLTWH(bx + bw * 0.55, by + bh * 0.15, bw * 0.25, bh * 0.25), pen);
    canvas.drawLine(
      Offset(bx + bw * 0.35, by + bh * 0.275),
      Offset(bx + bw * 0.55, by + bh * 0.275),
      pen,
    );
    // Arrow head
    canvas.drawLine(
      Offset(bx + bw * 0.55, by + bh * 0.275),
      Offset(bx + bw * 0.50, by + bh * 0.22),
      pen,
    );

    // Squiggly lines (notes)
    final notePen = Paint()
      ..color = const Color(0xFF44AA44)
      ..strokeWidth = 0.8;
    for (int i = 0; i < 3; i++) {
      final ly = by + bh * 0.55 + i * bh * 0.12;
      canvas.drawLine(
        Offset(bx + bw * 0.1, ly),
        Offset(bx + bw * 0.6 + i * bw * 0.1, ly),
        notePen,
      );
    }
  }

  // ── Kanban Board ───────────────────────────────────────────────────

  void _drawKanbanBoard(Canvas canvas, Size s) {
    final bx = s.width * _kanban.dx - s.width * 0.02;
    final by = s.height * 0.20;
    final bw = s.width * 0.10;
    final bh = s.height * 0.18;

    // Board background
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(bx, by, bw, bh), const Radius.circular(3)),
      Paint()..color = _isDark ? const Color(0xFF2a2a44) : const Color(0xFFe0e0e8),
    );

    // Column headers
    final colW = bw / 3;
    final headers = ['To Do', 'WIP', 'Done'];
    final headerColors = [JarvisTheme.orange, JarvisTheme.accent, JarvisTheme.green];
    for (int c = 0; c < 3; c++) {
      final hx = bx + c * colW;
      canvas.drawRect(
        Rect.fromLTWH(hx, by, colW, bh * 0.15),
        Paint()..color = headerColors[c].withValues(alpha: 0.3),
      );
      final tp = TextPainter(
        text: TextSpan(
          text: headers[c],
          style: TextStyle(color: _textCol(), fontSize: 5, fontWeight: FontWeight.w600),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(hx + (colW - tp.width) / 2, by + 2));

      // Sticky notes
      final noteColors = [
        const Color(0xFFFFEB3B),
        const Color(0xFF81D4FA),
        const Color(0xFFA5D6A7),
        const Color(0xFFFFAB91),
      ];
      final noteCount = c == 0 ? 3 : (c == 1 ? 2 : 2);
      for (int n = 0; n < noteCount; n++) {
        final ny = by + bh * 0.18 + n * bh * 0.24;
        final noteW = colW * 0.7;
        final noteH = bh * 0.18;
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(hx + (colW - noteW) / 2, ny, noteW, noteH),
            const Radius.circular(1.5),
          ),
          Paint()..color = noteColors[(c * 3 + n) % noteColors.length].withValues(alpha: 0.85),
        );
        // Tiny text lines on sticky
        for (int l = 0; l < 2; l++) {
          canvas.drawLine(
            Offset(hx + (colW - noteW) / 2 + 2, ny + 3 + l * 4),
            Offset(hx + (colW - noteW) / 2 + noteW - 3, ny + 3 + l * 4),
            Paint()
              ..color = Colors.black26
              ..strokeWidth = 0.5,
          );
        }
      }
    }

    // Border
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(bx, by, bw, bh), const Radius.circular(3)),
      Paint()
        ..color = _furnitureDark()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );
  }

  // ── Ceiling Lights ─────────────────────────────────────────────────

  void _drawCeilingLights(Canvas canvas, Size s) {
    for (int i = 0; i < 3; i++) {
      final lx = s.width * (0.22 + i * 0.28);
      final ly = s.height * 0.16;
      final lw = s.width * 0.14;
      const lh = 6.0;

      // Fixture
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(center: Offset(lx, ly), width: lw, height: lh),
          const Radius.circular(2),
        ),
        Paint()..color = const Color(0xFF888898),
      );

      // Glow
      final glowAlpha = 0.06 + 0.02 * _osc(0.8, i * 2.0);
      canvas.drawOval(
        Rect.fromCenter(center: Offset(lx, ly + 10), width: lw * 1.3, height: 30),
        Paint()
          ..color = Colors.white.withValues(alpha: glowAlpha)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 15),
      );
    }
  }

  // ── Floor ──────────────────────────────────────────────────────────

  void _drawFloor(Canvas canvas, Size s) {
    final floorTop = s.height * 0.45;
    final floorHeight = s.height - floorTop;

    // Base floor gradient
    final floorPaint = Paint()
      ..shader = ui.Gradient.linear(
        Offset(0, floorTop),
        Offset(0, s.height),
        [_floorBase(), _floorBase().withValues(alpha: 0.85)],
      );
    canvas.drawRect(Rect.fromLTWH(0, floorTop, s.width, floorHeight), floorPaint);

    // Checkered tiles with perspective
    final tileSize = s.width / 16;
    final rows = (floorHeight / tileSize).ceil() + 1;
    final cols = (s.width / tileSize).ceil() + 1;

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        final tx = c * tileSize;
        final ty = floorTop + r * tileSize;
        if (ty >= s.height) continue;
        final isAlt = (r + c) % 2 == 0;
        final tilePaint = Paint()
          ..color = (isAlt ? _floorAlt() : _floorBase()).withValues(
            alpha: 0.5 + 0.3 * ((ty - floorTop) / floorHeight),
          );
        canvas.drawRect(Rect.fromLTWH(tx, ty, tileSize, tileSize), tilePaint);
      }
    }

    // Subtle reflections
    final reflPaint = Paint()
      ..color = Colors.white.withValues(alpha: _isDark ? 0.02 : 0.04)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(s.width * 0.5, s.height * 0.65),
        width: s.width * 0.6,
        height: s.height * 0.15,
      ),
      reflPaint,
    );
  }

  // ── Cables ─────────────────────────────────────────────────────────

  void _drawCables(Canvas canvas, Size s) {
    final cablePaint = Paint()
      ..color = _isDark ? const Color(0xFF333350) : const Color(0xFFaaaabc)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    // Cable from desk2 to server
    final path = Path();
    path.moveTo(s.width * _desk2.dx + s.width * 0.06, s.height * _desk2.dy + s.height * 0.05);
    path.quadraticBezierTo(
      s.width * 0.75,
      s.height * 0.58,
      s.width * _server.dx,
      s.height * _server.dy + s.height * 0.12,
    );
    canvas.drawPath(path, cablePaint);

    // Cable from desk1 to desk2
    final path2 = Path();
    path2.moveTo(s.width * _desk1.dx + s.width * 0.06, s.height * _desk1.dy + s.height * 0.06);
    path2.quadraticBezierTo(
      s.width * 0.32,
      s.height * 0.62,
      s.width * _desk2.dx - s.width * 0.02,
      s.height * _desk2.dy + s.height * 0.05,
    );
    canvas.drawPath(path2, cablePaint);
  }

  // ── Desks ──────────────────────────────────────────────────────────

  void _drawDesks(Canvas canvas, Size s) {
    final desks = [_desk1, _desk2, _desk3];
    for (int i = 0; i < desks.length; i++) {
      _drawDesk(canvas, s, desks[i], i);
    }
  }

  void _drawDesk(Canvas canvas, Size s, Offset pos, int index) {
    final dx = s.width * pos.dx;
    final dy = s.height * pos.dy;
    final dw = s.width * 0.12;
    final dh = s.height * 0.08;

    // Desk surface
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(dx - dw / 2, dy, dw, dh),
        const Radius.circular(3),
      ),
      Paint()..color = _furnitureLight(),
    );
    // Desk edge highlight
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(dx - dw / 2, dy, dw, 3),
        const Radius.circular(3),
      ),
      Paint()..color = _furnitureDark(),
    );

    // Desk legs
    final legPaint = Paint()..color = _furnitureDark();
    canvas.drawRect(Rect.fromLTWH(dx - dw / 2 + 4, dy + dh, 3, s.height * 0.06), legPaint);
    canvas.drawRect(Rect.fromLTWH(dx + dw / 2 - 7, dy + dh, 3, s.height * 0.06), legPaint);

    // Monitor
    final monW = dw * 0.5;
    final monH = dh * 0.7;
    final monX = dx - monW / 2;
    final monY = dy - monH - 4;

    // Monitor stand
    canvas.drawRect(
      Rect.fromLTWH(dx - 2, monY + monH, 4, 5),
      Paint()..color = const Color(0xFF444466),
    );
    canvas.drawRect(
      Rect.fromLTWH(dx - 8, monY + monH + 4, 16, 2),
      Paint()..color = const Color(0xFF444466),
    );

    // Monitor screen
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(monX, monY, monW, monH),
        const Radius.circular(2),
      ),
      Paint()..color = const Color(0xFF111122),
    );

    // Animated screen content (code lines)
    final screenPaint = Paint()..strokeWidth = 1.0;
    final lineRng = Random(index * 7 + 13);
    for (int l = 0; l < 5; l++) {
      final ly = monY + 3 + l * (monH - 6) / 5;
      final lineW = monW * (0.3 + lineRng.nextDouble() * 0.5);
      // Scroll effect
      final scrollOffset = (time * 8 + index * 20) % (monH * 2);
      final adjustedY = ly + scrollOffset % 3;
      if (adjustedY < monY + monH - 2) {
        screenPaint.color = JarvisTheme.accent.withValues(alpha: 0.5 + 0.2 * _osc(3, l + index * 5.0));
        canvas.drawLine(
          Offset(monX + 3, adjustedY),
          Offset(monX + 3 + lineW, adjustedY),
          screenPaint,
        );
      }
    }

    // Screen glow
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(monX, monY, monW, monH),
        const Radius.circular(2),
      ),
      Paint()
        ..color = JarvisTheme.accent.withValues(alpha: 0.04 + 0.02 * _osc(2, index.toDouble()))
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6),
    );

    // Keyboard
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(dx - dw * 0.2, dy + 4, dw * 0.4, dh * 0.2),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFF555568),
    );

    // Small items: coffee cup on desk 0, post-its on desk 1, pen holder on desk 2
    if (index == 0) {
      // Coffee cup
      final cx = dx + dw * 0.3;
      final cy = dy + 3;
      canvas.drawRRect(
        RRect.fromRectAndRadius(Rect.fromLTWH(cx, cy, 6, 8), const Radius.circular(1)),
        Paint()..color = const Color(0xFFDDDDDD),
      );
      // Coffee surface
      canvas.drawRect(
        Rect.fromLTWH(cx + 1, cy + 1, 4, 3),
        Paint()..color = const Color(0xFF6B4226),
      );
    } else if (index == 1) {
      // Post-it notes stack
      final colors = [const Color(0xFFFFEB3B), const Color(0xFFFF80AB), const Color(0xFF80D8FF)];
      for (int n = 0; n < 3; n++) {
        canvas.drawRect(
          Rect.fromLTWH(dx + dw * 0.25 + n * 2, dy + 3 + n * 1.5, 8, 8),
          Paint()..color = colors[n].withValues(alpha: 0.8),
        );
      }
    } else {
      // Pen holder
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(dx - dw * 0.35, dy + 2, 7, 10),
          const Radius.circular(1),
        ),
        Paint()..color = const Color(0xFF777790),
      );
      // Pens
      canvas.drawLine(
        Offset(dx - dw * 0.33, dy + 2),
        Offset(dx - dw * 0.30, dy - 4),
        Paint()
          ..color = JarvisTheme.red
          ..strokeWidth = 1,
      );
      canvas.drawLine(
        Offset(dx - dw * 0.31, dy + 2),
        Offset(dx - dw * 0.29, dy - 3),
        Paint()
          ..color = JarvisTheme.accent
          ..strokeWidth = 1,
      );
    }
  }

  // ── Server Rack ────────────────────────────────────────────────────

  void _drawServerRack(Canvas canvas, Size s) {
    final rx = s.width * _server.dx - s.width * 0.035;
    final ry = s.height * 0.28;
    final rw = s.width * 0.07;
    final rh = s.height * 0.22;

    // Rack body
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(rx, ry, rw, rh), const Radius.circular(3)),
      Paint()..color = const Color(0xFF2a2a44),
    );
    // Rack border
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(rx, ry, rw, rh), const Radius.circular(3)),
      Paint()
        ..color = const Color(0xFF444466)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );

    // Rack shelves (server units)
    for (int i = 0; i < 5; i++) {
      final sy = ry + 6 + i * (rh - 12) / 5;
      final sh = (rh - 12) / 5 - 3;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(rx + 4, sy, rw - 8, sh),
          const Radius.circular(1.5),
        ),
        Paint()..color = const Color(0xFF1a1a30),
      );

      // Blinking LEDs
      for (int led = 0; led < 3; led++) {
        final ledColors = [
          JarvisTheme.green,
          JarvisTheme.orange,
          JarvisTheme.red,
        ];
        final blink = sin(time * (3 + led) + i * 1.5 + led * 2.1);
        final on = blink > (led == 2 ? 0.7 : 0.0); // red blinks less
        canvas.drawCircle(
          Offset(rx + 8 + led * 5, sy + sh / 2),
          1.8,
          Paint()..color = on ? ledColors[led].withValues(alpha: 0.9) : const Color(0xFF333350),
        );
        // LED glow
        if (on) {
          canvas.drawCircle(
            Offset(rx + 8 + led * 5, sy + sh / 2),
            4,
            Paint()
              ..color = ledColors[led].withValues(alpha: 0.15)
              ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
          );
        }
      }

      // Vent lines
      for (int v = 0; v < 3; v++) {
        canvas.drawLine(
          Offset(rx + rw - 14 + v * 3, sy + 2),
          Offset(rx + rw - 14 + v * 3, sy + sh - 2),
          Paint()
            ..color = const Color(0xFF333350)
            ..strokeWidth = 0.5,
        );
      }
    }

    // Cables hanging from bottom
    final cablePaint = Paint()
      ..color = const Color(0xFF333350)
      ..strokeWidth = 1;
    for (int c = 0; c < 3; c++) {
      final cx = rx + 10 + c * 12;
      final path = Path()
        ..moveTo(cx.toDouble(), ry + rh)
        ..quadraticBezierTo(
          cx + 4 * _osc(1.2, c.toDouble()),
          ry + rh + 12,
          cx + 2.0,
          ry + rh + 18,
        );
      canvas.drawPath(path, cablePaint);
    }
  }

  // ── Coffee Machine ─────────────────────────────────────────────────

  void _drawCoffeeMachine(Canvas canvas, Size s) {
    final cx = s.width * _coffee.dx;
    final cy = s.height * _coffee.dy;
    final cw = s.width * 0.035;
    final ch = s.height * 0.08;

    // Small table
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(cx - cw * 0.8, cy + ch * 0.3, cw * 1.6, ch * 0.12),
        const Radius.circular(2),
      ),
      Paint()..color = _furnitureLight(),
    );
    // Table leg
    canvas.drawRect(
      Rect.fromLTWH(cx - 2, cy + ch * 0.42, 4, s.height * 0.05),
      Paint()..color = _furnitureDark(),
    );

    // Machine body
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(cx - cw / 2, cy - ch * 0.2, cw, ch * 0.5),
        const Radius.circular(3),
      ),
      Paint()..color = const Color(0xFF555568),
    );
    // Machine top
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(cx - cw / 2 - 1, cy - ch * 0.22, cw + 2, 4),
        const Radius.circular(2),
      ),
      Paint()..color = const Color(0xFF666680),
    );
    // Dispenser nozzle
    canvas.drawRect(
      Rect.fromLTWH(cx - 2, cy + ch * 0.15, 4, ch * 0.1),
      Paint()..color = const Color(0xFF333350),
    );
    // Cup
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(cx - 4, cy + ch * 0.22, 8, 8),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFFDDDDDD),
    );

    // Status LED
    canvas.drawCircle(
      Offset(cx + cw / 2 - 4, cy - ch * 0.1),
      2,
      Paint()..color = JarvisTheme.green.withValues(alpha: 0.6 + 0.4 * _osc(2)),
    );

    // Steam particles
    for (int p = 0; p < 6; p++) {
      final age = (time * 1.5 + p * 0.8) % 3.0;
      final py = cy + ch * 0.15 - age * 12;
      final px = cx + sin(time * 2 + p * 1.3) * 4;
      final alpha = (1.0 - age / 3.0).clamp(0.0, 1.0) * 0.3;
      canvas.drawCircle(
        Offset(px, py),
        1.5 + age * 0.8,
        Paint()
          ..color = Colors.white.withValues(alpha: alpha)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2),
      );
    }
  }

  // ── Plants ─────────────────────────────────────────────────────────

  void _drawPlants(Canvas canvas, Size s) {
    _drawPlant(canvas, s, s.width * 0.04, s.height * 0.48, 0);
    _drawPlant(canvas, s, s.width * 0.82, s.height * 0.46, 1);
  }

  void _drawPlant(Canvas canvas, Size s, double px, double py, int seed) {
    final potW = s.width * 0.03;
    final potH = s.height * 0.04;

    // Pot
    final potPath = Path();
    potPath.moveTo(px - potW / 2, py);
    potPath.lineTo(px - potW * 0.35, py + potH);
    potPath.lineTo(px + potW * 0.35, py + potH);
    potPath.lineTo(px + potW / 2, py);
    potPath.close();
    canvas.drawPath(
      potPath,
      Paint()..color = const Color(0xFFC67B5C),
    );
    // Pot rim
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(px - potW / 2 - 1, py - 2, potW + 2, 4),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFFB06B4C),
    );
    // Soil
    canvas.drawOval(
      Rect.fromLTWH(px - potW / 2 + 2, py - 1, potW - 4, 4),
      Paint()..color = const Color(0xFF5C3D2E),
    );

    // Leaves
    final leafPaint = Paint()..color = const Color(0xFF4CAF50);
    for (int l = 0; l < 5; l++) {
      final angle = -pi / 2 + (l - 2) * 0.4;
      final sway = _osc(0.8 + seed * 0.3, l * 1.5 + seed * 3.0) * 0.08;
      final leafLen = potH * (1.2 + l % 2 * 0.4);
      final endX = px + cos(angle + sway) * leafLen;
      final endY = py - 2 + sin(angle + sway) * leafLen;

      final leafPath = Path();
      leafPath.moveTo(px, py - 2);
      leafPath.quadraticBezierTo(
        px + cos(angle + sway + 0.3) * leafLen * 0.6,
        py - 2 + sin(angle + sway + 0.3) * leafLen * 0.6,
        endX,
        endY,
      );
      leafPath.quadraticBezierTo(
        px + cos(angle + sway - 0.3) * leafLen * 0.6,
        py - 2 + sin(angle + sway - 0.3) * leafLen * 0.6,
        px,
        py - 2,
      );
      canvas.drawPath(leafPath, leafPaint);
    }

    // Stem
    canvas.drawLine(
      Offset(px, py - 2),
      Offset(px, py - potH * 0.8),
      Paint()
        ..color = const Color(0xFF388E3C)
        ..strokeWidth = 1.5,
    );
  }

  // ── Robot Drawing ──────────────────────────────────────────────────

  void _drawRobot(Canvas canvas, Robot robot, Size s) {
    final rx = robot.x * s.width;
    final ry = robot.y * s.height;
    final scale = 0.6 + 0.4 * ((robot.y - 0.3) / 0.6).clamp(0.0, 1.0);
    final bodyW = s.width * 0.04 * scale;
    final bodyH = s.height * 0.06 * scale;
    final headW = bodyW * 0.8;
    final headH = bodyH * 0.55;
    final legH = bodyH * 0.35;

    // Shadow
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(rx, ry + bodyH / 2 + legH + 4),
        width: bodyW * 1.2,
        height: bodyH * 0.2,
      ),
      Paint()..color = Colors.black.withValues(alpha: 0.2),
    );

    // Walk animation
    final walking = robot.state == RobotState.walking || robot.state == RobotState.carrying;
    final walkPhase = walking ? time * 8 : 0.0;
    final bobY = robot.state == RobotState.idle ? sin(time * 2) * 1.5 : 0.0;

    // Legs
    final legPaint = Paint()
      ..color = robot.type.color.withValues(alpha: 0.7)
      ..strokeWidth = 2.5 * scale
      ..strokeCap = StrokeCap.round;

    final legSwing = walking ? sin(walkPhase) * 4 * scale : 0.0;
    // Left leg
    canvas.drawLine(
      Offset(rx - bodyW * 0.2, ry + bodyH * 0.3 + bobY),
      Offset(rx - bodyW * 0.2 - legSwing, ry + bodyH * 0.3 + legH + bobY),
      legPaint,
    );
    // Left foot
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(rx - bodyW * 0.2 - legSwing, ry + bodyH * 0.3 + legH + 2 + bobY),
        width: 6 * scale,
        height: 3 * scale,
      ),
      Paint()..color = robot.type.color.withValues(alpha: 0.5),
    );
    // Right leg
    canvas.drawLine(
      Offset(rx + bodyW * 0.2, ry + bodyH * 0.3 + bobY),
      Offset(rx + bodyW * 0.2 + legSwing, ry + bodyH * 0.3 + legH + bobY),
      legPaint,
    );
    // Right foot
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(rx + bodyW * 0.2 + legSwing, ry + bodyH * 0.3 + legH + 2 + bobY),
        width: 6 * scale,
        height: 3 * scale,
      ),
      Paint()..color = robot.type.color.withValues(alpha: 0.5),
    );

    // Body
    final bodyRect = RRect.fromRectAndRadius(
      Rect.fromCenter(
        center: Offset(rx, ry + bobY),
        width: bodyW,
        height: bodyH,
      ),
      Radius.circular(6 * scale),
    );
    // Body gradient
    canvas.drawRRect(
      bodyRect,
      Paint()
        ..shader = ui.Gradient.linear(
          Offset(rx - bodyW / 2, ry - bodyH / 2),
          Offset(rx + bodyW / 2, ry + bodyH / 2),
          [
            robot.type.color,
            Color.lerp(robot.type.color, Colors.black, 0.3)!,
          ],
        ),
    );
    // Body highlight
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(
          rx - bodyW * 0.35,
          ry - bodyH * 0.45 + bobY,
          bodyW * 0.3,
          bodyH * 0.4,
        ),
        Radius.circular(4 * scale),
      ),
      Paint()..color = Colors.white.withValues(alpha: 0.12),
    );

    // Chest LED (pulsing)
    final ledPulse = 0.5 + 0.5 * sin(time * 3 + robot.type.id.hashCode.toDouble());
    canvas.drawCircle(
      Offset(rx, ry + bodyH * 0.1 + bobY),
      3 * scale,
      Paint()..color = robot.type.eyeColor.withValues(alpha: 0.3 + 0.5 * ledPulse),
    );
    canvas.drawCircle(
      Offset(rx, ry + bodyH * 0.1 + bobY),
      6 * scale,
      Paint()
        ..color = robot.type.eyeColor.withValues(alpha: 0.08 * ledPulse)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4),
    );

    // Arms
    final armPaint = Paint()
      ..color = robot.type.color.withValues(alpha: 0.8)
      ..strokeWidth = 2.5 * scale
      ..strokeCap = StrokeCap.round;

    final armSwing = walking ? sin(walkPhase + pi) * 6 * scale : 0.0;
    final typing = robot.state == RobotState.working;
    final typeAnim = typing ? sin(time * 12) * 2 * scale : 0.0;

    // Left arm
    canvas.drawLine(
      Offset(rx - bodyW / 2, ry - bodyH * 0.15 + bobY),
      Offset(
        rx - bodyW / 2 - 5 * scale + (typing ? typeAnim : armSwing),
        ry + bodyH * 0.15 + bobY + (typing ? -bodyH * 0.1 : 0),
      ),
      armPaint,
    );
    // Right arm
    canvas.drawLine(
      Offset(rx + bodyW / 2, ry - bodyH * 0.15 + bobY),
      Offset(
        rx + bodyW / 2 + 5 * scale + (typing ? -typeAnim : -armSwing),
        ry + bodyH * 0.15 + bobY + (typing ? -bodyH * 0.1 : 0),
      ),
      armPaint,
    );

    // Document in hand when carrying
    if (robot.state == RobotState.carrying) {
      final docX = rx + bodyW / 2 + 6 * scale - armSwing;
      final docY = ry + bodyH * 0.1 + bobY;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(center: Offset(docX, docY), width: 8 * scale, height: 10 * scale),
          const Radius.circular(1),
        ),
        Paint()..color = Colors.white.withValues(alpha: 0.9),
      );
      // Lines on document
      for (int dl = 0; dl < 3; dl++) {
        canvas.drawLine(
          Offset(docX - 2 * scale, docY - 3 * scale + dl * 3 * scale),
          Offset(docX + 2 * scale, docY - 3 * scale + dl * 3 * scale),
          Paint()
            ..color = Colors.black38
            ..strokeWidth = 0.5,
        );
      }
    }

    // Head
    final headY = ry - bodyH / 2 - headH / 2 - 2 * scale + bobY;
    final headRect = RRect.fromRectAndRadius(
      Rect.fromCenter(center: Offset(rx, headY), width: headW, height: headH),
      Radius.circular(5 * scale),
    );
    canvas.drawRRect(
      headRect,
      Paint()
        ..shader = ui.Gradient.linear(
          Offset(rx, headY - headH / 2),
          Offset(rx, headY + headH / 2),
          [
            Color.lerp(robot.type.color, Colors.white, 0.15)!,
            robot.type.color,
          ],
        ),
    );
    // Head highlight
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(rx - headW * 0.35, headY - headH * 0.4, headW * 0.3, headH * 0.35),
        Radius.circular(3 * scale),
      ),
      Paint()..color = Colors.white.withValues(alpha: 0.15),
    );

    // Eyes (with blink)
    final blinkCycle = (time * 0.5 + robot.type.id.hashCode * 0.1) % 4.0;
    final blinking = blinkCycle > 3.8; // blink for 0.2s every 4s
    final eyeH = blinking ? 1.0 * scale : 4.0 * scale;
    final eyeW = 4.5 * scale;
    final eyeSpacing = headW * 0.22;

    // Eye sockets
    for (final side in [-1.0, 1.0]) {
      final eyeX = rx + side * eyeSpacing;
      // Socket
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(eyeX, headY),
          width: eyeW + 2 * scale,
          height: (blinking ? 2 : 6) * scale,
        ),
        Paint()..color = Colors.black.withValues(alpha: 0.3),
      );
      // Eye
      canvas.drawOval(
        Rect.fromCenter(center: Offset(eyeX, headY), width: eyeW, height: eyeH),
        Paint()..color = robot.type.eyeColor,
      );
      // Eye glow
      canvas.drawOval(
        Rect.fromCenter(center: Offset(eyeX, headY), width: eyeW * 1.8, height: eyeH * 2),
        Paint()
          ..color = robot.type.eyeColor.withValues(alpha: 0.12)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
      );
    }

    // Mouth — expression based on state
    final mouthY = headY + headH * 0.25;
    if (robot.state == RobotState.working) {
      // Happy smile
      final smilePath = Path();
      smilePath.moveTo(rx - headW * 0.15, mouthY);
      smilePath.quadraticBezierTo(rx, mouthY + 3 * scale, rx + headW * 0.15, mouthY);
      canvas.drawPath(
        smilePath,
        Paint()
          ..color = robot.type.eyeColor.withValues(alpha: 0.6)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.2 * scale
          ..strokeCap = StrokeCap.round,
      );
    } else {
      // Neutral line
      canvas.drawLine(
        Offset(rx - headW * 0.12, mouthY),
        Offset(rx + headW * 0.12, mouthY),
        Paint()
          ..color = robot.type.eyeColor.withValues(alpha: 0.4)
          ..strokeWidth = 1 * scale
          ..strokeCap = StrokeCap.round,
      );
    }

    // Antenna (Planner & Researcher)
    if (robot.type.antenna) {
      final antennaBase = headY - headH / 2;
      final wobble = _osc(3, robot.type.id.hashCode.toDouble()) * 2;
      canvas.drawLine(
        Offset(rx, antennaBase),
        Offset(rx + wobble, antennaBase - 10 * scale),
        Paint()
          ..color = robot.type.color.withValues(alpha: 0.7)
          ..strokeWidth = 1.5 * scale,
      );
      // Antenna tip glow
      final tipPulse = 0.5 + 0.5 * _osc(4, robot.type.id.hashCode.toDouble() + 1);
      canvas.drawCircle(
        Offset(rx + wobble, antennaBase - 10 * scale),
        2.5 * scale,
        Paint()..color = robot.type.eyeColor.withValues(alpha: tipPulse),
      );
      canvas.drawCircle(
        Offset(rx + wobble, antennaBase - 10 * scale),
        5 * scale,
        Paint()
          ..color = robot.type.eyeColor.withValues(alpha: 0.15 * tipPulse)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
      );
    }

    // Name tag
    final tagTp = TextPainter(
      text: TextSpan(
        text: robot.type.name,
        style: TextStyle(
          color: _textCol().withValues(alpha: 0.8),
          fontSize: 7 * scale,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tagTp.paint(
      canvas,
      Offset(rx - tagTp.width / 2, ry + bodyH * 0.3 + legH + 6 + bobY),
    );

    // Speech bubble
    if (robot.speechTimer > 0 && robot.speechText.isNotEmpty) {
      _drawSpeechBubble(canvas, rx, headY - headH / 2 - 12 * scale, robot.speechText, scale);
    }

    // Floating emoji
    if (robot.emojiTimer > 0 && robot.floatingEmoji.isNotEmpty) {
      final emojiY = headY - headH / 2 - 18 * scale - robot.emojiTimer * 8;
      final emojiAlpha = robot.emojiTimer.clamp(0.0, 1.0);
      final emojiTp = TextPainter(
        text: TextSpan(
          text: robot.floatingEmoji,
          style: TextStyle(fontSize: 12 * scale, color: Colors.white.withValues(alpha: emojiAlpha)),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      emojiTp.paint(canvas, Offset(rx - emojiTp.width / 2, emojiY));
    }
  }

  void _drawSpeechBubble(Canvas canvas, double x, double y, String text, double scale) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(color: _textCol(), fontSize: 6 * scale, fontWeight: FontWeight.w500),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final bw = tp.width + 10 * scale;
    final bh = tp.height + 6 * scale;
    final bx = x - bw / 2;
    final by = y - bh;

    // Bubble
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(bx, by, bw, bh), Radius.circular(4 * scale)),
      Paint()..color = (_isDark ? const Color(0xFF2a2a44) : Colors.white).withValues(alpha: 0.92),
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(bx, by, bw, bh), Radius.circular(4 * scale)),
      Paint()
        ..color = _furnitureDark()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 0.8,
    );

    // Triangle pointer
    final triPath = Path();
    triPath.moveTo(x - 3 * scale, by + bh);
    triPath.lineTo(x, by + bh + 4 * scale);
    triPath.lineTo(x + 3 * scale, by + bh);
    triPath.close();
    canvas.drawPath(
      triPath,
      Paint()..color = (_isDark ? const Color(0xFF2a2a44) : Colors.white).withValues(alpha: 0.92),
    );

    tp.paint(canvas, Offset(bx + 5 * scale, by + 3 * scale));
  }

  // ── Data Streams (inter-agent communication) ───────────────────────

  void _drawDataStreams(Canvas canvas, Size s) {
    // Find pairs of robots that are both "working"
    final working = robots.where((r) => r.state == RobotState.working).toList();
    if (working.length < 2) return;

    for (int i = 0; i < working.length - 1; i++) {
      for (int j = i + 1; j < working.length; j++) {
        final a = working[i];
        final b = working[j];
        _drawParticleStream(canvas, s, a.x, a.y, b.x, b.y, a.type.color, b.type.color);
      }
    }
  }

  void _drawParticleStream(
    Canvas canvas,
    Size s,
    double ax,
    double ay,
    double bx,
    double by,
    Color colorA,
    Color colorB,
  ) {
    const particleCount = 8;
    for (int p = 0; p < particleCount; p++) {
      final t = ((time * 0.8 + p / particleCount) % 1.0);
      final px = (ax + (bx - ax) * t) * s.width;
      final py = (ay + (by - ay) * t) * s.height;
      // Arc upward
      final arc = -sin(t * pi) * s.height * 0.04;
      final color = Color.lerp(colorA, colorB, t)!;
      canvas.drawCircle(
        Offset(px, py + arc),
        2.0 + sin(time * 4 + p) * 0.5,
        Paint()..color = color.withValues(alpha: 0.5 + 0.3 * sin(time * 3 + p)),
      );
      // Particle glow
      canvas.drawCircle(
        Offset(px, py + arc),
        5,
        Paint()
          ..color = color.withValues(alpha: 0.1)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
      );
    }
  }
}

// ── Robot Office Widget (hosts the painter + animation + AI logic) ────

class RobotOffice extends StatefulWidget {
  const RobotOffice({super.key, this.isRunning = true});

  final bool isRunning;

  @override
  State<RobotOffice> createState() => _RobotOfficeState();
}

class _RobotOfficeState extends State<RobotOffice> with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final List<Robot> _robots;
  final _rng = Random();
  double _elapsed = 0;

  static const _speechMessages = [
    'Plan erstellt',
    'Task erledigt!',
    'Suche laeuft...',
    'Geprueft!',
    'Analysiere...',
    'Optimierung',
    'Daten sync',
    'Bericht fertig',
  ];

  static const _emojis = ['✓', '⚡', '🔍', '🛡', '📊', '💡', '🚀', '✨'];

  @override
  void initState() {
    super.initState();
    _robots = [
      Robot(type: _planner, x: _desk1.dx, y: _desk1.dy + 0.12),
      Robot(type: _executor, x: _desk2.dx, y: _desk2.dy + 0.12),
      Robot(type: _researcher, x: _desk3.dx, y: _desk3.dy + 0.12),
      Robot(type: _gatekeeper, x: 0.50, y: 0.70),
    ];

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..addListener(_onTick);

    _controller.repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onTick() {
    final dt = 1.0 / 60.0; // approximate dt
    _elapsed += dt;
    _updateRobots(dt);
    // setState is implicitly called by the listener triggering a rebuild
    setState(() {});
  }

  void _updateRobots(double dt) {
    for (final robot in _robots) {
      robot.stateTimer += dt;
      robot.blinkTimer += dt;

      // Decrease speech / emoji timers
      if (robot.speechTimer > 0) robot.speechTimer -= dt;
      if (robot.emojiTimer > 0) robot.emojiTimer -= dt;

      switch (robot.state) {
        case RobotState.idle:
          if (robot.stateTimer >= robot.nextSwitch) {
            _chooseNewAction(robot);
          }
          break;

        case RobotState.walking:
        case RobotState.carrying:
          // Move toward target
          final dx = robot.targetX - robot.x;
          final dy = robot.targetY - robot.y;
          final dist = sqrt(dx * dx + dy * dy);
          if (dist < 0.01) {
            robot.x = robot.targetX;
            robot.y = robot.targetY;
            robot.state = RobotState.working;
            robot.stateTimer = 0;
            robot.nextSwitch = 2.0 + _rng.nextDouble() * 3.0;
            // Show speech bubble
            robot.speechText = _speechMessages[_rng.nextInt(_speechMessages.length)];
            robot.speechTimer = 2.5;
          } else {
            final speed = 0.15 * dt; // normalized units per second
            robot.x += dx / dist * speed;
            robot.y += dy / dist * speed;
          }
          break;

        case RobotState.working:
          if (robot.stateTimer >= robot.nextSwitch) {
            // Chance to show completion emoji
            if (_rng.nextDouble() < 0.4) {
              robot.floatingEmoji = _emojis[_rng.nextInt(_emojis.length)];
              robot.emojiTimer = 1.5;
            }
            _chooseNewAction(robot);
          }
          break;
      }
    }
  }

  void _chooseNewAction(Robot robot) {
    final roll = _rng.nextDouble();
    Offset target;

    if (roll < 0.35) {
      // Go to a random desk
      final desks = [_desk1, _desk2, _desk3];
      final desk = desks[_rng.nextInt(desks.length)];
      target = Offset(desk.dx, desk.dy + 0.12);
    } else if (roll < 0.50) {
      // Go to server
      target = Offset(_server.dx, _server.dy + 0.12);
    } else if (roll < 0.60) {
      // Go to kanban board
      target = Offset(_kanban.dx + 0.04, _kanban.dy + 0.22);
    } else if (roll < 0.70) {
      // Coffee break
      target = Offset(_coffee.dx, _coffee.dy + 0.12);
    } else {
      // Stay and work at current position
      robot.state = RobotState.working;
      robot.stateTimer = 0;
      robot.nextSwitch = 2.0 + _rng.nextDouble() * 3.0;
      return;
    }

    robot.targetX = target.dx;
    robot.targetY = target.dy;
    robot.state = roll > 0.50 && roll < 0.55 ? RobotState.carrying : RobotState.walking;
    robot.stateTimer = 0;
    robot.nextSwitch = 10.0; // max walk time before re-evaluation
  }

  @override
  Widget build(BuildContext context) {
    final brightness = Theme.of(context).brightness;
    return RepaintBoundary(
      child: CustomPaint(
        painter: OfficePainter(
          robots: _robots,
          time: _elapsed,
          isRunning: widget.isRunning,
          brightness: brightness,
        ),
        size: Size.infinite,
      ),
    );
  }
}
