import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

// ──────────────────────────────────────────────────────────────────────
// Robot Office – Animated CustomPainter (centerpiece visualization)
//
// Draws a fully procedural office scene at 60 fps with 8 AI-agent robots,
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

const _coder = RobotType(
  id: 'coder',
  name: 'Coder',
  role: 'Programmierung',
  color: Color(0xFF8b5cf6),
  eyeColor: Color(0xFFc4b5fd),
);

const _analyst = RobotType(
  id: 'analyst',
  name: 'Analyst',
  role: 'Datenanalyse',
  color: Color(0xFF06b6d4),
  eyeColor: Color(0xFF67e8f9),
  antenna: true,
);

const _memory = RobotType(
  id: 'memory',
  name: 'Memory',
  role: 'Wissen',
  color: Color(0xFFec4899),
  eyeColor: Color(0xFFf9a8d4),
);

const _ops = RobotType(
  id: 'ops',
  name: 'DevOps',
  role: 'Infrastruktur',
  color: Color(0xFF84cc16),
  eyeColor: Color(0xFFbef264),
  antenna: true,
);

/// Helper for flower positioning.
class _FlowerInfo {
  const _FlowerInfo(this.x, this.y, this.color, this.seed);
  final double x;
  final double y;
  final Color color;
  final int seed;
}

// ── Office Landmark Positions (normalized 0..1) ──────────────────────

const _desk1 = Offset(0.18, 0.52);
const _desk2 = Offset(0.45, 0.50);
const _desk3 = Offset(0.12, 0.66);
const _desk4 = Offset(0.38, 0.64);
const _desk5 = Offset(0.64, 0.66);
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
    this.dog,
    this.cat,
    this.particles,
    this.cpuUsage = 0,
    this.memoryUsage = 0,
    this.activePhase = 0,
    this.systemLoad = 0,
    this.kanbanCounts = const {},
    this.kanbanTasks = const {},
  });

  final List<Robot> robots;
  final double time; // elapsed seconds
  final bool isRunning;
  final Brightness brightness;
  final dynamic dog;
  final dynamic cat;
  final dynamic particles;

  /// CPU usage 0.0-1.0 — controls server rack LED blink speed.
  final double cpuUsage;

  /// Memory usage 0.0-1.0 — LEDs shift toward red when > 0.8.
  final double memoryUsage;

  /// Active pipeline phase 0-4 — highlights the matching kanban column.
  final int activePhase;

  /// System load 0.0-1.0 — controls ceiling light brightness.
  final double systemLoad;

  /// Kanban column counts by status (e.g. 'backlog': 3, 'in_progress': 2).
  final Map<String, int> kanbanCounts;

  /// Kanban task titles by status (e.g. 'backlog': ['Task 1', 'Task 2']).
  final Map<String, List<String>> kanbanTasks;

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
    _drawCarpetArea(canvas, size);
    _drawLightBeams(canvas, size);
    _drawCables(canvas, size);
    _drawDesks(canvas, size);
    _drawServerRack(canvas, size);
    _drawCoffeeMachine(canvas, size);
    _drawPlants(canvas, size);
    _drawFlowers(canvas, size);
    _drawCeilingLightBeams(canvas, size);
    _drawSystemMonitor(canvas, size);

    // Sort robots by Y for depth ordering
    final sorted = [...robots]..sort((a, b) => a.y.compareTo(b.y));
    for (final robot in sorted) {
      _drawRobot(canvas, robot, size);
    }
    if (isRunning) _drawDataStreams(canvas, size);
  }

  @override
  bool shouldRepaint(covariant OfficePainter old) => true; // animation-driven — always repaint

  // ── Helpers ────────────────────────────────────────────────────────

  bool get _isDark => brightness == Brightness.dark;

  /// 0.0 = full night, 1.0 = full day. Smooth transitions at dawn/dusk.
  double get _dayFactor {
    final h = DateTime.now().hour + DateTime.now().minute / 60.0;
    if (h >= 8 && h <= 17) return 1.0; // full day
    if (h >= 20 || h <= 5) return 0.0; // full night
    if (h < 8) return (h - 5) / 3.0; // dawn transition
    return 1.0 - (h - 17) / 3.0; // dusk transition
  }

  Color _wallColor() =>
      _isDark ? const Color(0xFF1a1a2e) : const Color(0xFFe8e8f0);
  Color _floorBase() =>
      _isDark ? const Color(0xFF2E2E3E) : const Color(0xFFD8D0C8);
  Color _floorAlt() =>
      _isDark ? const Color(0xFF343448) : const Color(0xFFE0D8D0);
  Color _furnitureDark() =>
      _isDark ? const Color(0xFF2c2c44) : const Color(0xFFb8b8c8);
  Color _furnitureLight() =>
      _isDark ? const Color(0xFF3a3a56) : const Color(0xFFcacad8);
  Color _textCol() =>
      _isDark ? CognithorTheme.textPrimary : const Color(0xFF1A1A2E);

  double _osc(double speed, [double phase = 0]) => sin(time * speed + phase);

  // ── Ceiling ────────────────────────────────────────────────────────

  void _drawCeiling(Canvas canvas, Size s) {
    final rect = Rect.fromLTWH(0, 0, s.width, s.height * 0.18);
    final paint = Paint()
      ..shader = ui.Gradient.linear(rect.topCenter, rect.bottomCenter, [
        _isDark ? const Color(0xFF12122a) : const Color(0xFFd8d8e8),
        _wallColor(),
      ]);
    canvas.drawRect(rect, paint);
  }

  // ── Back Wall ──────────────────────────────────────────────────────

  void _drawBackWall(Canvas canvas, Size s) {
    final wallTop = s.height * 0.18;
    final wallBottom = s.height * 0.45;
    final rect = Rect.fromLTWH(0, wallTop, s.width, wallBottom - wallTop);
    final paint = Paint()
      ..shader = ui.Gradient.linear(rect.topCenter, rect.bottomCenter, [
        _wallColor(),
        _wallColor().withValues(alpha: 0.95),
      ]);
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

  void _drawSun(
    Canvas canvas,
    double wx,
    double wy,
    double ww,
    double wh,
    double hour,
  ) {
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
      ..shader = ui.Gradient.radial(Offset(sunX, sunY), sunR * 4, [
        const Color(0x66FFD700),
        const Color(0x00FFD700),
      ]);
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

  void _drawMoon(
    Canvas canvas,
    double wx,
    double wy,
    double ww,
    double wh,
    double hour,
  ) {
    final moonX = wx + ww * 0.65;
    final moonY = wy + wh * 0.30;
    final moonR = ww * 0.055;

    // Moon glow
    final glowPaint = Paint()
      ..shader = ui.Gradient.radial(Offset(moonX, moonY), moonR * 5, [
        const Color(0x33C0C0FF),
        const Color(0x00C0C0FF),
      ]);
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
    if (_dayFactor < 0.1) return; // no window beams at night

    final wx = s.width * 0.32;
    final wy = s.height * 0.06;
    final ww = s.width * 0.36;
    final wh = s.height * 0.30;

    final beamAlpha = (_isDark ? 0.06 : 0.04) * _dayFactor;
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

    // Gold neon glow
    final glowPaint = Paint()
      ..color = const Color(
        0xFFFFD700,
      ).withValues(alpha: 0.20 + 0.08 * _osc(1.5))
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 14);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(sx, sy),
          width: s.width * 0.18,
          height: 22,
        ),
        const Radius.circular(4),
      ),
      glowPaint,
    );

    // Sign background
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(sx, sy),
          width: s.width * 0.18,
          height: 22,
        ),
        const Radius.circular(4),
      ),
      Paint()
        ..color = (_isDark ? const Color(0xFF1a1a2e) : const Color(0xFFd0d0dc)),
    );

    // Text
    final tp = TextPainter(
      text: const TextSpan(
        text: text,
        style: TextStyle(
          color: Color(0xFFFFD700),
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
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, bh),
        const Radius.circular(3),
      ),
      Paint()..color = Colors.white.withValues(alpha: _isDark ? 0.9 : 1.0),
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, bh),
        const Radius.circular(3),
      ),
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
    canvas.drawRect(
      Rect.fromLTWH(bx + bw * 0.1, by + bh * 0.15, bw * 0.25, bh * 0.25),
      pen,
    );
    canvas.drawRect(
      Rect.fromLTWH(bx + bw * 0.55, by + bh * 0.15, bw * 0.25, bh * 0.25),
      pen,
    );
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
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, bh),
        const Radius.circular(3),
      ),
      Paint()
        ..color = _isDark ? const Color(0xFF2a2a44) : const Color(0xFFe0e0e8),
    );

    // ── "Sprint Board" header strip ──
    final headerH = bh * 0.10;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, headerH),
        const Radius.circular(3),
      ),
      Paint()..color = CognithorTheme.accent.withValues(alpha: 0.25),
    );
    final headerTp = TextPainter(
      text: TextSpan(
        text: 'Sprint Board',
        style: TextStyle(
          color: _textCol(),
          fontSize: 4.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    headerTp.paint(canvas, Offset(bx + (bw - headerTp.width) / 2, by + 1));

    // Column headers below the sprint header
    final colW = bw / 3;
    final colHeaderY = by + headerH;
    final colHeaderH = bh * 0.10;
    final headers = ['To Do', 'WIP', 'Done'];
    final headerColors = [
      CognithorTheme.orange,
      CognithorTheme.accent,
      CognithorTheme.green,
    ];
    // Map activePhase (0-4) to kanban columns: 0-1 = To Do, 2-3 = WIP, 4 = Done
    final highlightCol = activePhase <= 1 ? 0 : (activePhase <= 3 ? 1 : 2);
    for (int c = 0; c < 3; c++) {
      final hx = bx + c * colW;
      final isActive = c == highlightCol && activePhase > 0;
      canvas.drawRect(
        Rect.fromLTWH(hx, colHeaderY, colW, colHeaderH),
        Paint()
          ..color = headerColors[c].withValues(alpha: isActive ? 0.55 : 0.25),
      );
      // Active column glow highlight
      if (isActive) {
        canvas.drawRect(
          Rect.fromLTWH(
            hx,
            colHeaderY + colHeaderH,
            colW,
            bh - headerH - colHeaderH,
          ),
          Paint()
            ..color = headerColors[c].withValues(
              alpha: 0.08 + 0.04 * _osc(2.0),
            ),
        );
      }
      final tp = TextPainter(
        text: TextSpan(
          text: headers[c],
          style: TextStyle(
            color: _textCol(),
            fontSize: 4.5,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(hx + (colW - tp.width) / 2, colHeaderY + 1.5));

      // Column separator line
      if (c > 0) {
        canvas.drawLine(
          Offset(hx, colHeaderY + colHeaderH),
          Offset(hx, by + bh),
          Paint()
            ..color = _furnitureDark().withValues(alpha: 0.4)
            ..strokeWidth = 0.5,
        );
      }

      // Data-driven kanban dots
      final statusKeys = c == 0
          ? ['backlog']
          : c == 1
          ? ['in_progress']
          : ['done', 'verifying'];
      // Also include 'blocked' in column 1 (WIP)
      final dotStatusColors = <String, Color>{
        'backlog': const Color(0xFF9CA3AF),
        'in_progress': const Color(0xFF3B82F6),
        'verifying': const Color(0xFFF59E0B),
        'done': const Color(0xFF22C55E),
        'blocked': const Color(0xFFEF4444),
      };
      final dotStartY = colHeaderY + colHeaderH + 2;
      final dotRadius = colW * 0.12;
      final dotSpacingX = dotRadius * 2.8;
      final dotSpacingY = dotRadius * 2.8;
      int dotIndex = 0;
      int totalForCol = 0;
      for (final key in statusKeys) {
        totalForCol += kanbanCounts[key] ?? 0;
      }
      const maxDots = 8;
      for (final key in statusKeys) {
        final count = kanbanCounts[key] ?? 0;
        final color = dotStatusColors[key] ?? const Color(0xFF9CA3AF);
        for (int d = 0; d < count && dotIndex < maxDots; d++) {
          final col2 = dotIndex % 2;
          final row2 = dotIndex ~/ 2;
          final dx = hx + colW * 0.25 + col2 * dotSpacingX;
          final dy = dotStartY + row2 * dotSpacingY;
          canvas.drawCircle(Offset(dx, dy), dotRadius, Paint()..color = color);
          dotIndex++;
        }
      }
      // Overflow indicator: "+N" if more than 8 tasks
      if (totalForCol > maxDots) {
        final overflow = totalForCol - maxDots;
        final overflowTp = TextPainter(
          text: TextSpan(
            text: '+$overflow',
            style: TextStyle(
              color: _textCol(),
              fontSize: 3.5,
              fontWeight: FontWeight.w600,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        overflowTp.paint(
          canvas,
          Offset(
            hx + colW * 0.25 + 2 * dotSpacingX,
            dotStartY + (maxDots ~/ 2 - 1) * dotSpacingY - 1,
          ),
        );
      }
    }

    // Border
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, bh),
        const Radius.circular(3),
      ),
      Paint()
        ..color = _furnitureDark()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );
  }

  // ── Ceiling Lights ─────────────────────────────────────────────────

  void _drawCeilingLights(Canvas canvas, Size s) {
    final nightIntensity = 1.0 - _dayFactor; // 1.0 at night, 0.0 during day
    // systemLoad boosts brightness (0.0 = normal, 1.0 = max brightness)
    final loadBoost = systemLoad * 0.3;
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

      // Light strip glow — brighter at night and when system load is high
      final baseAlpha = nightIntensity * 0.3 + 0.03 + loadBoost;
      final glowAlpha = (baseAlpha + 0.02 * _osc(0.8, i * 2.0)).clamp(0.0, 1.0);
      final glowColor = (nightIntensity > 0.3 || systemLoad > 0.5)
          ? const Color(0xFFFFE0A0) // warm yellow at night or high load
          : Colors.white;
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(lx, ly + 10),
          width: lw * 1.3,
          height: 30,
        ),
        Paint()
          ..color = glowColor.withValues(alpha: glowAlpha)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 15),
      );

      // Warm light strip on fixture — visible at night or under high load
      if (nightIntensity > 0.1 || systemLoad > 0.2) {
        final stripAlpha = ((nightIntensity + systemLoad) * 0.6).clamp(
          0.0,
          1.0,
        );
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromCenter(
              center: Offset(lx, ly + 1),
              width: lw * 0.9,
              height: 3,
            ),
            const Radius.circular(1.5),
          ),
          Paint()
            ..color = const Color(0xFFFFE0A0).withValues(alpha: stripAlpha),
        );
      }
    }
  }

  // ── Floor ──────────────────────────────────────────────────────────

  void _drawFloor(Canvas canvas, Size s) {
    final floorTop = s.height * 0.45;
    final floorHeight = s.height - floorTop;

    // Base floor gradient — warm light gray laminate
    final floorPaint = Paint()
      ..shader = ui.Gradient.linear(Offset(0, floorTop), Offset(0, s.height), [
        _floorBase(),
        _floorAlt(),
      ]);
    canvas.drawRect(
      Rect.fromLTWH(0, floorTop, s.width, floorHeight),
      floorPaint,
    );

    // Laminate plank pattern (horizontal long planks)
    final plankH = s.height * 0.035;
    final plankRows = (floorHeight / plankH).ceil() + 1;
    final plankGap = Paint()
      ..color = (_isDark ? const Color(0xFF252538) : const Color(0xFFC8C0B8))
      ..strokeWidth = 0.7;
    final grainLine = Paint()
      ..color = (_isDark ? Colors.white : Colors.black).withValues(alpha: 0.03)
      ..strokeWidth = 0.3;

    for (int r = 0; r < plankRows; r++) {
      final py = floorTop + r * plankH;
      if (py > s.height) break;
      // Horizontal plank seam
      canvas.drawLine(Offset(0, py), Offset(s.width, py), plankGap);

      // Staggered vertical joints (offset every other row)
      final plankW = s.width / 4;
      final offset = (r.isOdd ? plankW * 0.5 : 0.0);
      for (double jx = offset; jx < s.width; jx += plankW) {
        canvas.drawLine(Offset(jx, py), Offset(jx, py + plankH), plankGap);
      }

      // Subtle wood-grain direction lines within planks
      for (int g = 0; g < 3; g++) {
        final gy = py + plankH * (0.25 + g * 0.25);
        if (gy < s.height) {
          canvas.drawLine(Offset(0, gy), Offset(s.width, gy), grainLine);
        }
      }
    }

    // Subtle reflections
    final reflPaint = Paint()
      ..color = Colors.white.withValues(alpha: _isDark ? 0.02 : 0.03)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(s.width * 0.5, s.height * 0.65),
        width: s.width * 0.5,
        height: s.height * 0.12,
      ),
      reflPaint,
    );
  }

  // ── Carpet Area (under coffee machine) ─────────────────────────────

  void _drawCarpetArea(Canvas canvas, Size s) {
    final cx = s.width * _coffee.dx;
    final cy = s.height * _coffee.dy + s.height * 0.06;
    final cw = s.width * 0.12;
    final ch = s.height * 0.08;

    // Warm-toned area rug
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(cx, cy), width: cw, height: ch),
        const Radius.circular(3),
      ),
      Paint()
        ..color = (_isDark ? const Color(0xFF3D2B2B) : const Color(0xFFC4A882))
            .withValues(alpha: 0.6),
    );
    // Rug border pattern
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(cx, cy), width: cw - 4, height: ch - 4),
        const Radius.circular(2),
      ),
      Paint()
        ..color = (_isDark ? const Color(0xFF5C3A3A) : const Color(0xFFB8956E))
            .withValues(alpha: 0.4)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );
    // Inner rug pattern line
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(cx, cy),
          width: cw - 10,
          height: ch - 10,
        ),
        const Radius.circular(1),
      ),
      Paint()
        ..color = (_isDark ? const Color(0xFF6B4444) : const Color(0xFFA07850))
            .withValues(alpha: 0.3)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 0.8,
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
    path.moveTo(
      s.width * _desk2.dx + s.width * 0.06,
      s.height * _desk2.dy + s.height * 0.05,
    );
    path.quadraticBezierTo(
      s.width * 0.75,
      s.height * 0.55,
      s.width * _server.dx,
      s.height * _server.dy + s.height * 0.12,
    );
    canvas.drawPath(path, cablePaint);

    // Cable from desk1 to desk2 (back row)
    final path2 = Path();
    path2.moveTo(
      s.width * _desk1.dx + s.width * 0.06,
      s.height * _desk1.dy + s.height * 0.05,
    );
    path2.quadraticBezierTo(
      s.width * 0.32,
      s.height * 0.58,
      s.width * _desk2.dx - s.width * 0.02,
      s.height * _desk2.dy + s.height * 0.05,
    );
    canvas.drawPath(path2, cablePaint);

    // Cable from desk5 to server (front row connection)
    final path3 = Path();
    path3.moveTo(
      s.width * _desk5.dx + s.width * 0.06,
      s.height * _desk5.dy + s.height * 0.05,
    );
    path3.quadraticBezierTo(
      s.width * 0.80,
      s.height * 0.62,
      s.width * _server.dx,
      s.height * _server.dy + s.height * 0.18,
    );
    canvas.drawPath(path3, cablePaint);
  }

  // ── Desks ──────────────────────────────────────────────────────────

  void _drawDesks(Canvas canvas, Size s) {
    final desks = [_desk1, _desk2, _desk3, _desk4, _desk5];
    for (int i = 0; i < desks.length; i++) {
      _drawDesk(canvas, s, desks[i], i);
    }
  }

  void _drawDesk(Canvas canvas, Size s, Offset pos, int index) {
    final dx = s.width * pos.dx;
    final dy = s.height * pos.dy;
    final dw = s.width * 0.12;
    final dh = s.height * 0.06;
    final legH = s.height * 0.06;

    // Wood colors
    const oakTop = Color(0xFFB8913A); // lighter oak desktop
    const oakEdge = Color(0xFF8B6914); // darker oak edge
    const legColor = Color(0xFF6B4F1A); // dark brown legs

    // ── 4 wooden legs (slightly angled outward) ──
    const legW = 3.5;
    final legPaint = Paint()..color = legColor;
    // Front-left
    canvas.drawRect(
      Rect.fromLTWH(dx - dw / 2 + 4, dy + dh, legW, legH),
      legPaint,
    );
    // Front-right
    canvas.drawRect(
      Rect.fromLTWH(dx + dw / 2 - 4 - legW, dy + dh, legW, legH),
      legPaint,
    );
    // Back-left (slightly inset since it's the back)
    canvas.drawRect(
      Rect.fromLTWH(dx - dw / 2 + 7, dy + dh, legW - 0.5, legH * 0.3),
      legPaint,
    );
    // Back-right
    canvas.drawRect(
      Rect.fromLTWH(
        dx + dw / 2 - 7 - legW + 0.5,
        dy + dh,
        legW - 0.5,
        legH * 0.3,
      ),
      legPaint,
    );

    // ── Wood desktop surface ──
    final deskRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(dx - dw / 2, dy, dw, dh),
      const Radius.circular(2),
    );
    // Main wood surface with gradient for grain feel
    canvas.drawRRect(
      deskRect,
      Paint()
        ..shader = ui.Gradient.linear(
          Offset(dx - dw / 2, dy),
          Offset(dx + dw / 2, dy + dh),
          [
            oakTop,
            oakTop.withValues(alpha: 0.85),
            oakEdge.withValues(alpha: 0.6),
          ],
          [0.0, 0.7, 1.0],
        ),
    );
    // Subtle wood grain lines
    final grainPaint = Paint()
      ..color = oakEdge.withValues(alpha: 0.12)
      ..strokeWidth = 0.5;
    for (int g = 0; g < 4; g++) {
      final gy = dy + dh * (0.2 + g * 0.2);
      canvas.drawLine(
        Offset(dx - dw / 2 + 3, gy),
        Offset(dx + dw / 2 - 3, gy),
        grainPaint,
      );
    }
    // Desk edge/lip
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(dx - dw / 2, dy + dh - 2.5, dw, 3),
        const Radius.circular(1),
      ),
      Paint()..color = oakEdge,
    );

    // ── Flat-screen monitor with bezel ──
    final monW = dw * 0.5;
    final monH = dh * 0.85;
    final monX = dx - monW / 2;
    final monY = dy - monH - 6;
    const bezel = 2.0;

    // Monitor stand arm (thin metal)
    canvas.drawRect(
      Rect.fromLTWH(dx - 1.5, monY + monH, 3, 6),
      Paint()..color = const Color(0xFF606080),
    );
    // Stand base
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(dx - 10, monY + monH + 5, 20, 2.5),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFF606080),
    );

    // Outer monitor frame (bezel)
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(
          monX - bezel,
          monY - bezel,
          monW + bezel * 2,
          monH + bezel * 2,
        ),
        const Radius.circular(3),
      ),
      Paint()..color = const Color(0xFF222233),
    );

    // Screen background
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(monX, monY, monW, monH),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFF0C0C1A),
    );

    // Webcam dot on top bezel
    canvas.drawCircle(
      Offset(dx, monY - bezel / 2 - 0.5),
      1.2,
      Paint()..color = const Color(0xFF333355),
    );
    // Webcam active indicator (subtle green)
    canvas.drawCircle(
      Offset(dx, monY - bezel / 2 - 0.5),
      0.7,
      Paint()
        ..color = const Color(
          0xFF00E676,
        ).withValues(alpha: 0.3 + 0.2 * _osc(1.5, index * 3.0)),
    );

    // Animated screen content — colored code lines / UI mockup
    final lineRng = Random(index * 7 + 13);
    final codeColors = [
      CognithorTheme.accent,
      const Color(0xFFFF80AB), // pink
      const Color(0xFFB388FF), // purple
      const Color(0xFF80CBC4), // teal
      const Color(0xFFFFD54F), // yellow (keywords)
      const Color(0xFF81C784), // green (strings)
    ];
    for (int l = 0; l < 6; l++) {
      final ly = monY + 2.5 + l * (monH - 5) / 6;
      final indent = lineRng.nextInt(3) * monW * 0.06;
      final lineW = monW * (0.2 + lineRng.nextDouble() * 0.4);
      final scrollOffset = (time * 6 + index * 15) % (monH * 2);
      final adjustedY = ly + scrollOffset % 2.5;
      if (adjustedY < monY + monH - 2) {
        final colorIdx = (l + index * 2) % codeColors.length;
        canvas.drawLine(
          Offset(monX + 3 + indent, adjustedY),
          Offset(monX + 3 + indent + lineW, adjustedY),
          Paint()
            ..color = codeColors[colorIdx].withValues(
              alpha: 0.45 + 0.15 * _osc(3, l + index * 5.0),
            )
            ..strokeWidth = 1.0,
        );
        // Second segment on same line (different color for syntax highlighting)
        if (lineRng.nextDouble() > 0.4) {
          final seg2W = monW * (0.1 + lineRng.nextDouble() * 0.2);
          final seg2Color = codeColors[(colorIdx + 2) % codeColors.length];
          final seg2Start = monX + 3 + indent + lineW + 2;
          if (seg2Start + seg2W < monX + monW - 2) {
            canvas.drawLine(
              Offset(seg2Start, adjustedY),
              Offset(seg2Start + seg2W, adjustedY),
              Paint()
                ..color = seg2Color.withValues(alpha: 0.35)
                ..strokeWidth = 1.0,
            );
          }
        }
      }
    }

    // Screen glow — brighter at night
    final nightBoost = (1.0 - _dayFactor) * 0.08;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(monX, monY, monW, monH),
        const Radius.circular(1),
      ),
      Paint()
        ..color = CognithorTheme.accent.withValues(
          alpha: 0.03 + nightBoost + 0.02 * _osc(2, index.toDouble()),
        )
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, 5 + nightBoost * 25),
    );

    // ── Keyboard (small dark rectangle with key lines) ──
    final kbW = dw * 0.35;
    final kbH = dh * 0.22;
    final kbX = dx - kbW / 2;
    final kbY = dy + 3;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(kbX, kbY, kbW, kbH),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFF3A3A50),
    );
    // Key rows
    for (int kr = 0; kr < 2; kr++) {
      canvas.drawLine(
        Offset(kbX + 1, kbY + 2 + kr * (kbH * 0.4)),
        Offset(kbX + kbW - 1, kbY + 2 + kr * (kbH * 0.4)),
        Paint()
          ..color = const Color(0xFF555570)
          ..strokeWidth = 0.4,
      );
    }

    // ── Mouse (tiny oval to the right of keyboard) ──
    canvas.drawOval(
      Rect.fromLTWH(dx + kbW / 2 + 3, kbY + 1, 5, 7),
      Paint()..color = const Color(0xFF444460),
    );

    // ── Desk items vary by index ──
    if (index == 0 || index == 3) {
      // Coffee cup
      final cx = dx + dw * 0.3;
      final cy = dy + 2;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(cx, cy, 5, 7),
          const Radius.circular(1),
        ),
        Paint()..color = const Color(0xFFDDDDDD),
      );
      // Coffee surface
      canvas.drawRect(
        Rect.fromLTWH(cx + 0.8, cy + 0.8, 3.4, 2.5),
        Paint()..color = const Color(0xFF6B4226),
      );
      // Cup handle
      canvas.drawArc(
        Rect.fromLTWH(cx + 4, cy + 2, 3, 4),
        -1.2,
        2.4,
        false,
        Paint()
          ..color = const Color(0xFFCCCCCC)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 0.8,
      );
    } else if (index == 1 || index == 4) {
      // Post-it notes stack
      final colors = [
        const Color(0xFFFFEB3B),
        const Color(0xFFFF80AB),
        const Color(0xFF80D8FF),
      ];
      for (int n = 0; n < 3; n++) {
        canvas.drawRect(
          Rect.fromLTWH(dx + dw * 0.25 + n * 2, dy + 2 + n * 1.2, 7, 7),
          Paint()..color = colors[n].withValues(alpha: 0.8),
        );
      }
    } else {
      // Pen holder
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(dx - dw * 0.35, dy + 2, 6, 9),
          const Radius.circular(1),
        ),
        Paint()..color = const Color(0xFF777790),
      );
      // Pens
      canvas.drawLine(
        Offset(dx - dw * 0.33, dy + 2),
        Offset(dx - dw * 0.30, dy - 4),
        Paint()
          ..color = CognithorTheme.red
          ..strokeWidth = 1,
      );
      canvas.drawLine(
        Offset(dx - dw * 0.31, dy + 2),
        Offset(dx - dw * 0.29, dy - 3),
        Paint()
          ..color = CognithorTheme.accent
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
      RRect.fromRectAndRadius(
        Rect.fromLTWH(rx, ry, rw, rh),
        const Radius.circular(3),
      ),
      Paint()..color = const Color(0xFF2a2a44),
    );
    // Rack border
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(rx, ry, rw, rh),
        const Radius.circular(3),
      ),
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

      // Blinking LEDs — speed scales with cpuUsage, color shifts with memoryUsage
      final blinkSpeed = 3.0 + cpuUsage * 8.0; // faster blink when CPU is high
      for (int led = 0; led < 3; led++) {
        final baseColors = memoryUsage > 0.8
            ? [CognithorTheme.orange, CognithorTheme.red, CognithorTheme.red]
            : [CognithorTheme.green, CognithorTheme.orange, CognithorTheme.red];
        final blink = sin(time * (blinkSpeed + led) + i * 1.5 + led * 2.1);
        final on = blink > (led == 2 ? 0.7 : 0.0); // red blinks less
        canvas.drawCircle(
          Offset(rx + 8 + led * 5, sy + sh / 2),
          1.8,
          Paint()
            ..color = on
                ? baseColors[led].withValues(alpha: 0.9)
                : const Color(0xFF333350),
        );
        // LED glow
        if (on) {
          canvas.drawCircle(
            Offset(rx + 8 + led * 5, sy + sh / 2),
            4,
            Paint()
              ..color = baseColors[led].withValues(alpha: 0.15)
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
      Paint()
        ..color = CognithorTheme.green.withValues(alpha: 0.6 + 0.4 * _osc(2)),
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
    // Large plant near entrance area (bottom-left)
    _drawPlant(canvas, s, s.width * 0.12, s.height * 0.82, 2);
    // Hanging plant removed — not realistic for office
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
    canvas.drawPath(potPath, Paint()..color = const Color(0xFFC67B5C));
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

  // ── Flowers (day/night bloom) ───────────────────────────────────

  void _drawFlowers(Canvas canvas, Size s) {
    // Flowers ON desk surfaces (positioned at desk top edge)
    final flowers = [
      // On desk 1 (right side of desk)
      _FlowerInfo(
        s.width * (_desk1.dx + 0.08),
        s.height * (_desk1.dy - 0.02),
        const Color(0xFFE53935),
        0,
      ),
      // On desk 3 (left side)
      _FlowerInfo(
        s.width * (_desk3.dx + 0.02),
        s.height * (_desk3.dy - 0.02),
        const Color(0xFF9C27B0),
        1,
      ),
      // On desk 5 (right side)
      _FlowerInfo(
        s.width * (_desk5.dx + 0.08),
        s.height * (_desk5.dy - 0.02),
        const Color(0xFFFFEB3B),
        2,
      ),
      // On desk 2 (left side)
      _FlowerInfo(
        s.width * (_desk2.dx + 0.02),
        s.height * (_desk2.dy - 0.02),
        const Color(0xFFE91E63),
        3,
      ),
    ];

    for (final f in flowers) {
      _drawFlowerPot(canvas, s, f.x, f.y, f.color, f.seed);
    }
  }

  void _drawFlowerPot(
    Canvas canvas,
    Size s,
    double fx,
    double fy,
    Color petalColor,
    int seed,
  ) {
    final potW = s.width * 0.018;
    final potH = s.height * 0.025;
    final sway = _osc(0.7 + seed * 0.15, seed * 3.0) * 2;

    // Bloom factor: 1.0 = fully open (day), 0.0 = closed bud (night)
    final bloom = _dayFactor;

    // Terracotta pot (trapezoid)
    final potPath = Path();
    potPath.moveTo(fx - potW * 0.5, fy);
    potPath.lineTo(fx - potW * 0.35, fy + potH);
    potPath.lineTo(fx + potW * 0.35, fy + potH);
    potPath.lineTo(fx + potW * 0.5, fy);
    potPath.close();
    canvas.drawPath(potPath, Paint()..color = const Color(0xFFC67B5C));

    // Pot rim
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(fx - potW * 0.55, fy - 1.5, potW * 1.1, 3),
        const Radius.circular(1),
      ),
      Paint()..color = const Color(0xFFB06B4C),
    );

    // Green stem
    final stemTop = fy - potH * 1.2;
    canvas.drawLine(
      Offset(fx + sway * 0.3, fy - 1),
      Offset(fx + sway, stemTop),
      Paint()
        ..color = const Color(0xFF388E3C)
        ..strokeWidth = 1.5
        ..strokeCap = StrokeCap.round,
    );

    // Small leaf on stem
    final leafPath = Path();
    final leafMidY = fy - potH * 0.6;
    leafPath.moveTo(fx + sway * 0.5, leafMidY);
    leafPath.quadraticBezierTo(
      fx + sway * 0.5 + 5,
      leafMidY - 3,
      fx + sway * 0.5 + 3,
      leafMidY - 6,
    );
    canvas.drawPath(
      leafPath,
      Paint()
        ..color = const Color(0xFF4CAF50)
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );

    // Flower head
    final flowerCx = fx + sway;
    final flowerCy = stemTop;
    const petalCount = 5;
    const centerR = 2.5;
    final petalLen = 5.0 * bloom + 1.5; // closed: small bud, open: full petals
    final petalWidth = 3.0 * bloom + 1.0;

    if (bloom > 0.05) {
      // Draw petals
      for (int p = 0; p < petalCount; p++) {
        final angle = (p / petalCount) * pi * 2 + seed * 1.2 + sway * 0.05;
        // Petal spread depends on bloom factor
        final petalEndX = flowerCx + cos(angle) * petalLen;
        final petalEndY = flowerCy + sin(angle) * petalLen;
        final ctrlX = flowerCx + cos(angle + 0.3) * petalLen * 0.6;
        final ctrlY = flowerCy + sin(angle + 0.3) * petalLen * 0.6;
        final ctrlX2 = flowerCx + cos(angle - 0.3) * petalLen * 0.6;
        final ctrlY2 = flowerCy + sin(angle - 0.3) * petalLen * 0.6;

        final pPath = Path();
        pPath.moveTo(flowerCx, flowerCy);
        pPath.quadraticBezierTo(ctrlX, ctrlY, petalEndX, petalEndY);
        pPath.quadraticBezierTo(ctrlX2, ctrlY2, flowerCx, flowerCy);
        canvas.drawPath(
          pPath,
          Paint()..color = petalColor.withValues(alpha: 0.8),
        );
      }
    } else {
      // Closed bud: draw a small oval
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(flowerCx, flowerCy),
          width: petalWidth * 1.2,
          height: petalWidth * 1.8,
        ),
        Paint()..color = petalColor.withValues(alpha: 0.6),
      );
    }

    // Yellow center (visible when open)
    if (bloom > 0.3) {
      canvas.drawCircle(
        Offset(flowerCx, flowerCy),
        centerR * bloom,
        Paint()..color = const Color(0xFFFFEB3B),
      );
    }
  }

  // ── Ceiling Light Beams (night-time downward cones) ──────────────

  void _drawCeilingLightBeams(Canvas canvas, Size s) {
    final nightIntensity = 1.0 - _dayFactor;
    if (nightIntensity < 0.1) return; // no beams during day

    for (int i = 0; i < 3; i++) {
      final lx = s.width * (0.22 + i * 0.28);
      final ly = s.height * 0.18;
      final beamW = s.width * 0.08;
      final beamH = s.height * 0.35;

      // Cone-shaped light beam downward
      final beamPath = Path();
      beamPath.moveTo(lx - beamW * 0.15, ly);
      beamPath.lineTo(lx - beamW, ly + beamH);
      beamPath.lineTo(lx + beamW, ly + beamH);
      beamPath.lineTo(lx + beamW * 0.15, ly);
      beamPath.close();

      final beamAlpha = nightIntensity * (0.04 + 0.01 * _osc(0.8, i * 2.0));
      canvas.drawPath(
        beamPath,
        Paint()
          ..shader =
              ui.Gradient.linear(Offset(lx, ly), Offset(lx, ly + beamH), [
                const Color(0xFFFFE0A0).withValues(alpha: beamAlpha * 2),
                const Color(0xFFFFE0A0).withValues(alpha: 0),
              ]),
      );

      // Floor light pool
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(lx, ly + beamH),
          width: beamW * 2.2,
          height: beamH * 0.15,
        ),
        Paint()
          ..color = const Color(
            0xFFFFE0A0,
          ).withValues(alpha: nightIntensity * 0.06)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 12),
      );
    }
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
    final walking =
        robot.state == RobotState.walking || robot.state == RobotState.carrying;
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
        center: Offset(
          rx - bodyW * 0.2 - legSwing,
          ry + bodyH * 0.3 + legH + 2 + bobY,
        ),
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
        center: Offset(
          rx + bodyW * 0.2 + legSwing,
          ry + bodyH * 0.3 + legH + 2 + bobY,
        ),
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
          [robot.type.color, Color.lerp(robot.type.color, Colors.black, 0.3)!],
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
    final ledPulse =
        0.5 + 0.5 * sin(time * 3 + robot.type.id.hashCode.toDouble());
    canvas.drawCircle(
      Offset(rx, ry + bodyH * 0.1 + bobY),
      3 * scale,
      Paint()
        ..color = robot.type.eyeColor.withValues(alpha: 0.3 + 0.5 * ledPulse),
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
          Rect.fromCenter(
            center: Offset(docX, docY),
            width: 8 * scale,
            height: 10 * scale,
          ),
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
          [Color.lerp(robot.type.color, Colors.white, 0.15)!, robot.type.color],
        ),
    );
    // Head highlight
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(
          rx - headW * 0.35,
          headY - headH * 0.4,
          headW * 0.3,
          headH * 0.35,
        ),
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
        Rect.fromCenter(
          center: Offset(eyeX, headY),
          width: eyeW * 1.8,
          height: eyeH * 2,
        ),
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
      smilePath.quadraticBezierTo(
        rx,
        mouthY + 3 * scale,
        rx + headW * 0.15,
        mouthY,
      );
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
      final tipPulse =
          0.5 + 0.5 * _osc(4, robot.type.id.hashCode.toDouble() + 1);
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
      _drawSpeechBubble(
        canvas,
        rx,
        headY - headH / 2 - 12 * scale,
        robot.speechText,
        scale,
      );
    }

    // Floating emoji
    if (robot.emojiTimer > 0 && robot.floatingEmoji.isNotEmpty) {
      final emojiY = headY - headH / 2 - 18 * scale - robot.emojiTimer * 8;
      final emojiAlpha = robot.emojiTimer.clamp(0.0, 1.0);
      final emojiTp = TextPainter(
        text: TextSpan(
          text: robot.floatingEmoji,
          style: TextStyle(
            fontSize: 12 * scale,
            color: Colors.white.withValues(alpha: emojiAlpha),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      emojiTp.paint(canvas, Offset(rx - emojiTp.width / 2, emojiY));
    }
  }

  void _drawSpeechBubble(
    Canvas canvas,
    double x,
    double y,
    String text,
    double scale,
  ) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: _textCol(),
          fontSize: 6 * scale,
          fontWeight: FontWeight.w500,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final bw = tp.width + 10 * scale;
    final bh = tp.height + 6 * scale;
    final bx = x - bw / 2;
    final by = y - bh;

    // Bubble
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, bh),
        Radius.circular(4 * scale),
      ),
      Paint()
        ..color = (_isDark ? const Color(0xFF2a2a44) : Colors.white).withValues(
          alpha: 0.92,
        ),
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(bx, by, bw, bh),
        Radius.circular(4 * scale),
      ),
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
      Paint()
        ..color = (_isDark ? const Color(0xFF2a2a44) : Colors.white).withValues(
          alpha: 0.92,
        ),
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
        _drawParticleStream(
          canvas,
          s,
          a.x,
          a.y,
          b.x,
          b.y,
          a.type.color,
          b.type.color,
        );
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

  // ── System Monitor (wall-mounted screen) ─────────────────────────

  void _drawSystemMonitor(Canvas canvas, Size s) {
    final mx = s.width * 0.74;
    final my = s.height * 0.22;
    final mw = s.width * 0.12;
    final mh = s.height * 0.20;

    // Monitor frame (dark rectangle with rounded corners)
    final frame = RRect.fromRectAndRadius(
      Rect.fromLTWH(mx, my, mw, mh),
      const Radius.circular(4),
    );
    canvas.drawRRect(frame, Paint()..color = const Color(0xFF1A1A2E));
    canvas.drawRRect(
      frame,
      Paint()
        ..color = const Color(0xFF2A3366)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );

    // Screen background (slightly lighter)
    const screenInset = 3.0;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(
          mx + screenInset,
          my + screenInset,
          mw - screenInset * 2,
          mh - screenInset * 2,
        ),
        const Radius.circular(2),
      ),
      Paint()..color = const Color(0xFF050510),
    );

    // Subtle scanline / CRT effect
    final scanPaint = Paint()..color = Colors.white.withValues(alpha: 0.02);
    for (double sy = my + screenInset; sy < my + mh - screenInset; sy += 3) {
      canvas.drawLine(
        Offset(mx + screenInset, sy),
        Offset(mx + mw - screenInset, sy),
        scanPaint,
      );
    }

    // Title "SYSTEM" text
    final titlePainter = TextPainter(
      text: TextSpan(
        text: 'SYSTEM',
        style: TextStyle(
          color: CognithorTheme.sectionDashboard.withValues(alpha: 0.7),
          fontSize: mh * 0.10,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.5,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    titlePainter.paint(
      canvas,
      Offset(mx + screenInset + 4, my + screenInset + 2),
    );

    // Bar layout — 4 bars: CPU, GPU, RAM, LOAD
    final barX = mx + screenInset + 4;
    final barW = mw - screenInset * 2 - 8;
    final barH = mh * 0.09;
    final barSpacing = mh * 0.22;
    final firstBarY = my + mh * 0.25;

    // GPU usage = approximate from systemLoad (no separate GPU metric yet)
    final gpuUsage = (systemLoad * 1.2).clamp(0.0, 1.0);

    // CPU
    _drawMonitorLabel(canvas, 'CPU', Offset(barX, firstBarY), mh * 0.07);
    _drawMonitorBar(
      canvas,
      barX,
      firstBarY + mh * 0.07,
      barW,
      barH,
      cpuUsage,
      cpuUsage > 0.8
          ? CognithorTheme.red
          : cpuUsage > 0.5
          ? CognithorTheme.orange
          : CognithorTheme.sectionDashboard,
    );

    // GPU
    final gpuBarY = firstBarY + barSpacing;
    _drawMonitorLabel(canvas, 'GPU', Offset(barX, gpuBarY), mh * 0.07);
    _drawMonitorBar(
      canvas,
      barX,
      gpuBarY + mh * 0.07,
      barW,
      barH,
      gpuUsage,
      gpuUsage > 0.8
          ? CognithorTheme.red
          : gpuUsage > 0.5
          ? CognithorTheme.orange
          : CognithorTheme.accent,
    );

    // RAM
    final ramBarY = gpuBarY + barSpacing;
    _drawMonitorLabel(canvas, 'RAM', Offset(barX, ramBarY), mh * 0.07);
    _drawMonitorBar(
      canvas,
      barX,
      ramBarY + mh * 0.07,
      barW,
      barH,
      memoryUsage,
      memoryUsage > 0.8
          ? CognithorTheme.red
          : memoryUsage > 0.5
          ? CognithorTheme.orange
          : CognithorTheme.blue,
    );

    // LOAD
    final loadBarY = ramBarY + barSpacing;
    _drawMonitorLabel(canvas, 'LOAD', Offset(barX, loadBarY), mh * 0.07);
    _drawMonitorBar(
      canvas,
      barX,
      loadBarY + mh * 0.07,
      barW,
      barH,
      systemLoad,
      systemLoad > 0.8
          ? CognithorTheme.red
          : systemLoad > 0.5
          ? CognithorTheme.orange
          : CognithorTheme.gold,
    );

    // Monitor stand (small trapezoid below the monitor)
    final standW = mw * 0.25;
    final standH = mh * 0.06;
    final standX = mx + mw / 2 - standW / 2;
    final standY = my + mh;
    canvas.drawRect(
      Rect.fromLTWH(standX, standY, standW, standH),
      Paint()..color = const Color(0xFF2A2A3E),
    );
  }

  void _drawMonitorLabel(
    Canvas canvas,
    String text,
    Offset pos,
    double fontSize,
  ) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white70,
          fontSize: fontSize,
          fontWeight: FontWeight.w500,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, pos);
  }

  void _drawMonitorBar(
    Canvas canvas,
    double x,
    double y,
    double w,
    double h,
    double value,
    Color color,
  ) {
    // Background track
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(x, y, w, h),
        const Radius.circular(2),
      ),
      Paint()..color = Colors.white.withValues(alpha: 0.08),
    );
    // Value fill
    final fillW = w * value.clamp(0.0, 1.0);
    if (fillW > 0) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(x, y, fillW, h),
          const Radius.circular(2),
        ),
        Paint()..color = color,
      );
      // Subtle glow
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(x, y - 1, fillW, h + 2),
          const Radius.circular(2),
        ),
        Paint()
          ..color = color.withValues(alpha: 0.3)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4),
      );
    }
    // Percentage text
    final pct = '${(value * 100).round()}%';
    final tp = TextPainter(
      text: TextSpan(
        text: pct,
        style: TextStyle(
          color: color,
          fontSize: h * 0.9,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, Offset(x + w + 3, y - 1));
  }
}

// ── Robot Office Widget (hosts the painter + animation + AI logic) ────

class RobotOffice extends StatefulWidget {
  const RobotOffice({super.key, this.isRunning = true});

  final bool isRunning;

  @override
  State<RobotOffice> createState() => _RobotOfficeState();
}

class _RobotOfficeState extends State<RobotOffice>
    with SingleTickerProviderStateMixin {
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
      Robot(type: _planner, x: _desk1.dx, y: _desk1.dy + 0.10),
      Robot(type: _executor, x: _desk2.dx, y: _desk2.dy + 0.10),
      Robot(type: _researcher, x: _desk3.dx, y: _desk3.dy + 0.10),
      Robot(type: _gatekeeper, x: _desk4.dx, y: _desk4.dy + 0.10),
      Robot(type: _coder, x: _desk5.dx, y: _desk5.dy + 0.10),
      Robot(type: _analyst, x: 0.80, y: 0.60),
      Robot(type: _memory, x: 0.75, y: 0.72),
      Robot(type: _ops, x: 0.50, y: 0.78),
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
    const dt = 1.0 / 60.0; // approximate dt
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
            robot.speechText =
                _speechMessages[_rng.nextInt(_speechMessages.length)];
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
      final desks = [_desk1, _desk2, _desk3, _desk4, _desk5];
      final desk = desks[_rng.nextInt(desks.length)];
      target = Offset(desk.dx, desk.dy + 0.10);
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
    robot.state = roll > 0.50 && roll < 0.55
        ? RobotState.carrying
        : RobotState.walking;
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
