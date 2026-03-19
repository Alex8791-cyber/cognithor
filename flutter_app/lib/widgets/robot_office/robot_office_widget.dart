import 'dart:math';

import 'package:flutter/material.dart';

import 'package:jarvis_ui/widgets/robot_office/furniture.dart';
import 'package:jarvis_ui/widgets/robot_office/robot.dart';
import 'package:jarvis_ui/widgets/robot_office/robot_office_painter.dart';

// ---------------------------------------------------------------------------
// Robot Office Widget — animated isometric office with robot agents
// ---------------------------------------------------------------------------

class RobotOfficeWidget extends StatefulWidget {
  const RobotOfficeWidget({
    super.key,
    this.isRunning = true,
    this.onTaskCompleted,
    this.onStateChanged,
  });

  final bool isRunning;
  final VoidCallback? onTaskCompleted;

  /// Notifies parent about current task text and total completed count.
  final void Function(String currentTask, int taskCount)? onStateChanged;

  @override
  State<RobotOfficeWidget> createState() => _RobotOfficeWidgetState();
}

class _RobotOfficeWidgetState extends State<RobotOfficeWidget>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late List<Robot> _robots;
  final _rng = Random();

  int _taskCount = 0;
  String _currentTask = 'Warte auf Aufgabe...';

  // ── Task message pool ───────────────────────────────────────
  static const _taskMessages = [
    'Kontext laden...',
    'API aufrufen...',
    'Daten parsen...',
    'Plan erstellen...',
    'Tool ausführen...',
    'Antwort prüfen...',
    'Memory speichern...',
    'Ergebnis validieren...',
    'Tokens zählen...',
    'Chain bauen...',
    'Prompt optimieren...',
    'Logs schreiben...',
  ];

  static const _emojis = [
    '✓',
    '⚡',
    '🔧',
    '📊',
    '💡',
    '🔍',
    '📝',
    '🚀',
  ];

  // ── Lifecycle ───────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _robots = _createRobots();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1), // loops forever
    )..addListener(_tick);

    if (widget.isRunning) _controller.repeat();
  }

  @override
  void didUpdateWidget(covariant RobotOfficeWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isRunning && !_controller.isAnimating) {
      _controller.repeat();
    } else if (!widget.isRunning && _controller.isAnimating) {
      _controller.stop();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // ── Robot factory ───────────────────────────────────────────

  List<Robot> _createRobots() {
    return [
      Robot(
        id: 'planner',
        name: 'Planner',
        color: const Color(0xFF6366f1),
        eyeColor: const Color(0xFFa5b4fc),
        role: 'Strategie',
        hasAntenna: true,
        x: 0.18,
        y: 0.55,
      ),
      Robot(
        id: 'executor',
        name: 'Executor',
        color: const Color(0xFF10b981),
        eyeColor: const Color(0xFF6ee7b7),
        role: 'Ausführung',
        x: 0.45,
        y: 0.40,
      ),
      Robot(
        id: 'researcher',
        name: 'Researcher',
        color: const Color(0xFFf59e0b),
        eyeColor: const Color(0xFFfcd34d),
        role: 'Recherche',
        hasAntenna: true,
        x: 0.71,
        y: 0.60,
      ),
      Robot(
        id: 'gatekeeper',
        name: 'Gatekeeper',
        color: const Color(0xFFef4444),
        eyeColor: const Color(0xFFfca5a5),
        role: 'Sicherheit',
        x: 0.55,
        y: 0.22,
      ),
    ];
  }

  // ── Per-frame update ────────────────────────────────────────

  double _elapsed = 0;
  DateTime _lastTick = DateTime.now();

  void _tick() {
    final now = DateTime.now();
    final dt = (now.difference(_lastTick).inMicroseconds / 1e6).clamp(0.0, 0.1);
    _lastTick = now;
    _elapsed += dt;

    for (final r in _robots) {
      _updateRobot(r, dt);
    }

    // Collision avoidance
    _resolveCollisions();

    setState(() {});
  }

  void _updateRobot(Robot r, double dt) {
    r.bobPhase += dt * 3.5;
    r.blinkTimer -= dt;
    if (r.blinkTimer <= 0) {
      r.blinkTimer = 2.0 + _rng.nextDouble() * 4.0;
    }
    r.msgTimer = (r.msgTimer - dt).clamp(0.0, double.infinity);
    r.emojiTimer = (r.emojiTimer - dt).clamp(0.0, double.infinity);
    r.stateTimer -= dt;

    switch (r.state) {
      case RobotState.idle:
        if (r.stateTimer <= 0) {
          _assignTask(r);
        }
      case RobotState.walking:
        _moveToTarget(r, dt);
        if (_atTarget(r)) {
          r.state = RobotState.working;
          r.stateTimer = 1.5 + _rng.nextDouble() * 2.5;
          r.typing = true;
        }
      case RobotState.working:
        if (r.stateTimer <= 0) {
          r.typing = false;
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 3.0;
          r.emoji = _emojis[_rng.nextInt(_emojis.length)];
          r.emojiTimer = 1.5;
          _taskCount++;
          widget.onTaskCompleted?.call();
          _currentTask = r.taskMsg.isNotEmpty ? r.taskMsg : _currentTask;
          widget.onStateChanged?.call(_currentTask, _taskCount);
        }
    }
  }

  void _assignTask(Robot r) {
    // Pick a random furniture target
    final targets = officeFurniture
        .where((f) => f.type == 'desk' || f.type == 'server' || f.type == 'coffee')
        .toList();
    final target = targets[_rng.nextInt(targets.length)];

    r.targetX = target.x + target.w / 2 + (_rng.nextDouble() - 0.5) * 0.04;
    r.targetY = target.y + target.h + 0.02 + _rng.nextDouble() * 0.03;
    r.targetX = r.targetX.clamp(0.05, 0.95);
    r.targetY = r.targetY.clamp(0.15, 0.90);

    r.state = RobotState.walking;
    r.stateTimer = 10; // safety timeout
    r.taskMsg = _taskMessages[_rng.nextInt(_taskMessages.length)];
    r.msgTimer = 3.0;
    _currentTask = r.taskMsg;
    widget.onStateChanged?.call(_currentTask, _taskCount);
  }

  void _moveToTarget(Robot r, double dt) {
    const speed = 0.15; // normalized units per second
    final dx = r.targetX - r.x;
    final dy = r.targetY - r.y;
    final dist = sqrt(dx * dx + dy * dy);
    if (dist < 0.005) {
      r.x = r.targetX;
      r.y = r.targetY;
      return;
    }
    final step = min(speed * dt, dist);
    r.x += dx / dist * step;
    r.y += dy / dist * step;
    r.facing = dx >= 0 ? 1 : -1;
  }

  bool _atTarget(Robot r) {
    final dx = r.targetX - r.x;
    final dy = r.targetY - r.y;
    return dx * dx + dy * dy < 0.005 * 0.005;
  }

  void _resolveCollisions() {
    const minDist = 0.06;
    for (var i = 0; i < _robots.length; i++) {
      for (var j = i + 1; j < _robots.length; j++) {
        final a = _robots[i];
        final b = _robots[j];
        final dx = b.x - a.x;
        final dy = b.y - a.y;
        final dist = sqrt(dx * dx + dy * dy);
        if (dist < minDist && dist > 0.001) {
          final overlap = (minDist - dist) / 2;
          final nx = dx / dist;
          final ny = dy / dist;
          a.x -= nx * overlap;
          a.y -= ny * overlap;
          b.x += nx * overlap;
          b.y += ny * overlap;
        }
      }
    }
  }

  // ── Build ───────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: CustomPaint(
        painter: RobotOfficePainter(
          robots: _robots,
          furniture: officeFurniture,
          elapsed: _elapsed,
        ),
        child: const SizedBox.expand(),
      ),
    );
  }
}
