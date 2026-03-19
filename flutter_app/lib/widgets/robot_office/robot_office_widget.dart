import 'dart:math';

import 'package:flutter/material.dart';

import 'package:jarvis_ui/widgets/robot_office/furniture.dart';
import 'package:jarvis_ui/widgets/robot_office/robot.dart';
import 'package:jarvis_ui/widgets/robot_office/office_painter.dart' as bg;
import 'package:jarvis_ui/widgets/robot_office/robot_office_painter.dart';

// ---------------------------------------------------------------------------
// Robot Office Widget — animated isometric office with robot agents
// ---------------------------------------------------------------------------

/// Funny German chat messages for robot conversations.
const _chatMessages = [
  'Hast du den neuen Prompt gesehen?',
  'Mein Context-Window ist voll...',
  'Wer hat den Server neugestartet?',
  'Token-Limit erreicht!',
  'Die API antwortet nicht...',
  'Kaffee?',
  'Ja bitte!',
  'Bug gefunden!',
  'Wo denn?',
  'Ich brauche mehr VRAM!',
  'Das Training dauert ewig...',
  'Hast du das geloggt?',
  'Wer hat meinen Prompt geaendert?',
  'Mittagspause?',
  'Gleich!',
  'Der Gatekeeper hat mich blockiert!',
  'Schon wieder ein Timeout...',
  'Mein Modell halluziniert!',
  'Hast du den Patch deployed?',
  'Ich compile seit Stunden...',
];

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
  late final OfficePet _dog;
  late final OfficePet _cat;
  final ParticleSystem _particles = ParticleSystem();
  final _rng = Random();

  int _taskCount = 0;
  String _currentTask = 'Warte auf Aufgabe...';

  // ── Task message pool ───────────────────────────────────────
  static const _taskMessages = [
    'Kontext laden...',
    'API aufrufen...',
    'Daten parsen...',
    'Plan erstellen...',
    'Tool ausfuehren...',
    'Antwort pruefen...',
    'Memory speichern...',
    'Ergebnis validieren...',
    'Tokens zaehlen...',
    'Chain bauen...',
    'Prompt optimieren...',
    'Logs schreiben...',
  ];

  // ── Emoji pools per state ────────────────────────────────────
  static const _workEmojis = ['⚡', '💡', '🔧', '✅', '📊', '🔬'];
  static const _napEmojis = ['😴', '💤', '🌙'];
  static const _coffeeEmojis = ['☕', '🫖'];
  static const _celebrateEmojis = ['🎉', '🏆', '🥳', '✨'];
  static const _prankEmojis = ['😈', '🤫'];
  static const _playEmojis = ['🏃', '🎮'];
  static const _danceEmojis = ['💃', '🕺', '🎵'];
  static const _thinkEmojis = ['🤔', '💭', '❓'];

  // ── Lifecycle ───────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _robots = _createRobots();
    _dog = OfficePet(
      type: PetType.dog,
      x: 0.25,
      y: 0.85,
      color: const Color(0xFF8B6914),
    );
    _cat = OfficePet(
      type: PetType.cat,
      x: 0.75,
      y: 0.30,
      color: const Color(0xFF9E9E9E),
    );
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1), // loops forever
    )..addListener(_tick);

    if (widget.isRunning) _controller.repeat();
  }

  @override
  void didUpdateWidget(covariant RobotOfficeWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Keep animation running always (robots animate in idle too)
    if (!_controller.isAnimating) _controller.repeat();

    // WAKE UP! When switching from idle to active, all robots rush to work
    if (widget.isRunning && !oldWidget.isRunning) {
      _wakeUpAll();
    }
  }

  /// All robots jump up and rush to their stations — frantic energy!
  void _wakeUpAll() {
    for (final r in _robots) {
      // Everyone gets "!" shock emoji
      r.emoji = '⚡';
      r.emojiTimer = 1.5;
      r.taskMsg = '!!! Anfrage eingetroffen !!!';
      r.msgTimer = 2.0;
      // Immediately assign work
      r.stateTimer = 0;
      r.typing = false;
      r.carrying = false;
      r.state = RobotState.idle; // will trigger _assignWorkBehavior on next tick
    }
    // Spark burst for the excitement
    _particles.emit(ParticleType.spark, 0.5, 0.4,
        const Color(0xFF00d4ff), count: 20);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // ── Robot factory ───────────────────────────────────────────

  List<Robot> _createRobots() {
    // Place robots at different positions, each with staggered timers
    // so they don't all start acting simultaneously
    return [
      Robot(
        id: 'planner', name: 'Planner',
        color: const Color(0xFF6366f1), eyeColor: const Color(0xFFa5b4fc),
        role: 'Strategie', hasAntenna: true,
        x: 0.18, y: 0.72,
        state: RobotState.working, typing: true,
        stateTimer: 3.0 + _rng.nextDouble() * 3,
      ),
      Robot(
        id: 'executor', name: 'Executor',
        color: const Color(0xFF10b981), eyeColor: const Color(0xFF6ee7b7),
        role: 'Ausfuehrung',
        x: 0.45, y: 0.58,
        state: RobotState.working, typing: true,
        stateTimer: 2.0 + _rng.nextDouble() * 4,
      ),
      Robot(
        id: 'researcher', name: 'Researcher',
        color: const Color(0xFFf59e0b), eyeColor: const Color(0xFFfcd34d),
        role: 'Recherche', hasAntenna: true,
        x: 0.72, y: 0.75,
        state: RobotState.working, typing: true,
        stateTimer: 1.5 + _rng.nextDouble() * 2,
      ),
      Robot(
        id: 'gatekeeper', name: 'Gatekeeper',
        color: const Color(0xFFef4444), eyeColor: const Color(0xFFfca5a5),
        role: 'Sicherheit',
        x: 0.88, y: 0.42,
        state: RobotState.idle,
        stateTimer: 0.5 + _rng.nextDouble(),
      ),
      Robot(
        id: 'coder', name: 'Coder',
        color: const Color(0xFF8b5cf6), eyeColor: const Color(0xFFc4b5fd),
        role: 'Programmierung',
        x: 0.30, y: 0.52,
        state: RobotState.walking,
        stateTimer: 1.0 + _rng.nextDouble() * 2,
        targetX: 0.72, targetY: 0.75,
      ),
      Robot(
        id: 'analyst', name: 'Analyst',
        color: const Color(0xFF06b6d4), eyeColor: const Color(0xFF67e8f9),
        role: 'Datenanalyse', hasAntenna: true,
        x: 0.08, y: 0.35,
        state: RobotState.thinking,
        stateTimer: 2.0 + _rng.nextDouble() * 2,
      ),
      Robot(
        id: 'memory', name: 'Memory',
        color: const Color(0xFFec4899), eyeColor: const Color(0xFFf9a8d4),
        role: 'Wissen',
        x: 0.58, y: 0.28,
        state: RobotState.coffeeBreak,
        stateTimer: 3.0 + _rng.nextDouble() * 2,
      ),
      Robot(
        id: 'ops', name: 'DevOps',
        color: const Color(0xFF84cc16), eyeColor: const Color(0xFFbef264),
        role: 'Infrastruktur', hasAntenna: true,
        x: 0.60, y: 0.80,
        state: RobotState.walking,
        stateTimer: 1.0 + _rng.nextDouble(),
        targetX: 0.88, targetY: 0.42,
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

    // Update pets
    _updatePet(_dog, dt);
    _updatePet(_cat, dt);

    // Update particles
    _particles.update(dt);

    // Emit data packets between working robots
    _emitDataPackets(dt);

    // Collision avoidance
    _resolveCollisions();

    setState(() {});
  }

  // ── Data packet emission ──────────────────────────────────
  double _dataPacketCooldown = 0;

  void _emitDataPackets(double dt) {
    _dataPacketCooldown -= dt;
    if (_dataPacketCooldown > 0) return;
    _dataPacketCooldown = 0.4 + _rng.nextDouble() * 0.6;

    final working = _robots.where((r) => r.state == RobotState.working).toList();
    if (working.length < 2) return;

    final sender = working[_rng.nextInt(working.length)];
    Robot receiver;
    do {
      receiver = working[_rng.nextInt(working.length)];
    } while (receiver == sender);

    _particles.emitDataPacket(sender.x, sender.y, receiver.x, receiver.y, sender.color);
  }

  void _updateRobot(Robot r, double dt) {
    r.bobPhase += dt * 3.5;
    r.blinkTimer -= dt;
    if (r.blinkTimer <= 0) {
      r.blinkTimer = 2.0 + _rng.nextDouble() * 4.0;
    }
    r.msgTimer = (r.msgTimer - dt).clamp(0.0, double.infinity);
    r.emojiTimer = (r.emojiTimer - dt).clamp(0.0, double.infinity);
    r.chatBubbleTimer = (r.chatBubbleTimer - dt).clamp(0.0, double.infinity);
    r.stateTimer -= dt;

    switch (r.state) {
      case RobotState.idle:
        if (r.stateTimer <= 0) {
          _assignRandomBehavior(r);
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
          r.emoji = _workEmojis[_rng.nextInt(_workEmojis.length)];
          r.emojiTimer = 1.5;
          _taskCount++;
          widget.onTaskCompleted?.call();
          _currentTask = r.taskMsg.isNotEmpty ? r.taskMsg : _currentTask;
          widget.onStateChanged?.call(_currentTask, _taskCount);
        }
      case RobotState.carrying:
        _moveToTarget(r, dt);
        if (_atTarget(r)) {
          r.carrying = false;
          r.state = RobotState.idle;
          r.stateTimer = 1.0 + _rng.nextDouble() * 2.0;
          r.emoji = '📦';
          r.emojiTimer = 1.2;
        }
      case RobotState.napping:
        // Emit Z particles periodically
        if ((_elapsed * 2).floor() % 2 == 0 && _rng.nextDouble() < 0.03) {
          _particles.emit(
            ParticleType.text,
            r.x,
            r.y - 0.05,
            const Color(0xFF90CAF9),
            text: 'Z',
            count: 1,
          );
        }
        if (r.stateTimer <= 0) {
          r.state = RobotState.idle;
          r.stateTimer = 1.5 + _rng.nextDouble() * 2.0;
          r.emoji = _napEmojis[_rng.nextInt(_napEmojis.length)];
          r.emojiTimer = 1.2;
        }
      case RobotState.chatting:
        // Alternate chat bubbles
        if (r.chatBubbleTimer <= 0 && r.stateTimer > 1.0) {
          r.chatBubble = _chatMessages[_rng.nextInt(_chatMessages.length)];
          r.chatBubbleTimer = 1.5 + _rng.nextDouble();
        }
        if (r.stateTimer <= 0) {
          r.chatBubble = '';
          r.chatBubbleTimer = 0;
          r.interactionPartner?.chatBubble = '';
          r.interactionPartner?.chatBubbleTimer = 0;
          r.interactionPartner?.state = RobotState.idle;
          r.interactionPartner?.stateTimer = 1.0 + _rng.nextDouble() * 2.0;
          r.interactionPartner = null;
          r.state = RobotState.idle;
          r.stateTimer = 1.5 + _rng.nextDouble() * 2.0;
        }
      case RobotState.playing:
        // Chase the partner
        if (r.interactionPartner != null) {
          final partner = r.interactionPartner!;
          final dx = partner.x - r.x;
          final dy = partner.y - r.y;
          final dist = sqrt(dx * dx + dy * dy);
          if (dist > 0.03) {
            const speed = 0.22;
            r.x += dx / dist * speed * dt;
            r.y += dy / dist * speed * dt;
            r.facing = dx >= 0 ? 1 : -1;
            // The partner runs away randomly
            partner.x += (_rng.nextDouble() - 0.5) * 0.1 * dt;
            partner.y += (_rng.nextDouble() - 0.5) * 0.1 * dt;
            partner.x = partner.x.clamp(0.05, 0.95);
            partner.y = partner.y.clamp(0.15, 0.90);
          }
        }
        if (r.stateTimer <= 0) {
          r.interactionPartner?.state = RobotState.idle;
          r.interactionPartner?.stateTimer = 1.0 + _rng.nextDouble() * 2.0;
          r.interactionPartner = null;
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
          r.emoji = _playEmojis[_rng.nextInt(_playEmojis.length)];
          r.emojiTimer = 1.2;
        }
      case RobotState.pranking:
        if (r.isPranker && r.interactionPartner != null) {
          // Sneak toward partner
          final partner = r.interactionPartner!;
          final dx = partner.x - r.x;
          final dy = partner.y - r.y;
          final dist = sqrt(dx * dx + dy * dy);
          if (dist > 0.04) {
            const speed = 0.08; // slow sneak
            r.x += dx / dist * speed * dt;
            r.y += dy / dist * speed * dt;
            r.facing = dx >= 0 ? 1 : -1;
          } else if (r.stateTimer < 1.5) {
            // Close enough - trigger the scare
            if (partner.emoji != '!' && partner.emojiTimer <= 0) {
              partner.emoji = '!';
              partner.emojiTimer = 1.5;
              _particles.emit(
                ParticleType.text,
                partner.x,
                partner.y - 0.06,
                const Color(0xFFFF5252),
                text: '!',
                count: 1,
              );
            }
          }
        }
        if (r.stateTimer <= 0) {
          if (r.isPranker) {
            r.emoji = _prankEmojis[_rng.nextInt(_prankEmojis.length)];
            r.emojiTimer = 1.2;
          }
          r.isPranker = false;
          r.interactionPartner?.state = RobotState.idle;
          r.interactionPartner?.stateTimer = 1.0 + _rng.nextDouble() * 2.0;
          r.interactionPartner = null;
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
        }
      case RobotState.celebrating:
        r.celebratePhase += dt;
        // Emit confetti periodically
        if (_rng.nextDouble() < 0.15) {
          _particles.emit(
            ParticleType.confetti,
            r.x,
            r.y - 0.06,
            Color.fromARGB(
              255,
              _rng.nextInt(256),
              _rng.nextInt(256),
              _rng.nextInt(256),
            ),
            count: 3,
          );
        }
        if (r.stateTimer <= 0) {
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
          r.emoji = _celebrateEmojis[_rng.nextInt(_celebrateEmojis.length)];
          r.emojiTimer = 1.5;
        }
      case RobotState.coffeeBreak:
        _moveToTarget(r, dt);
        if (_atTarget(r)) {
          // Sipping at coffee machine
          if (r.stateTimer <= 0) {
            r.state = RobotState.idle;
            r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
            r.emoji = _coffeeEmojis[_rng.nextInt(_coffeeEmojis.length)];
            r.emojiTimer = 1.5;
          }
        }
      case RobotState.stretching:
        if (r.stateTimer <= 0) {
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
        }
      case RobotState.highFive:
        if (r.interactionPartner != null) {
          // Move toward each other
          final partner = r.interactionPartner!;
          final dx = partner.x - r.x;
          final dy = partner.y - r.y;
          final dist = sqrt(dx * dx + dy * dy);
          if (dist > 0.06) {
            const speed = 0.18;
            r.x += dx / dist * speed * dt;
            r.y += dy / dist * speed * dt;
            r.facing = dx >= 0 ? 1 : -1;
          } else if (r.stateTimer < 1.0 && r.emojiTimer <= 0) {
            // High five moment
            r.emoji = '🙌';
            r.emojiTimer = 1.2;
            partner.emoji = '🙌';
            partner.emojiTimer = 1.2;
            _particles.emit(
              ParticleType.spark,
              (r.x + partner.x) / 2,
              (r.y + partner.y) / 2 - 0.04,
              const Color(0xFFFFD700),
              count: 8,
            );
          }
        }
        if (r.stateTimer <= 0) {
          r.interactionPartner?.state = RobotState.idle;
          r.interactionPartner?.stateTimer = 1.0 + _rng.nextDouble() * 2.0;
          r.interactionPartner = null;
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
        }
      case RobotState.dancing:
        r.dancePhase += dt * 6;
        if (r.stateTimer <= 0) {
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
          r.emoji = _danceEmojis[_rng.nextInt(_danceEmojis.length)];
          r.emojiTimer = 1.2;
        }
      case RobotState.thinking:
        // Emit question marks
        if (_rng.nextDouble() < 0.03) {
          _particles.emit(
            ParticleType.text,
            r.x,
            r.y - 0.05,
            const Color(0xFFFFD54F),
            text: '?',
            count: 1,
          );
        }
        if (r.stateTimer <= 0) {
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
          r.emoji = _thinkEmojis[_rng.nextInt(_thinkEmojis.length)];
          r.emojiTimer = 1.2;
        }
    }
  }

  // ── Behavior assignment with weighted random ───────────────

  void _assignRandomBehavior(Robot r) {
    // When NOT running (no user request): robots chill, play, rest
    // When running (user request active): robots work frantically
    if (!widget.isRunning) {
      _assignIdleBehavior(r);
      return;
    }
    _assignWorkBehavior(r);
  }

  /// Idle mode: robots relax, play, nap, chat — no real work.
  void _assignIdleBehavior(Robot r) {
    final roll = _rng.nextDouble() * 100;
    if (roll < 25) {
      _assignNap(r);
    } else if (roll < 45) {
      _assignChat(r);
    } else if (roll < 55) {
      _assignCoffeeBreak(r);
    } else if (roll < 65) {
      _assignPlayTag(r);
    } else if (roll < 75) {
      _assignDance(r);
    } else if (roll < 85) {
      _assignStretch(r);
    } else if (roll < 92) {
      _assignPrank(r);
    } else {
      _assignThink(r);
    }
  }

  /// Work mode: robots rush to desks, servers, kanban — frantic activity.
  void _assignWorkBehavior(Robot r) {
    final roll = _rng.nextDouble() * 100;
    if (roll < 35) {
      _assignWorkAtDesk(r);
    } else if (roll < 50) {
      _assignWalk(r);
    } else if (roll < 60) {
      _assignCarry(r);
    } else if (roll < 70) {
      _assignKanban(r);
    } else if (roll < 80) {
      _assignCelebrate(r);
    } else if (roll < 90) {
      _assignHighFive(r);
    } else {
      _assignCoffeeBreak(r);
    }
  }

  void _assignWorkAtDesk(Robot r) {
    final desks = officeFurniture.where((f) => f.type == 'desk').toList();
    final desk = desks[_rng.nextInt(desks.length)];
    r.targetX = (desk.x + desk.w / 2 + (_rng.nextDouble() - 0.5) * 0.04).clamp(0.05, 0.95);
    r.targetY = (desk.y + desk.h + 0.02 + _rng.nextDouble() * 0.03).clamp(0.15, 0.90);
    r.state = RobotState.walking;
    r.stateTimer = 10;
    r.taskMsg = _taskMessages[_rng.nextInt(_taskMessages.length)];
    r.msgTimer = 3.0;
    _currentTask = r.taskMsg;
    widget.onStateChanged?.call(_currentTask, _taskCount);
  }

  void _assignWalk(Robot r) {
    final targets = officeFurniture
        .where((f) => f.type == 'desk' || f.type == 'server' || f.type == 'board')
        .toList();
    final target = targets[_rng.nextInt(targets.length)];
    r.targetX = (target.x + target.w / 2 + (_rng.nextDouble() - 0.5) * 0.04).clamp(0.05, 0.95);
    r.targetY = (target.y + target.h + 0.02 + _rng.nextDouble() * 0.03).clamp(0.15, 0.90);
    r.state = RobotState.walking;
    r.stateTimer = 10;
    r.taskMsg = _taskMessages[_rng.nextInt(_taskMessages.length)];
    r.msgTimer = 2.5;
  }

  void _assignNap(Robot r) {
    r.state = RobotState.napping;
    r.stateTimer = 5.0 + _rng.nextDouble() * 5.0;
    r.emoji = '😴';
    r.emojiTimer = 2.0;
  }

  void _assignChat(Robot r) {
    final partner = _findNearestAvailable(r);
    if (partner == null) {
      _assignWorkAtDesk(r);
      return;
    }
    r.state = RobotState.chatting;
    r.stateTimer = 4.0 + _rng.nextDouble() * 3.0;
    r.interactionPartner = partner;
    r.chatBubble = _chatMessages[_rng.nextInt(_chatMessages.length)];
    r.chatBubbleTimer = 1.5;

    partner.state = RobotState.chatting;
    partner.stateTimer = r.stateTimer;
    partner.interactionPartner = r;
    partner.chatBubbleTimer = 0.8; // offset so they alternate

    // Face each other
    r.facing = partner.x > r.x ? 1 : -1;
    partner.facing = r.x > partner.x ? 1 : -1;
  }

  void _assignPlayTag(Robot r) {
    final partner = _findNearestAvailable(r);
    if (partner == null) {
      _assignDance(r);
      return;
    }
    r.state = RobotState.playing;
    r.stateTimer = 4.0 + _rng.nextDouble() * 2.0;
    r.interactionPartner = partner;

    partner.state = RobotState.playing;
    partner.stateTimer = r.stateTimer;
    partner.interactionPartner = r;
  }

  void _assignPrank(Robot r) {
    final partner = _findNearestAvailable(r);
    if (partner == null) {
      _assignThink(r);
      return;
    }
    r.state = RobotState.pranking;
    r.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
    r.isPranker = true;
    r.interactionPartner = partner;
    r.emoji = '🤫';
    r.emojiTimer = 1.5;

    partner.state = RobotState.pranking;
    partner.stateTimer = r.stateTimer;
    partner.interactionPartner = r;
    partner.isPranker = false;
  }

  void _assignCelebrate(Robot r) {
    r.state = RobotState.celebrating;
    r.stateTimer = 2.5 + _rng.nextDouble() * 1.5;
    r.celebratePhase = 0;
    // Initial confetti burst
    _particles.emit(
      ParticleType.confetti,
      r.x,
      r.y - 0.06,
      Colors.amber,
      count: 20,
    );
  }

  void _assignCoffeeBreak(Robot r) {
    final coffeeSpots = officeFurniture.where((f) => f.type == 'coffee').toList();
    if (coffeeSpots.isEmpty) {
      _assignWorkAtDesk(r);
      return;
    }
    final spot = coffeeSpots[_rng.nextInt(coffeeSpots.length)];
    r.targetX = (spot.x + spot.w / 2).clamp(0.05, 0.95);
    r.targetY = (spot.y + spot.h + 0.03).clamp(0.15, 0.90);
    r.state = RobotState.coffeeBreak;
    r.stateTimer = 3.0 + _rng.nextDouble() * 3.0;
    r.emoji = '☕';
    r.emojiTimer = 2.0;
    r.msgTimer = 2.0;
    r.taskMsg = 'Kaffeepause!';
  }

  void _assignStretch(Robot r) {
    r.state = RobotState.stretching;
    r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
  }

  void _assignHighFive(Robot r) {
    final partner = _findNearestAvailable(r);
    if (partner == null) {
      _assignStretch(r);
      return;
    }
    r.state = RobotState.highFive;
    r.stateTimer = 2.0 + _rng.nextDouble() * 1.5;
    r.interactionPartner = partner;

    partner.state = RobotState.highFive;
    partner.stateTimer = r.stateTimer;
    partner.interactionPartner = r;
  }

  void _assignDance(Robot r) {
    r.state = RobotState.dancing;
    r.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
    r.dancePhase = _rng.nextDouble() * 6.28;
    r.emoji = _danceEmojis[_rng.nextInt(_danceEmojis.length)];
    r.emojiTimer = 1.5;
  }

  void _assignThink(Robot r) {
    r.state = RobotState.thinking;
    r.stateTimer = 3.0 + _rng.nextDouble() * 3.0;
    r.emoji = '🤔';
    r.emojiTimer = 2.0;
  }

  void _assignCarry(Robot r) {
    final servers = officeFurniture.where((f) => f.type == 'server').toList();
    if (servers.isEmpty) {
      _assignWalk(r);
      return;
    }
    final server = servers[_rng.nextInt(servers.length)];
    r.targetX = (server.x + server.w / 2).clamp(0.05, 0.95);
    r.targetY = (server.y + server.h + 0.03).clamp(0.15, 0.90);
    r.state = RobotState.carrying;
    r.carrying = true;
    r.stateTimer = 10;
    r.taskMsg = 'Dokument sichern...';
    r.msgTimer = 2.5;
  }

  void _assignKanban(Robot r) {
    final boards = officeFurniture.where((f) => f.type == 'board').toList();
    if (boards.isEmpty) {
      _assignWalk(r);
      return;
    }
    final board = boards[_rng.nextInt(boards.length)];
    r.targetX = (board.x + board.w / 2 + 0.02).clamp(0.05, 0.95);
    r.targetY = (board.y + board.h + 0.04).clamp(0.15, 0.90);
    r.state = RobotState.walking;
    r.stateTimer = 10;
    r.taskMsg = 'Board aktualisieren...';
    r.msgTimer = 2.5;
  }

  /// Find the nearest robot that is currently idle or working (available).
  Robot? _findNearestAvailable(Robot r) {
    Robot? nearest;
    double nearestDist = double.infinity;
    for (final other in _robots) {
      if (other == r) continue;
      if (other.state != RobotState.idle && other.state != RobotState.working) continue;
      final dx = other.x - r.x;
      final dy = other.y - r.y;
      final dist = dx * dx + dy * dy;
      if (dist < nearestDist) {
        nearestDist = dist;
        nearest = other;
      }
    }
    return nearest;
  }

  // ── Pet update ──────────────────────────────────────────────

  void _updatePet(OfficePet pet, double dt) {
    pet.animPhase += dt * 3;
    pet.stateTimer -= dt;

    if (pet.stateTimer <= 0) {
      _assignPetBehavior(pet);
    }

    // Move toward target
    final dx = pet.targetX - pet.x;
    final dy = pet.targetY - pet.y;
    final dist = sqrt(dx * dx + dy * dy);
    if (dist > 0.005 && pet.petState != PetState.sleeping) {
      final speed = pet.petState == PetState.chasingOther ? 0.12 : 0.06;
      pet.x += dx / dist * speed * dt;
      pet.y += dy / dist * speed * dt;
      pet.facing = dx >= 0 ? 1 : -1;
    }

    // Dog tail wag speed increases near robots
    if (pet.type == PetType.dog) {
      double minDist = double.infinity;
      for (final r in _robots) {
        final rdx = r.x - pet.x;
        final rdy = r.y - pet.y;
        final rd = rdx * rdx + rdy * rdy;
        if (rd < minDist) minDist = rd;
      }
      pet.tailWagSpeed = minDist < 0.02 ? 12.0 : 5.0;
    }

    // Cat occasionally knocks items off desks
    if (pet.type == PetType.cat && pet.petState == PetState.sittingOnDesk) {
      if (_rng.nextDouble() < 0.005) {
        _particles.emit(
          ParticleType.fallingItem,
          pet.x + 0.02,
          pet.y,
          const Color(0xFF90A4AE),
          count: 1,
        );
      }
    }

    // Paw prints from dog
    if (pet.type == PetType.dog && dist > 0.005 && pet.petState != PetState.sleeping) {
      pet.pawPrintTimer -= dt;
      if (pet.pawPrintTimer <= 0) {
        pet.pawPrintTimer = 0.5;
        _particles.emit(
          ParticleType.pawPrint,
          pet.x,
          pet.y + 0.02,
          const Color(0xFF5D4037).withValues(alpha: 0.3),
          count: 1,
        );
      }
    }
  }

  void _assignPetBehavior(OfficePet pet) {
    final roll = _rng.nextDouble();

    if (pet.type == PetType.dog) {
      if (roll < 0.25) {
        // Wander
        pet.petState = PetState.wandering;
        pet.targetX = (0.05 + _rng.nextDouble() * 0.90).clamp(0.05, 0.95);
        pet.targetY = (0.50 + _rng.nextDouble() * 0.40).clamp(0.50, 0.90);
        pet.stateTimer = 3.0 + _rng.nextDouble() * 4.0;
      } else if (roll < 0.45) {
        // Follow a robot
        pet.petState = PetState.followingRobot;
        final target = _robots[_rng.nextInt(_robots.length)];
        pet.targetX = target.x + 0.03;
        pet.targetY = target.y + 0.03;
        pet.stateTimer = 4.0 + _rng.nextDouble() * 3.0;
      } else if (roll < 0.60) {
        // Sleep in corner
        pet.petState = PetState.sleeping;
        pet.targetX = 0.05;
        pet.targetY = 0.88;
        pet.stateTimer = 5.0 + _rng.nextDouble() * 5.0;
      } else if (roll < 0.80) {
        // Chase cat
        pet.petState = PetState.chasingOther;
        pet.targetX = _cat.x;
        pet.targetY = _cat.y;
        pet.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
      } else {
        // Play (fetch ball)
        pet.petState = PetState.playing;
        pet.targetX = (pet.x + (_rng.nextDouble() - 0.5) * 0.2).clamp(0.05, 0.95);
        pet.targetY = (pet.y + (_rng.nextDouble() - 0.5) * 0.15).clamp(0.50, 0.90);
        pet.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
      }
    } else {
      // Cat behaviors
      if (roll < 0.30) {
        // Sleep on server rack (warm!)
        pet.petState = PetState.sleeping;
        final servers = officeFurniture.where((f) => f.type == 'server').toList();
        if (servers.isNotEmpty) {
          final server = servers.first;
          pet.targetX = server.x + server.w / 2;
          pet.targetY = server.y - 0.02;
        }
        pet.stateTimer = 6.0 + _rng.nextDouble() * 6.0;
      } else if (roll < 0.45) {
        // Wash face (stay in place)
        pet.petState = PetState.washingFace;
        pet.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
      } else if (roll < 0.60) {
        // Sit on desk watching monitor
        pet.petState = PetState.sittingOnDesk;
        final desks = officeFurniture.where((f) => f.type == 'desk').toList();
        if (desks.isNotEmpty) {
          final desk = desks[_rng.nextInt(desks.length)];
          pet.targetX = desk.x + desk.w / 2;
          pet.targetY = desk.y;
        }
        pet.stateTimer = 4.0 + _rng.nextDouble() * 3.0;
      } else if (roll < 0.75) {
        // Run from dog
        pet.petState = PetState.chasingOther; // fleeing = reversed chase
        pet.targetX = (_dog.x > 0.5 ? 0.1 : 0.9).clamp(0.05, 0.95);
        pet.targetY = (0.30 + _rng.nextDouble() * 0.3).clamp(0.15, 0.90);
        pet.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
      } else {
        // Wander
        pet.petState = PetState.wandering;
        pet.targetX = (0.05 + _rng.nextDouble() * 0.90).clamp(0.05, 0.95);
        pet.targetY = (0.30 + _rng.nextDouble() * 0.50).clamp(0.15, 0.90);
        pet.stateTimer = 3.0 + _rng.nextDouble() * 3.0;
      }
    }
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
        // Don't push apart robots that are interacting
        if (a.interactionPartner == b || b.interactionPartner == a) continue;
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
      child: Stack(
        children: [
          // Background: detailed office (walls, window, desks, lights)
          CustomPaint(
            painter: bg.OfficePainter(
              robots: const [],
              time: _elapsed,
              isRunning: true,
              brightness: Theme.of(context).brightness,
            ),
            child: const SizedBox.expand(),
          ),
          // Foreground: robots, pets, particles
          CustomPaint(
            painter: RobotOfficePainter(
              robots: _robots,
              furniture: officeFurniture,
              elapsed: _elapsed,
              dog: _dog,
              cat: _cat,
              particles: _particles,
            ),
            child: const SizedBox.expand(),
          ),
        ],
      ),
    );
  }
}

// ── Pet types and states ──────────────────────────────────────

enum PetType { dog, cat }

enum PetState {
  wandering,
  followingRobot,
  sleeping,
  chasingOther,
  playing,
  washingFace,
  sittingOnDesk,
}

class OfficePet {
  OfficePet({
    required this.type,
    required this.x,
    required this.y,
    required this.color,
  });

  final PetType type;
  final Color color;
  double x;
  double y;
  double targetX = 0.5;
  double targetY = 0.7;
  int facing = 1;
  PetState petState = PetState.wandering;
  double stateTimer = 2.0;
  double animPhase = 0;
  double tailWagSpeed = 5.0;
  double pawPrintTimer = 0;
}

// ── Particle System ──────────────────────────────────────────

enum ParticleType {
  spark,
  confetti,
  text,
  dataPacket,
  pawPrint,
  fallingItem,
}

class Particle {
  Particle({
    required this.x,
    required this.y,
    required this.vx,
    required this.vy,
    required this.life,
    required this.maxLife,
    required this.color,
    required this.size,
    required this.type,
    this.rotation = 0,
    this.rotationSpeed = 0,
    this.text,
    this.progress = 0,
    this.startX = 0,
    this.startY = 0,
    this.endX = 0,
    this.endY = 0,
  });

  double x, y, vx, vy;
  double life;
  final double maxLife;
  Color color;
  double size;
  double rotation;
  double rotationSpeed;
  final ParticleType type;
  String? text;
  // For data packets
  double progress;
  double startX, startY, endX, endY;
}

class ParticleSystem {
  final List<Particle> particles = [];

  static const int _maxParticles = 200;

  void emit(
    ParticleType type,
    double x,
    double y,
    Color color, {
    int count = 1,
    String? text,
  }) {
    final rng = Random();
    for (int i = 0; i < count && particles.length < _maxParticles; i++) {
      switch (type) {
        case ParticleType.spark:
          final angle = rng.nextDouble() * 2 * pi;
          final speed = 0.05 + rng.nextDouble() * 0.1;
          particles.add(Particle(
            x: x,
            y: y,
            vx: cos(angle) * speed,
            vy: sin(angle) * speed,
            life: 0.5 + rng.nextDouble() * 0.5,
            maxLife: 1.0,
            color: color,
            size: 2 + rng.nextDouble() * 2,
            type: type,
          ));
        case ParticleType.confetti:
          particles.add(Particle(
            x: x + (rng.nextDouble() - 0.5) * 0.06,
            y: y,
            vx: (rng.nextDouble() - 0.5) * 0.04,
            vy: 0.02 + rng.nextDouble() * 0.03,
            life: 2.0 + rng.nextDouble() * 1.0,
            maxLife: 3.0,
            color: Color.fromARGB(
              255,
              rng.nextInt(256),
              rng.nextInt(256),
              rng.nextInt(256),
            ),
            size: 2 + rng.nextDouble() * 3,
            type: type,
            rotation: rng.nextDouble() * 6.28,
            rotationSpeed: (rng.nextDouble() - 0.5) * 8,
          ));
        case ParticleType.text:
          particles.add(Particle(
            x: x + (rng.nextDouble() - 0.5) * 0.02,
            y: y,
            vx: (rng.nextDouble() - 0.5) * 0.005,
            vy: -0.02 - rng.nextDouble() * 0.01,
            life: 1.5 + rng.nextDouble() * 0.5,
            maxLife: 2.0,
            color: color,
            size: 8 + rng.nextDouble() * 6,
            type: type,
            text: text,
          ));
        case ParticleType.pawPrint:
          particles.add(Particle(
            x: x,
            y: y,
            vx: 0,
            vy: 0,
            life: 3.0,
            maxLife: 3.0,
            color: color,
            size: 3,
            type: type,
          ));
        case ParticleType.fallingItem:
          particles.add(Particle(
            x: x,
            y: y,
            vx: (rng.nextDouble() - 0.5) * 0.02,
            vy: 0.05 + rng.nextDouble() * 0.03,
            life: 1.5,
            maxLife: 1.5,
            color: color,
            size: 4,
            type: type,
            rotation: rng.nextDouble() * 6.28,
            rotationSpeed: (rng.nextDouble() - 0.5) * 6,
          ));
        case ParticleType.dataPacket:
          break; // handled by emitDataPacket
      }
    }
  }

  void emitDataPacket(
    double startX,
    double startY,
    double endX,
    double endY,
    Color color,
  ) {
    if (particles.length >= _maxParticles) return;
    particles.add(Particle(
      x: startX,
      y: startY,
      vx: 0,
      vy: 0,
      life: 1.2,
      maxLife: 1.2,
      color: color,
      size: 3,
      type: ParticleType.dataPacket,
      progress: 0,
      startX: startX,
      startY: startY,
      endX: endX,
      endY: endY,
    ));
  }

  void update(double dt) {
    for (int i = particles.length - 1; i >= 0; i--) {
      final p = particles[i];
      p.life -= dt;
      if (p.life <= 0) {
        particles.removeAt(i);
        continue;
      }

      switch (p.type) {
        case ParticleType.dataPacket:
          // Advance progress along bezier
          p.progress += dt / p.maxLife;
          p.progress = p.progress.clamp(0.0, 1.0);
          // Compute position on bezier curve
          final t = p.progress;
          final midX = (p.startX + p.endX) / 2;
          final midY = min(p.startY, p.endY) - 0.08;
          p.x = (1 - t) * (1 - t) * p.startX + 2 * (1 - t) * t * midX + t * t * p.endX;
          p.y = (1 - t) * (1 - t) * p.startY + 2 * (1 - t) * t * midY + t * t * p.endY;
          // Spark explosion on arrival
          if (p.progress >= 0.98 && p.life > 0.1) {
            emit(ParticleType.spark, p.endX, p.endY, p.color, count: 8);
            p.life = 0; // remove the packet
          }
        case ParticleType.confetti:
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.vy += 0.02 * dt; // gravity
          p.vx += (Random().nextDouble() - 0.5) * 0.005; // drift
          p.rotation += p.rotationSpeed * dt;
        case ParticleType.text:
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.size += dt * 2; // grow
        case ParticleType.spark:
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.vx *= 0.95;
          p.vy *= 0.95;
          p.size *= 0.98;
        case ParticleType.pawPrint:
          // Static, just fading
          break;
        case ParticleType.fallingItem:
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.vy += 0.05 * dt; // gravity
          p.rotation += p.rotationSpeed * dt;
      }
    }
  }
}
