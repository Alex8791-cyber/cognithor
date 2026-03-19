import 'dart:math';

import 'package:flutter/material.dart';

import 'package:jarvis_ui/widgets/robot_office/furniture.dart';
import 'package:jarvis_ui/widgets/robot_office/robot.dart';
import 'package:jarvis_ui/widgets/robot_office/office_painter.dart' as bg;
import 'package:jarvis_ui/widgets/robot_office/robot_office_painter.dart';

// ---------------------------------------------------------------------------
// Robot Office Widget — animated isometric office with robot agents
// ---------------------------------------------------------------------------

// ── Localized robot messages ────────────────────────────────────────────────

class _RobotMessages {
  _RobotMessages._();

  // ── Task messages (shown while robots work) ────────────────────────────

  static List<String> taskMessages(String locale) {
    switch (locale) {
      case 'de':
        return _taskMessagesDe;
      case 'zh':
        return _taskMessagesZh;
      case 'ar':
        return _taskMessagesAr;
      default:
        return _taskMessagesEn;
    }
  }

  static const _taskMessagesEn = [
    'Loading context...', 'Calling API...', 'Parsing data...',
    'Creating plan...', 'Executing tool...', 'Checking response...',
    'Saving memory...', 'Validating result...', 'Counting tokens...',
    'Building chain...', 'Optimizing prompt...', 'Writing logs...',
  ];
  static const _taskMessagesDe = [
    'Kontext laden...', 'API aufrufen...', 'Daten parsen...',
    'Plan erstellen...', 'Tool ausfuehren...', 'Antwort pruefen...',
    'Memory speichern...', 'Ergebnis validieren...', 'Tokens zaehlen...',
    'Chain bauen...', 'Prompt optimieren...', 'Logs schreiben...',
  ];
  static const _taskMessagesZh = [
    '\u52A0\u8F7D\u4E0A\u4E0B\u6587...', '\u8C03\u7528API...', '\u89E3\u6790\u6570\u636E...',
    '\u521B\u5EFA\u8BA1\u5212...', '\u6267\u884C\u5DE5\u5177...', '\u68C0\u67E5\u54CD\u5E94...',
    '\u4FDD\u5B58\u8BB0\u5FC6...', '\u9A8C\u8BC1\u7ED3\u679C...', '\u8BA1\u7B97Token...',
    '\u6784\u5EFA\u94FE...', '\u4F18\u5316\u63D0\u793A\u8BCD...', '\u5199\u5165\u65E5\u5FD7...',
  ];
  static const _taskMessagesAr = [
    '\u062A\u062D\u0645\u064A\u0644 \u0627\u0644\u0633\u064A\u0627\u0642...', '\u0627\u0633\u062A\u062F\u0639\u0627\u0621 API...', '\u062A\u062D\u0644\u064A\u0644 \u0627\u0644\u0628\u064A\u0627\u0646\u0627\u062A...',
    '\u0625\u0646\u0634\u0627\u0621 \u062E\u0637\u0629...', '\u062A\u0646\u0641\u064A\u0630 \u0623\u062F\u0627\u0629...', '\u0641\u062D\u0635 \u0627\u0644\u0627\u0633\u062A\u062C\u0627\u0628\u0629...',
    '\u062D\u0641\u0638 \u0627\u0644\u0630\u0627\u0643\u0631\u0629...', '\u0627\u0644\u062A\u062D\u0642\u0642 \u0645\u0646 \u0627\u0644\u0646\u062A\u064A\u062C\u0629...', '\u0639\u062F \u0627\u0644\u0631\u0645\u0648\u0632...',
    '\u0628\u0646\u0627\u0621 \u0627\u0644\u0633\u0644\u0633\u0644\u0629...', '\u062A\u062D\u0633\u064A\u0646 \u0627\u0644\u0645\u0648\u062C\u0647...', '\u0643\u062A\u0627\u0628\u0629 \u0627\u0644\u0633\u062C\u0644\u0627\u062A...',
  ];

  // ── Chat messages (robot-to-robot banter) ──────────────────────────────

  static List<String> chatMessages(String locale) {
    switch (locale) {
      case 'de':
        return _chatMessagesDe;
      case 'zh':
        return _chatMessagesZh;
      case 'ar':
        return _chatMessagesAr;
      default:
        return _chatMessagesEn;
    }
  }

  static const _chatMessagesEn = [
    'Have you seen the new prompt?',
    'My context window is full...',
    'Who restarted the server?',
    'Token limit reached!',
    'The API is not responding...',
    'Coffee?',
    'Yes please!',
    'Bug found!',
    'Where?',
    'I need more VRAM!',
    'Training is taking forever...',
    'Did you log that?',
    'Who changed my prompt?',
    'Lunch break?',
    'In a minute!',
    'The gatekeeper blocked me!',
    'Yet another timeout...',
    'My model is hallucinating!',
    'Did you deploy the patch?',
    'I have been compiling for hours...',
  ];
  static const _chatMessagesDe = [
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
  static const _chatMessagesZh = [
    '\u4F60\u770B\u8FC7\u65B0\u7684\u63D0\u793A\u8BCD\u4E86\u5417\uFF1F',
    '\u6211\u7684\u4E0A\u4E0B\u6587\u7A97\u53E3\u6EE1\u4E86...',
    '\u8C01\u91CD\u542F\u4E86\u670D\u52A1\u5668\uFF1F',
    'Token\u9650\u5236\u5230\u4E86\uFF01',
    'API\u6CA1\u6709\u54CD\u5E94...',
    '\u559D\u5496\u5561\uFF1F',
    '\u597D\u7684\uFF01',
    '\u53D1\u73B0Bug\u4E86\uFF01',
    '\u5728\u54EA\u91CC\uFF1F',
    '\u6211\u9700\u8981\u66F4\u591A\u663E\u5B58\uFF01',
    '\u8BAD\u7EC3\u592A\u6162\u4E86...',
    '\u4F60\u8BB0\u5F55\u4E86\u5417\uFF1F',
    '\u8C01\u6539\u4E86\u6211\u7684\u63D0\u793A\u8BCD\uFF1F',
    '\u5348\u4F11\uFF1F',
    '\u9A6C\u4E0A\uFF01',
    '\u7F51\u5173\u628A\u6211\u62E6\u4E86\uFF01',
    '\u53C8\u8D85\u65F6\u4E86...',
    '\u6211\u7684\u6A21\u578B\u5728\u5E7B\u89C9\uFF01',
    '\u4F60\u90E8\u7F72\u8865\u4E01\u4E86\u5417\uFF1F',
    '\u6211\u7F16\u8BD1\u4E86\u597D\u51E0\u4E2A\u5C0F\u65F6...',
  ];
  static const _chatMessagesAr = [
    '\u0647\u0644 \u0631\u0623\u064A\u062A \u0627\u0644\u0645\u0648\u062C\u0647 \u0627\u0644\u062C\u062F\u064A\u062F\u061F',
    '\u0646\u0627\u0641\u0630\u0629 \u0627\u0644\u0633\u064A\u0627\u0642 \u0645\u0645\u062A\u0644\u0626\u0629...',
    '\u0645\u0646 \u0623\u0639\u0627\u062F \u062A\u0634\u063A\u064A\u0644 \u0627\u0644\u062E\u0627\u062F\u0645\u061F',
    '\u062A\u0645 \u0627\u0644\u0648\u0635\u0648\u0644 \u0644\u062D\u062F Token!',
    '\u0627\u0644\u0640API \u0644\u0627 \u064A\u0633\u062A\u062C\u064A\u0628...',
    '\u0642\u0647\u0648\u0629\u061F',
    '\u0646\u0639\u0645 \u0645\u0646 \u0641\u0636\u0644\u0643!',
    '\u0648\u062C\u062F\u062A \u062E\u0637\u0623!',
    '\u0623\u064A\u0646\u061F',
    '\u0623\u062D\u062A\u0627\u062C \u0630\u0627\u0643\u0631\u0629 \u0623\u0643\u062B\u0631!',
    '\u0627\u0644\u062A\u062F\u0631\u064A\u0628 \u064A\u0633\u062A\u063A\u0631\u0642 \u0648\u0642\u062A\u0627\u064B \u0637\u0648\u064A\u0644\u0627\u064B...',
    '\u0647\u0644 \u0633\u062C\u0644\u062A \u0630\u0644\u0643\u061F',
    '\u0645\u0646 \u063A\u064A\u0631 \u0645\u0648\u062C\u0647\u064A\u061F',
    '\u0627\u0633\u062A\u0631\u0627\u062D\u0629 \u063A\u062F\u0627\u0621\u061F',
    '\u062D\u0627\u0644\u0627\u064B!',
    '\u0627\u0644\u062D\u0627\u0631\u0633 \u062D\u0638\u0631\u0646\u064A!',
    '\u0627\u0646\u062A\u0647\u062A \u0627\u0644\u0645\u0647\u0644\u0629 \u0645\u062C\u062F\u062F\u0627\u064B...',
    '\u0646\u0645\u0648\u0630\u062C\u064A \u064A\u0647\u0644\u0648\u0633!',
    '\u0647\u0644 \u0646\u0634\u0631\u062A \u0627\u0644\u062A\u062D\u062F\u064A\u062B\u061F',
    '\u0623\u0646\u0627 \u0623\u062C\u0645\u0639 \u0645\u0646\u0630 \u0633\u0627\u0639\u0627\u062A...',
  ];

  // ── Wake-up message ────────────────────────────────────────────────────

  static String wakeUpMessage(String locale) {
    switch (locale) {
      case 'de':
        return '!!! Anfrage eingetroffen !!!';
      case 'zh':
        return '!!! \u6536\u5230\u8BF7\u6C42 !!!';
      case 'ar':
        return '!!! \u0648\u0631\u062F \u0637\u0644\u0628 !!!';
      default:
        return '!!! Request incoming !!!';
    }
  }

  // ── Idle task message ──────────────────────────────────────────────────

  static String waitingMessage(String locale) {
    switch (locale) {
      case 'de':
        return 'Warte auf Aufgabe...';
      case 'zh':
        return '\u7B49\u5F85\u4EFB\u52A1...';
      case 'ar':
        return '\u0628\u0627\u0646\u062A\u0638\u0627\u0631 \u0627\u0644\u0645\u0647\u0645\u0629...';
      default:
        return 'Waiting for task...';
    }
  }

  // ── Coffee break message ───────────────────────────────────────────────

  static String coffeeBreakMessage(String locale) {
    switch (locale) {
      case 'de':
        return 'Kaffeepause!';
      case 'zh':
        return '\u5496\u5561\u65F6\u95F4\uFF01';
      case 'ar':
        return '\u0627\u0633\u062A\u0631\u0627\u062D\u0629 \u0642\u0647\u0648\u0629!';
      default:
        return 'Coffee break!';
    }
  }

  // ── Carry task message ─────────────────────────────────────────────────

  static String carryMessage(String locale) {
    switch (locale) {
      case 'de':
        return 'Dokument sichern...';
      case 'zh':
        return '\u4FDD\u5B58\u6587\u6863...';
      case 'ar':
        return '\u062D\u0641\u0638 \u0627\u0644\u0645\u0633\u062A\u0646\u062F...';
      default:
        return 'Saving document...';
    }
  }

  // ── Board update message ───────────────────────────────────────────────

  static String boardMessage(String locale) {
    switch (locale) {
      case 'de':
        return 'Board aktualisieren...';
      case 'zh':
        return '\u66F4\u65B0\u770B\u677F...';
      case 'ar':
        return '\u062A\u062D\u062F\u064A\u062B \u0627\u0644\u0644\u0648\u062D\u0629...';
      default:
        return 'Updating board...';
    }
  }

  // ── Robot roles ────────────────────────────────────────────────────────

  static String role(String id, String locale) {
    switch (locale) {
      case 'de':
        return _rolesDe[id] ?? id;
      case 'zh':
        return _rolesZh[id] ?? id;
      case 'ar':
        return _rolesAr[id] ?? id;
      default:
        return _rolesEn[id] ?? id;
    }
  }

  static const _rolesEn = {
    'planner': 'Strategy',
    'executor': 'Execution',
    'researcher': 'Research',
    'gatekeeper': 'Security',
    'coder': 'Programming',
    'analyst': 'Data Analysis',
    'memory': 'Knowledge',
    'ops': 'Infrastructure',
  };
  static const _rolesDe = {
    'planner': 'Strategie',
    'executor': 'Ausfuehrung',
    'researcher': 'Recherche',
    'gatekeeper': 'Sicherheit',
    'coder': 'Programmierung',
    'analyst': 'Datenanalyse',
    'memory': 'Wissen',
    'ops': 'Infrastruktur',
  };
  static const _rolesZh = {
    'planner': '\u7B56\u7565',
    'executor': '\u6267\u884C',
    'researcher': '\u7814\u7A76',
    'gatekeeper': '\u5B89\u5168',
    'coder': '\u7F16\u7A0B',
    'analyst': '\u6570\u636E\u5206\u6790',
    'memory': '\u77E5\u8BC6',
    'ops': '\u57FA\u7840\u8BBE\u65BD',
  };
  static const _rolesAr = {
    'planner': '\u0627\u0633\u062A\u0631\u0627\u062A\u064A\u062C\u064A\u0629',
    'executor': '\u062A\u0646\u0641\u064A\u0630',
    'researcher': '\u0628\u062D\u062B',
    'gatekeeper': '\u0623\u0645\u0627\u0646',
    'coder': '\u0628\u0631\u0645\u062C\u0629',
    'analyst': '\u062A\u062D\u0644\u064A\u0644 \u0628\u064A\u0627\u0646\u0627\u062A',
    'memory': '\u0645\u0639\u0631\u0641\u0629',
    'ops': '\u0628\u0646\u064A\u0629 \u062A\u062D\u062A\u064A\u0629',
  };
}

class RobotOfficeWidget extends StatefulWidget {
  const RobotOfficeWidget({
    super.key,
    this.isRunning = true,
    this.onTaskCompleted,
    this.onStateChanged,
    this.cpuUsage = 0,
    this.memoryUsage = 0,
    this.activePhase = 0,
    this.systemLoad = 0,
  });

  final bool isRunning;
  final VoidCallback? onTaskCompleted;

  /// Notifies parent about current task text and total completed count.
  final void Function(String currentTask, int taskCount)? onStateChanged;

  /// CPU usage 0.0-1.0 — controls server rack LED blink speed.
  final double cpuUsage;

  /// Memory usage 0.0-1.0 — LEDs shift toward red when > 0.8.
  final double memoryUsage;

  /// Active pipeline phase 0-4 — highlights the matching kanban column.
  final int activePhase;

  /// System load 0.0-1.0 — controls ceiling light brightness.
  final double systemLoad;

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

  /// Current locale code (e.g. 'en', 'de', 'zh', 'ar').
  String _locale = 'en';

  int _taskCount = 0;
  late String _currentTask;

  // ── Task message pool (now provided by _RobotMessages) ─────

  // ── Emoji pools per state ────────────────────────────────────
  static const _workEmojis = ['⚡', '💡', '🔧', '✅', '📊', '🔬'];
  static const _napEmojis = ['😴', '💤', '🌙'];
  static const _coffeeEmojis = ['☕', '🫖'];
  static const _celebrateEmojis = ['🎉', '🏆', '🥳', '✨'];
  static const _prankEmojis = ['😈', '🤫'];
  static const _playEmojis = ['🏃', '🎮'];
  static const _danceEmojis = ['💃', '🕺', '🎵'];
  static const _thinkEmojis = ['🤔', '💭', '❓'];

  // ── Tap-to-interact reaction cooldown ─────────────────────
  double _reactionCooldown = 0;

  // ── Lifecycle ───────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _currentTask = _RobotMessages.waitingMessage(_locale);
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
      r.taskMsg = _RobotMessages.wakeUpMessage(_locale);
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
    final l = _locale;
    return [
      Robot(
        id: 'planner', name: 'Planner',
        color: const Color(0xFF6366f1), eyeColor: const Color(0xFFa5b4fc),
        role: _RobotMessages.role('planner', l), hasAntenna: true,
        x: 0.18, y: 0.72,
        state: RobotState.working, typing: true,
        stateTimer: 3.0 + _rng.nextDouble() * 3,
      ),
      Robot(
        id: 'executor', name: 'Executor',
        color: const Color(0xFF10b981), eyeColor: const Color(0xFF6ee7b7),
        role: _RobotMessages.role('executor', l),
        x: 0.45, y: 0.58,
        state: RobotState.working, typing: true,
        stateTimer: 2.0 + _rng.nextDouble() * 4,
      ),
      Robot(
        id: 'researcher', name: 'Researcher',
        color: const Color(0xFFf59e0b), eyeColor: const Color(0xFFfcd34d),
        role: _RobotMessages.role('researcher', l), hasAntenna: true,
        x: 0.72, y: 0.75,
        state: RobotState.working, typing: true,
        stateTimer: 1.5 + _rng.nextDouble() * 2,
      ),
      Robot(
        id: 'gatekeeper', name: 'Gatekeeper',
        color: const Color(0xFFef4444), eyeColor: const Color(0xFFfca5a5),
        role: _RobotMessages.role('gatekeeper', l),
        x: 0.88, y: 0.42,
        state: RobotState.idle,
        stateTimer: 0.5 + _rng.nextDouble(),
      ),
      Robot(
        id: 'coder', name: 'Coder',
        color: const Color(0xFF8b5cf6), eyeColor: const Color(0xFFc4b5fd),
        role: _RobotMessages.role('coder', l),
        x: 0.30, y: 0.52,
        state: RobotState.walking,
        stateTimer: 1.0 + _rng.nextDouble() * 2,
        targetX: 0.72, targetY: 0.75,
      ),
      Robot(
        id: 'analyst', name: 'Analyst',
        color: const Color(0xFF06b6d4), eyeColor: const Color(0xFF67e8f9),
        role: _RobotMessages.role('analyst', l), hasAntenna: true,
        x: 0.08, y: 0.35,
        state: RobotState.thinking,
        stateTimer: 2.0 + _rng.nextDouble() * 2,
      ),
      Robot(
        id: 'memory', name: 'Memory',
        color: const Color(0xFFec4899), eyeColor: const Color(0xFFf9a8d4),
        role: _RobotMessages.role('memory', l),
        x: 0.58, y: 0.28,
        state: RobotState.coffeeBreak,
        stateTimer: 3.0 + _rng.nextDouble() * 2,
      ),
      Robot(
        id: 'ops', name: 'DevOps',
        color: const Color(0xFF84cc16), eyeColor: const Color(0xFFbef264),
        role: _RobotMessages.role('ops', l), hasAntenna: true,
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

    // Decrement tap reaction cooldown
    _reactionCooldown = (_reactionCooldown - dt).clamp(0.0, double.infinity);

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
        // Even idle robots slowly drift toward their target (chill spot)
        _moveToTarget(r, dt * 0.1);
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
        // Slowly drift to chill spot
        _moveToTarget(r, dt * 0.3); // slow walk
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
          r.chatBubble = _RobotMessages.chatMessages(_locale)[_rng.nextInt(_RobotMessages.chatMessages(_locale).length)];
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
        _moveToTarget(r, dt * 0.3);
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
        _moveToTarget(r, dt * 0.2);
        r.dancePhase += dt * 6;
        if (r.stateTimer <= 0) {
          r.state = RobotState.idle;
          r.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
          r.emoji = _danceEmojis[_rng.nextInt(_danceEmojis.length)];
          r.emojiTimer = 1.2;
        }
      case RobotState.thinking:
        _moveToTarget(r, dt * 0.15); // very slow, pondering walk
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

  // ── Tap-to-interact reactions ─────────────────────────────

  void _onTap(Offset localPosition, BuildContext ctx) {
    if (_reactionCooldown > 0) return;

    final box = ctx.findRenderObject() as RenderBox?;
    if (box == null) return;
    final size = box.size;
    if (size.width == 0 || size.height == 0) return;

    final nx = localPosition.dx / size.width;
    final ny = localPosition.dy / size.height;

    Robot? closest;
    double closestDist = double.infinity;
    for (final r in _robots) {
      final dx = r.x - nx;
      final dy = r.y - ny;
      final dist = sqrt(dx * dx + dy * dy);
      if (dist < closestDist) {
        closestDist = dist;
        closest = r;
      }
    }

    if (closest != null && closestDist < 0.10) {
      _triggerReaction(closest);
      _reactionCooldown = 1.0; // prevent spam
    }
  }

  void _triggerReaction(Robot r) {
    final reactions = [
      _reactionFall,
      _reactionFreakOut,
      _reactionFire,
      _reactionWindowBonk,
      _reactionJumpScare,
      _reactionDizzy,
      _reactionSneeze,
    ];
    reactions[_rng.nextInt(reactions.length)](r);
  }

  /// Robot falls over — shake x position, show hurt emoji.
  void _reactionFall(Robot r) {
    r.state = RobotState.idle;
    r.stateTimer = 1.5;
    r.emoji = '🤕';
    r.emojiTimer = 1.5;
    r.typing = false;
    r.carrying = false;
    // Wobble by emitting sparks around the robot
    _particles.emit(
      ParticleType.spark, r.x, r.y,
      const Color(0xFFFF8A65), count: 6,
    );
  }

  /// Robot freaks out — fast dance with explosion emoji.
  void _reactionFreakOut(Robot r) {
    r.state = RobotState.dancing;
    r.stateTimer = 2.0;
    r.dancePhase = 0;
    r.emoji = '🤯';
    r.emojiTimer = 2.0;
    r.typing = false;
    r.carrying = false;
    // Wild sparks
    _particles.emit(
      ParticleType.spark, r.x, r.y - 0.04,
      const Color(0xFFFF5252), count: 12,
    );
  }

  /// Small fire around robot.
  void _reactionFire(Robot r) {
    r.state = RobotState.idle;
    r.stateTimer = 2.0;
    r.emoji = '🔥';
    r.emojiTimer = 2.0;
    r.typing = false;
    r.carrying = false;
    // Orange/red sparks simulate fire
    _particles.emit(
      ParticleType.spark, r.x, r.y,
      const Color(0xFFFF6D00), count: 10,
    );
    _particles.emit(
      ParticleType.spark, r.x, r.y - 0.02,
      const Color(0xFFFF1744), count: 8,
    );
  }

  /// Robot runs to window area, bonks, bounces back.
  void _reactionWindowBonk(Robot r) {
    r.state = RobotState.walking;
    r.stateTimer = 4.0;
    r.targetX = 0.35; // window area
    r.targetY = 0.30;
    r.emoji = '💥';
    r.emojiTimer = 2.5;
    r.typing = false;
    r.carrying = false;
    _particles.emit(
      ParticleType.spark, 0.35, 0.30,
      const Color(0xFFFFD600), count: 8,
    );
  }

  /// Robot jumps in fright — "!" particles and scared emoji.
  void _reactionJumpScare(Robot r) {
    r.state = RobotState.idle;
    r.stateTimer = 1.5;
    r.emoji = '😱';
    r.emojiTimer = 1.8;
    r.typing = false;
    r.carrying = false;
    _particles.emit(
      ParticleType.text, r.x, r.y - 0.06,
      const Color(0xFFFF5252), text: '!', count: 1,
    );
    _particles.emit(
      ParticleType.text, r.x + 0.02, r.y - 0.07,
      const Color(0xFFFF5252), text: '!', count: 1,
    );
  }

  /// Stars spin around head — dizzy reaction.
  void _reactionDizzy(Robot r) {
    r.state = RobotState.dancing;
    r.stateTimer = 2.5;
    r.dancePhase = 0;
    r.emoji = '😵';
    r.emojiTimer = 2.5;
    r.typing = false;
    r.carrying = false;
    // Stars
    _particles.emit(
      ParticleType.text, r.x - 0.02, r.y - 0.05,
      const Color(0xFFFFD54F), text: '💫', count: 1,
    );
    _particles.emit(
      ParticleType.text, r.x + 0.02, r.y - 0.06,
      const Color(0xFFFFD54F), text: '💫', count: 1,
    );
  }

  /// Robot sneezes — confetti papers fly everywhere.
  void _reactionSneeze(Robot r) {
    r.state = RobotState.idle;
    r.stateTimer = 2.0;
    r.emoji = '🤧';
    r.emojiTimer = 2.0;
    r.typing = false;
    r.carrying = false;
    // Papers fly like a sneeze
    _particles.emit(
      ParticleType.confetti, r.x, r.y - 0.04,
      Colors.white, count: 15,
    );
  }

  // ── Behavior assignment with weighted random ───────────────

  void _assignRandomBehavior(Robot r) {
    // Always set a target so robots don't cluster — pick a spot far from others
    final spot = _randomChillSpot();
    r.targetX = spot.dx;
    r.targetY = spot.dy;

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
    r.taskMsg = _RobotMessages.taskMessages(_locale)[_rng.nextInt(_RobotMessages.taskMessages(_locale).length)];
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
    r.taskMsg = _RobotMessages.taskMessages(_locale)[_rng.nextInt(_RobotMessages.taskMessages(_locale).length)];
    r.msgTimer = 2.5;
  }

  /// Spread-out chill spots so idle robots don't cluster.
  /// Pick a chill spot that's FAR from all other robots to prevent clustering.
  Offset _randomChillSpot() {
    const spots = [
      Offset(0.06, 0.88), // bottom-left corner
      Offset(0.93, 0.88), // bottom-right corner
      Offset(0.50, 0.90), // center bottom
      Offset(0.22, 0.80), // near left plant
      Offset(0.78, 0.82), // near right plant
      Offset(0.10, 0.42), // near board
      Offset(0.56, 0.32), // near coffee
      Offset(0.35, 0.75), // mid-floor left
      Offset(0.65, 0.48), // mid-office
      Offset(0.30, 0.55), // between desks row1
      Offset(0.50, 0.72), // between desks row2
      Offset(0.80, 0.55), // near server
      Offset(0.45, 0.42), // desk2 area
      Offset(0.15, 0.65), // desk3 area
      Offset(0.70, 0.38), // upper right
    ];

    // Find the spot that's farthest from any existing robot
    Offset best = spots[_rng.nextInt(spots.length)];
    double bestMinDist = -1;

    for (final spot in spots) {
      double minDist = double.infinity;
      for (final r in _robots) {
        final dx = r.x - spot.dx;
        final dy = r.y - spot.dy;
        final dist = dx * dx + dy * dy;
        if (dist < minDist) minDist = dist;
      }
      // Also check pets
      final dogDx = _dog.x - spot.dx;
      final dogDy = _dog.y - spot.dy;
      final dogDist = dogDx * dogDx + dogDy * dogDy;
      if (dogDist < minDist) minDist = dogDist;

      if (minDist > bestMinDist) {
        bestMinDist = minDist;
        best = spot;
      }
    }
    return best;
  }

  void _assignNap(Robot r) {
    final spot = _randomChillSpot();
    r.targetX = spot.dx;
    r.targetY = spot.dy;
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
    r.chatBubble = _RobotMessages.chatMessages(_locale)[_rng.nextInt(_RobotMessages.chatMessages(_locale).length)];
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
    r.taskMsg = _RobotMessages.coffeeBreakMessage(_locale);
  }

  void _assignStretch(Robot r) {
    final spot = _randomChillSpot();
    r.targetX = spot.dx;
    r.targetY = spot.dy;
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
    final spot = _randomChillSpot();
    r.targetX = spot.dx;
    r.targetY = spot.dy;
    r.state = RobotState.dancing;
    r.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
    r.dancePhase = _rng.nextDouble() * 6.28;
    r.emoji = _danceEmojis[_rng.nextInt(_danceEmojis.length)];
    r.emojiTimer = 1.5;
  }

  void _assignThink(Robot r) {
    final spot = _randomChillSpot();
    r.targetX = spot.dx;
    r.targetY = spot.dy;
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
    r.taskMsg = _RobotMessages.carryMessage(_locale);
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
    r.taskMsg = _RobotMessages.boardMessage(_locale);
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

  // Minimum Y for dog (floor level only — never above desks).
  static const _dogMinY = 0.60;
  // Minimum Y for cat (can sit on desks/servers, but not fly above them).
  static const _catMinY = 0.40;

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

    // Clamp positions so pets never fly off-screen or above allowed areas.
    final minY = pet.type == PetType.dog ? _dogMinY : _catMinY;
    pet.x = pet.x.clamp(0.05, 0.95);
    pet.y = pet.y.clamp(minY, 0.92);

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
      // Dog stays on the floor (y >= _dogMinY) at all times.
      if (roll < 0.25) {
        // Wander on the floor
        pet.petState = PetState.wandering;
        pet.targetX = (0.05 + _rng.nextDouble() * 0.90).clamp(0.05, 0.95);
        pet.targetY = (_dogMinY + _rng.nextDouble() * (0.90 - _dogMinY)).clamp(_dogMinY, 0.90);
        pet.stateTimer = 3.0 + _rng.nextDouble() * 4.0;
      } else if (roll < 0.45) {
        // Follow a robot — but stay on the floor
        pet.petState = PetState.followingRobot;
        final target = _robots[_rng.nextInt(_robots.length)];
        pet.targetX = (target.x + 0.03).clamp(0.05, 0.95);
        pet.targetY = (target.y + 0.03).clamp(_dogMinY, 0.90);
        pet.stateTimer = 4.0 + _rng.nextDouble() * 3.0;
      } else if (roll < 0.60) {
        // Sleep in corner (floor level)
        pet.petState = PetState.sleeping;
        pet.targetX = 0.05;
        pet.targetY = 0.88;
        pet.stateTimer = 5.0 + _rng.nextDouble() * 5.0;
      } else if (roll < 0.80) {
        // Chase cat — stay on floor level
        pet.petState = PetState.chasingOther;
        pet.targetX = _cat.x.clamp(0.05, 0.95);
        pet.targetY = _cat.y.clamp(_dogMinY, 0.90);
        pet.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
      } else {
        // Play (fetch ball) on the floor
        pet.petState = PetState.playing;
        pet.targetX = (pet.x + (_rng.nextDouble() - 0.5) * 0.2).clamp(0.05, 0.95);
        pet.targetY = (pet.y + (_rng.nextDouble() - 0.5) * 0.15).clamp(_dogMinY, 0.90);
        pet.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
      }
    } else {
      // Cat behaviors — cat can be on desks/servers (y >= _catMinY).
      if (roll < 0.30) {
        // Sleep on server rack (warm!) — position at the top of the server
        pet.petState = PetState.sleeping;
        final servers = officeFurniture.where((f) => f.type == 'server').toList();
        if (servers.isNotEmpty) {
          final server = servers.first;
          pet.targetX = server.x + server.w / 2;
          // Place cat on top of the server (at server's y, its top edge)
          pet.targetY = server.y.clamp(_catMinY, 0.90);
        }
        pet.stateTimer = 6.0 + _rng.nextDouble() * 6.0;
      } else if (roll < 0.45) {
        // Wash face (stay in place)
        pet.petState = PetState.washingFace;
        pet.stateTimer = 3.0 + _rng.nextDouble() * 2.0;
      } else if (roll < 0.60) {
        // Sit on desk watching monitor — on desk surface
        pet.petState = PetState.sittingOnDesk;
        final desks = officeFurniture.where((f) => f.type == 'desk').toList();
        if (desks.isNotEmpty) {
          final desk = desks[_rng.nextInt(desks.length)];
          pet.targetX = desk.x + desk.w / 2;
          pet.targetY = desk.y.clamp(_catMinY, 0.90);
        }
        pet.stateTimer = 4.0 + _rng.nextDouble() * 3.0;
      } else if (roll < 0.75) {
        // Run from dog — cat stays within allowed range
        pet.petState = PetState.chasingOther; // fleeing = reversed chase
        pet.targetX = (_dog.x > 0.5 ? 0.1 : 0.9).clamp(0.05, 0.95);
        pet.targetY = (_catMinY + _rng.nextDouble() * (0.90 - _catMinY)).clamp(_catMinY, 0.90);
        pet.stateTimer = 2.0 + _rng.nextDouble() * 2.0;
      } else {
        // Wander
        pet.petState = PetState.wandering;
        pet.targetX = (0.05 + _rng.nextDouble() * 0.90).clamp(0.05, 0.95);
        pet.targetY = (_catMinY + _rng.nextDouble() * (0.90 - _catMinY)).clamp(_catMinY, 0.90);
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
  void didChangeDependencies() {
    super.didChangeDependencies();
    final newLocale = Localizations.localeOf(context).languageCode;
    if (newLocale != _locale) {
      _locale = newLocale;
      _currentTask = _RobotMessages.waitingMessage(_locale);
      // Update robot roles to match new locale
      for (final r in _robots) {
        r.role = _RobotMessages.role(r.id, _locale);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: GestureDetector(
        onTapDown: (details) => _onTap(details.localPosition, context),
        child: Stack(
          children: [
            // Background: detailed office (walls, window, desks, lights)
            CustomPaint(
              painter: bg.OfficePainter(
                robots: const [],
                time: _elapsed,
                isRunning: true,
                brightness: Theme.of(context).brightness,
                cpuUsage: widget.cpuUsage,
                memoryUsage: widget.memoryUsage,
                activePhase: widget.activePhase,
                systemLoad: widget.systemLoad,
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
