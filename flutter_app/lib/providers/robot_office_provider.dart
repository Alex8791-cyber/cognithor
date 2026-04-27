import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

/// PGE pipeline phase.
enum PgePhase { idle, planning, gating, executing, streaming }

/// Info about a single configured agent.
class AgentInfo {
  AgentInfo({
    required this.name,
    this.displayName,
    this.isWorking = false,
    this.currentTask = '',
  });
  final String name;
  final String? displayName;
  bool isWorking;
  String currentTask;
}

/// System resource metrics.
class SystemMetrics {
  double cpu;
  double memory;
  double load;
  SystemMetrics({this.cpu = 0, this.memory = 0, this.load = 0});
}

/// Aggregates real-time data for the Robot Office visualization.
class RobotOfficeProvider extends ChangeNotifier {
  ApiClient? _api;
  WebSocketService? _ws;
  Timer? _pollTimer;
  bool _wsListenersAttached = false;

  // ── Public state ──────────────────────────────────────────
  PgePhase pgePhase = PgePhase.idle;
  String plannerTask = '';
  String executorTask = '';
  String gatekeeperTask = '';
  List<AgentInfo> agents = [];
  SystemMetrics metrics = SystemMetrics();
  Map<String, int> kanbanCounts = {};
  Map<String, List<String>> kanbanTasks = {};
  DateTime _lastEventTime = DateTime.now();

  // ── Lifecycle ─────────────────────────────────────────────

  void init(ApiClient api, WebSocketService? ws) {
    _api = api;
    _ws = ws;
    _startPolling();
    _attachWsListeners();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _detachWsListeners();
    super.dispose();
  }

  // ── REST Polling (every 10s) ──────────────────────────────

  void _startPolling() {
    _poll(); // immediate first poll
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 10), (_) => _poll());
  }

  Future<void> _poll() async {
    if (_api == null) return;
    await Future.wait([_pollMetrics(), _pollAgents(), _pollKanban()]);
    notifyListeners();
  }

  Future<void> _pollMetrics() async {
    try {
      final res = await _api!.get('monitoring/dashboard');
      final sys = res['system'] as Map<String, dynamic>? ?? {};
      metrics.cpu = ((sys['cpu_percent'] as num?) ?? 0).toDouble() / 100.0;
      metrics.memory =
          ((sys['memory_percent'] as num?) ?? 0).toDouble() / 100.0;
      metrics.load = ((metrics.cpu + metrics.memory) / 2).clamp(0.0, 1.0);
    } catch (_) {}
  }

  Future<void> _pollAgents() async {
    try {
      final res = await _api!.get('agents');
      final list = res['agents'] as List<dynamic>? ?? [];
      agents = list
          .map((a) {
            final m = a as Map<String, dynamic>;
            return AgentInfo(
              name: m['name']?.toString() ?? '',
              displayName: m['display_name']?.toString(),
            );
          })
          .where((a) => a.name.isNotEmpty)
          .toList();
    } catch (_) {}
  }

  Future<void> _pollKanban() async {
    try {
      final res = await _api!.get('tasks');
      final tasks = res['tasks'] as List<dynamic>? ?? [];
      final counts = <String, int>{};
      final titles = <String, List<String>>{};
      for (final t in tasks) {
        final m = t as Map<String, dynamic>;
        final status = m['status']?.toString() ?? 'backlog';
        final title = m['title']?.toString() ?? '';
        counts[status] = (counts[status] ?? 0) + 1;
        titles.putIfAbsent(status, () => []);
        if (titles[status]!.length < 5) titles[status]!.add(title);
        // Track agent assignment
        final agent = m['assigned_agent']?.toString() ?? '';
        if (agent.isNotEmpty && status == 'in_progress') {
          for (final a in agents) {
            if (a.name == agent) {
              a.isWorking = true;
              a.currentTask = title;
            }
          }
        }
      }
      kanbanCounts = counts;
      kanbanTasks = titles;
    } catch (_) {}
  }

  // ── WebSocket Events (real-time) ──────────────────────────

  void _attachWsListeners() {
    if (_ws == null || _wsListenersAttached) return;
    _ws!.on(WsType.statusUpdate, _onStatusUpdate);
    _ws!.on(WsType.pipelineEvent, _onPipelineEvent);
    _ws!.on(WsType.toolStart, _onToolStart);
    _ws!.on(WsType.toolResult, _onToolResult);
    _ws!.on(WsType.streamToken, _onStreamToken);
    _ws!.on(WsType.streamEnd, _onStreamEnd);
    _wsListenersAttached = true;
  }

  void _detachWsListeners() {
    if (_ws == null || !_wsListenersAttached) return;
    _ws!.off(WsType.statusUpdate, _onStatusUpdate);
    _ws!.off(WsType.pipelineEvent, _onPipelineEvent);
    _ws!.off(WsType.toolStart, _onToolStart);
    _ws!.off(WsType.toolResult, _onToolResult);
    _ws!.off(WsType.streamToken, _onStreamToken);
    _ws!.off(WsType.streamEnd, _onStreamEnd);
    _wsListenersAttached = false;
  }

  void _onStatusUpdate(Map<String, dynamic> msg) {
    _lastEventTime = DateTime.now();
    final status = msg['status']?.toString() ?? '';
    if (status == 'thinking') {
      pgePhase = PgePhase.planning;
      plannerTask = 'Planning: ${_truncate(msg['text']?.toString() ?? '', 30)}';
    }
    notifyListeners();
  }

  void _onPipelineEvent(Map<String, dynamic> msg) {
    _lastEventTime = DateTime.now();
    final phase = msg['phase']?.toString() ?? '';
    final action = msg['action']?.toString() ?? '';
    if (phase == 'plan' && action == 'start') {
      pgePhase = PgePhase.planning;
    } else if (phase == 'gate') {
      pgePhase = PgePhase.gating;
      gatekeeperTask =
          'Reviewing: ${_truncate(msg['tool']?.toString() ?? '', 25)}';
    } else if (phase == 'execute') {
      pgePhase = PgePhase.executing;
    } else if (phase == 'complete' || action == 'end') {
      pgePhase = PgePhase.idle;
      plannerTask = '';
      executorTask = '';
      gatekeeperTask = '';
    }
    notifyListeners();
  }

  void _onToolStart(Map<String, dynamic> msg) {
    _lastEventTime = DateTime.now();
    pgePhase = PgePhase.executing;
    executorTask = 'Running: ${msg['tool']?.toString() ?? 'tool'}';
    notifyListeners();
  }

  void _onToolResult(Map<String, dynamic> msg) {
    _lastEventTime = DateTime.now();
    executorTask = '';
    notifyListeners();
  }

  void _onStreamToken(Map<String, dynamic> msg) {
    _lastEventTime = DateTime.now();
    if (pgePhase != PgePhase.streaming) {
      pgePhase = PgePhase.streaming;
      executorTask = 'Streaming response...';
      notifyListeners();
    }
  }

  void _onStreamEnd(Map<String, dynamic> msg) {
    _lastEventTime = DateTime.now();
    pgePhase = PgePhase.idle;
    plannerTask = '';
    executorTask = '';
    gatekeeperTask = '';
    notifyListeners();
  }

  /// True if [init] has not yet been called (no API client wired up).
  bool get isUninitialized => _api == null;

  /// True if no WS events received for 30+ seconds.
  bool get isSystemIdle =>
      DateTime.now().difference(_lastEventTime).inSeconds > 30;

  /// Active phase as int 0-4 for the painter.
  int get activePhaseInt {
    switch (pgePhase) {
      case PgePhase.planning:
        return 0;
      case PgePhase.gating:
        return 1;
      case PgePhase.executing:
        return 2;
      case PgePhase.streaming:
        return 3;
      case PgePhase.idle:
        return 4;
    }
  }

  static String _truncate(String s, int max) =>
      s.length <= max ? s : '${s.substring(0, max)}...';
}
