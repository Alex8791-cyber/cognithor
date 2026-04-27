import 'package:flutter/foundation.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/services/trace_service.dart';

class TraceProvider extends ChangeNotifier {
  final TraceService _svc;

  TraceProvider({required TraceService traceService}) : _svc = traceService;

  List<CrewTraceMeta> _traces = [];
  String? _pinnedTraceId;
  List<CrewEvent> _pinnedEvents = [];
  CrewTraceStats? _pinnedStats;
  bool _isLoading = false;
  String? _errorMessage;

  List<CrewTraceMeta> get traces => List.unmodifiable(_traces);
  String? get pinnedTraceId => _pinnedTraceId;
  List<CrewEvent> get pinnedEvents => List.unmodifiable(_pinnedEvents);
  CrewTraceStats? get pinnedStats => _pinnedStats;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;

  Future<void> loadTraces({String? status, int limit = 50}) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();
    try {
      _traces = await _svc.listTraces(status: status, limit: limit);
    } catch (e) {
      _errorMessage = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void subscribeToLifecycle() {
    _svc.subscribeToLifecycle(_onLifecycleEvent);
  }

  void unsubscribeFromLifecycle() {
    _svc.unsubscribeFromLifecycle();
  }

  Future<void> pinTrace(String traceId) async {
    if (_pinnedTraceId == traceId) return;
    if (_pinnedTraceId != null) {
      _svc.unsubscribeFromTrace(_pinnedTraceId!);
    }
    _pinnedTraceId = traceId;
    _pinnedEvents = [];
    _pinnedStats = null;
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();
    try {
      final events = await _svc.fetchTrace(traceId);
      _pinnedEvents = events;
      _pinnedStats = await _svc.fetchTraceStats(traceId);
      _svc.subscribeToTrace(traceId, _onPinnedEvent);
    } catch (e) {
      _errorMessage = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void unpinTrace() {
    if (_pinnedTraceId != null) {
      _svc.unsubscribeFromTrace(_pinnedTraceId!);
    }
    _pinnedTraceId = null;
    _pinnedEvents = [];
    _pinnedStats = null;
    notifyListeners();
  }

  void _onLifecycleEvent(CrewEvent event) {
    final idx = _traces.indexWhere((t) => t.traceId == event.traceId);
    if (event.eventType == 'crew_kickoff_started') {
      if (idx == -1) {
        _traces.insert(
          0,
          CrewTraceMeta(
            traceId: event.traceId,
            status: TraceStatus.running,
            startedAt: event.timestamp,
            nTasks: (event.details['n_tasks'] as num?)?.toInt() ?? 0,
            totalTokens: 0,
            agentCount: 0,
            nFailedGuardrails: 0,
          ),
        );
      }
    } else if (event.eventType == 'crew_kickoff_completed' ||
        event.eventType == 'crew_kickoff_failed') {
      if (idx != -1) {
        final old = _traces[idx];
        _traces[idx] = CrewTraceMeta(
          traceId: old.traceId,
          status: event.eventType == 'crew_kickoff_failed'
              ? TraceStatus.failed
              : TraceStatus.completed,
          startedAt: old.startedAt,
          endedAt: event.timestamp,
          durationMs: old.durationMs,
          nTasks: old.nTasks,
          totalTokens: old.totalTokens,
          agentCount: old.agentCount,
          nFailedGuardrails: old.nFailedGuardrails,
        );
      }
    }
    notifyListeners();
  }

  void _onPinnedEvent(CrewEvent event) {
    if (event.traceId != _pinnedTraceId) return;
    _pinnedEvents = [..._pinnedEvents, event];
    notifyListeners();
  }
}
