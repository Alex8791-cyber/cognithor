import 'dart:async';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

/// API + WebSocket wrapper for the Trace-UI surface.
///
/// REST: list/get/stats endpoints under `/api/crew/`.
/// WS: subscribe to lifecycle stream + per-trace topic streams.
class TraceService {
  final ApiClient _api;
  final WebSocketService _ws;

  /// Map of trace_id → callback invoked on each `crew_event` frame for that trace.
  final Map<String, void Function(CrewEvent)> _topicCallbacks = {};

  /// Single optional callback invoked on each `crew_lifecycle` frame.
  void Function(CrewEvent)? _lifecycleCallback;

  TraceService({
    required ApiClient apiClient,
    required WebSocketService wsService,
  }) : _api = apiClient,
       _ws = wsService {
    _ws.on(WsType.crewLifecycle, _handleLifecycleFrame);
    _ws.on(WsType.crewEvent, _handleEventFrame);
  }

  /// GET /api/crew/traces — returns list of meta cards.
  Future<List<CrewTraceMeta>> listTraces({
    String? status,
    int limit = 50,
  }) async {
    final qs = <String>[];
    if (status != null) qs.add('status=$status');
    qs.add('limit=$limit');
    final path = '/api/crew/traces?${qs.join('&')}';
    final resp = await _api.get(path);
    final list = (resp['traces'] as List?) ?? const [];
    return list
        .map((j) => CrewTraceMeta.fromJson((j as Map).cast<String, dynamic>()))
        .toList();
  }

  /// GET /api/crew/trace/{id} — returns full event list.
  Future<List<CrewEvent>> fetchTrace(String traceId) async {
    final resp = await _api.get('/api/crew/trace/$traceId');
    final list = (resp['events'] as List?) ?? const [];
    return list
        .map((j) => CrewEvent.fromJson((j as Map).cast<String, dynamic>()))
        .toList();
  }

  /// GET /api/crew/trace/{id}/stats — returns derived aggregates.
  Future<CrewTraceStats> fetchTraceStats(String traceId) async {
    final resp = await _api.get('/api/crew/trace/$traceId/stats');
    return CrewTraceStats.fromJson(resp);
  }

  /// Subscribe to lifecycle stream (Dashboard view).
  void subscribeToLifecycle(void Function(CrewEvent) callback) {
    _lifecycleCallback = callback;
    _ws.send({'type': WsType.crewLifecycleSubscribe});
  }

  /// Unsubscribe from lifecycle stream (Dashboard close).
  void unsubscribeFromLifecycle() {
    _lifecycleCallback = null;
    // Server doesn't have a "lifecycle_unsubscribe" — disconnect handles it.
  }

  /// Subscribe to per-trace event stream (Detail view).
  void subscribeToTrace(String traceId, void Function(CrewEvent) callback) {
    _topicCallbacks[traceId] = callback;
    _ws.send({'type': WsType.crewSubscribe, 'trace_id': traceId});
  }

  /// Unsubscribe from per-trace event stream.
  void unsubscribeFromTrace(String traceId) {
    _topicCallbacks.remove(traceId);
    _ws.send({'type': WsType.crewUnsubscribe, 'trace_id': traceId});
  }

  void _handleLifecycleFrame(Map<String, dynamic> message) {
    final payload = (message['payload'] as Map?)?.cast<String, dynamic>();
    if (payload == null) return;
    final cb = _lifecycleCallback;
    if (cb != null) {
      cb(CrewEvent.fromJson(payload));
    }
  }

  void _handleEventFrame(Map<String, dynamic> message) {
    final payload = (message['payload'] as Map?)?.cast<String, dynamic>();
    if (payload == null) return;
    final tid = payload['session_id'] ?? payload['trace_id'];
    final cb = _topicCallbacks[tid as String?];
    if (cb != null) {
      cb(CrewEvent.fromJson(payload));
    }
  }
}
