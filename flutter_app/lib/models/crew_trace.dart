// Models for the Trace-UI screen — mirrors the JSON returned by
// `/api/crew/traces`, `/api/crew/trace/{id}`, and the WebSocket
// `crew_lifecycle` / `crew_event` frames.

enum TraceStatus { running, completed, failed }

class CrewTraceMeta {
  final String traceId;
  final TraceStatus status;
  final String? startedAt;
  final String? endedAt;
  final double? durationMs;
  final int nTasks;
  final int totalTokens;
  final int agentCount;
  final int nFailedGuardrails;

  const CrewTraceMeta({
    required this.traceId,
    required this.status,
    this.startedAt,
    this.endedAt,
    this.durationMs,
    required this.nTasks,
    required this.totalTokens,
    required this.agentCount,
    required this.nFailedGuardrails,
  });

  factory CrewTraceMeta.fromJson(Map<String, dynamic> j) {
    final raw = (j['status'] as String?)?.toLowerCase() ?? 'running';
    final TraceStatus status = switch (raw) {
      'completed' => TraceStatus.completed,
      'failed' => TraceStatus.failed,
      _ => TraceStatus.running,
    };
    return CrewTraceMeta(
      traceId: j['trace_id'] as String,
      status: status,
      startedAt: j['started_at'] as String?,
      endedAt: j['ended_at'] as String?,
      durationMs: (j['duration_ms'] as num?)?.toDouble(),
      nTasks: (j['n_tasks'] as num?)?.toInt() ?? 0,
      totalTokens: (j['total_tokens'] as num?)?.toInt() ?? 0,
      agentCount: (j['agent_count'] as num?)?.toInt() ?? 0,
      nFailedGuardrails: (j['n_failed_guardrails'] as num?)?.toInt() ?? 0,
    );
  }
}

class CrewEvent {
  final String traceId;
  final String eventType;
  final String? timestamp;
  final Map<String, dynamic> details;

  const CrewEvent({
    required this.traceId,
    required this.eventType,
    this.timestamp,
    this.details = const {},
  });

  factory CrewEvent.fromJson(Map<String, dynamic> j) {
    final details = (j['details'] as Map?)?.cast<String, dynamic>() ?? const {};
    return CrewEvent(
      traceId: (j['session_id'] ?? j['trace_id']) as String,
      eventType: (j['event_type'] ?? j['event']) as String,
      timestamp: j['timestamp'] as String?,
      details: details,
    );
  }

  String? get taskId => details['task_id'] as String?;
  String? get agentRole => details['agent_role'] as String?;
  int? get tokens => (details['tokens'] as num?)?.toInt();
  double? get durationMs => (details['duration_ms'] as num?)?.toDouble();
  String? get verdict => details['verdict'] as String?;
  int? get retryCount => (details['retry_count'] as num?)?.toInt();
}

class CrewTraceStats {
  final int totalTokens;
  final double? totalDurationMs;
  final Map<String, int> agentBreakdown;
  final int guardrailPass;
  final int guardrailFail;
  final int guardrailRetries;

  const CrewTraceStats({
    required this.totalTokens,
    this.totalDurationMs,
    required this.agentBreakdown,
    required this.guardrailPass,
    required this.guardrailFail,
    required this.guardrailRetries,
  });

  factory CrewTraceStats.fromJson(Map<String, dynamic> j) {
    final breakdown =
        (j['agent_breakdown'] as Map?)?.map(
          (k, v) => MapEntry(k as String, (v as num).toInt()),
        ) ??
        const <String, int>{};
    final summary =
        (j['guardrail_summary'] as Map?)?.cast<String, dynamic>() ?? const {};
    return CrewTraceStats(
      totalTokens: (j['total_tokens'] as num?)?.toInt() ?? 0,
      totalDurationMs: (j['total_duration_ms'] as num?)?.toDouble(),
      agentBreakdown: breakdown,
      guardrailPass: (summary['pass'] as num?)?.toInt() ?? 0,
      guardrailFail: (summary['fail'] as num?)?.toInt() ?? 0,
      guardrailRetries: (summary['retries'] as num?)?.toInt() ?? 0,
    );
  }
}
