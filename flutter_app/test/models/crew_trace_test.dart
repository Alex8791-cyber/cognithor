import 'package:flutter_test/flutter_test.dart';
import 'package:cognithor_ui/models/crew_trace.dart';

void main() {
  group('CrewTraceMeta', () {
    test('parses JSON with all fields', () {
      final meta = CrewTraceMeta.fromJson({
        'trace_id': 'abc123',
        'status': 'running',
        'started_at': '2026-04-26T10:00:00Z',
        'ended_at': null,
        'duration_ms': null,
        'n_tasks': 4,
        'total_tokens': 1234,
        'agent_count': 2,
        'n_failed_guardrails': 0,
      });
      expect(meta.traceId, 'abc123');
      expect(meta.status, TraceStatus.running);
      expect(meta.totalTokens, 1234);
      expect(meta.endedAt, isNull);
    });

    test('parses status string into enum', () {
      expect(
        CrewTraceMeta.fromJson({
          'trace_id': 'a',
          'status': 'completed',
          'n_tasks': 1,
          'total_tokens': 0,
          'agent_count': 1,
          'n_failed_guardrails': 0,
        }).status,
        TraceStatus.completed,
      );
      expect(
        CrewTraceMeta.fromJson({
          'trace_id': 'a',
          'status': 'failed',
          'n_tasks': 1,
          'total_tokens': 0,
          'agent_count': 1,
          'n_failed_guardrails': 0,
        }).status,
        TraceStatus.failed,
      );
      expect(
        CrewTraceMeta.fromJson({
          'trace_id': 'a',
          'status': 'unknown',
          'n_tasks': 1,
          'total_tokens': 0,
          'agent_count': 1,
          'n_failed_guardrails': 0,
        }).status,
        TraceStatus.running,
      );
    });
  });

  group('CrewEvent', () {
    test('parses JSON with details', () {
      final ev = CrewEvent.fromJson({
        'session_id': 'abc',
        'event_type': 'crew_task_started',
        'timestamp': '2026-04-26T10:00:01Z',
        'details': {'task_id': 't1', 'agent_role': 'researcher'},
      });
      expect(ev.traceId, 'abc');
      expect(ev.eventType, 'crew_task_started');
      expect(ev.taskId, 't1');
      expect(ev.agentRole, 'researcher');
    });

    test('extracts tokens from completed-event details', () {
      final ev = CrewEvent.fromJson({
        'session_id': 'abc',
        'event_type': 'crew_task_completed',
        'timestamp': '2026-04-26T10:00:05Z',
        'details': {'task_id': 't1', 'duration_ms': 4000.0, 'tokens': 1234},
      });
      expect(ev.tokens, 1234);
      expect(ev.durationMs, 4000.0);
    });
  });
}
