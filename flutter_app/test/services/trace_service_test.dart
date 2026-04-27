import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/trace_service.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockWsService extends Mock implements WebSocketService {}

void main() {
  group('TraceService', () {
    late _MockApiClient api;
    late _MockWsService ws;
    late TraceService svc;

    setUp(() {
      api = _MockApiClient();
      ws = _MockWsService();
      // send() returns bool — stub it so mocktail doesn't throw on non-nullable return.
      when(() => ws.send(any())).thenReturn(true);
      svc = TraceService(apiClient: api, wsService: ws);
    });

    test(
      'listTraces parses /api/crew/traces JSON into CrewTraceMeta list',
      () async {
        when(() => api.get(any())).thenAnswer(
          (_) async => {
            'traces': [
              {
                'trace_id': 't1',
                'status': 'running',
                'n_tasks': 2,
                'total_tokens': 100,
                'agent_count': 1,
                'n_failed_guardrails': 0,
              },
              {
                'trace_id': 't2',
                'status': 'completed',
                'n_tasks': 4,
                'total_tokens': 500,
                'agent_count': 2,
                'n_failed_guardrails': 1,
              },
            ],
            'meta': {'skipped_lines': 0},
          },
        );

        final result = await svc.listTraces();
        expect(result.length, 2);
        expect(result[0].traceId, 't1');
        expect(result[1].status, TraceStatus.completed);
      },
    );

    test('fetchTrace returns full event list', () async {
      when(() => api.get(any())).thenAnswer(
        (_) async => {
          'trace_id': 'abc',
          'events': [
            {
              'session_id': 'abc',
              'event_type': 'crew_kickoff_started',
              'details': {'n_tasks': 2},
            },
            {
              'session_id': 'abc',
              'event_type': 'crew_task_started',
              'details': {'task_id': 't1', 'agent_role': 'researcher'},
            },
          ],
          'meta': {'skipped_lines': 0},
        },
      );

      final events = await svc.fetchTrace('abc');
      expect(events.length, 2);
      expect(events[0].eventType, 'crew_kickoff_started');
    });

    test('subscribeToTrace sends crew_subscribe and registers callbacks', () {
      svc.subscribeToTrace('xyz', (event) {});
      verify(
        () => ws.send(
          any(
            that: predicate<Map<String, dynamic>>(
              (m) => m['type'] == 'crew_subscribe' && m['trace_id'] == 'xyz',
            ),
          ),
        ),
      ).called(1);
    });

    test('unsubscribeFromTrace sends crew_unsubscribe', () {
      svc.unsubscribeFromTrace('xyz');
      verify(
        () => ws.send(
          any(
            that: predicate<Map<String, dynamic>>(
              (m) => m['type'] == 'crew_unsubscribe' && m['trace_id'] == 'xyz',
            ),
          ),
        ),
      ).called(1);
    });
  });
}
