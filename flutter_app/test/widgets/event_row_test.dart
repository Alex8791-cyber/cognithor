import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/screens/trace/widgets/event_row.dart';

void main() {
  group('EventRow', () {
    testWidgets('renders task_started event with agent role', (tester) async {
      const event = CrewEvent(
        traceId: 'a',
        eventType: 'crew_task_started',
        timestamp: '2026-04-26T10:00:01Z',
        details: {'task_id': 't1', 'agent_role': 'researcher'},
      );
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: EventRow(
              event: event,
              traceStartedAt: '2026-04-26T10:00:00Z',
            ),
          ),
        ),
      );
      expect(find.textContaining('researcher'), findsOneWidget);
      expect(find.textContaining('crew_task_started'), findsOneWidget);
    });

    testWidgets('renders task_completed with token + duration badges', (
      tester,
    ) async {
      const event = CrewEvent(
        traceId: 'a',
        eventType: 'crew_task_completed',
        timestamp: '2026-04-26T10:00:05Z',
        details: {'task_id': 't1', 'duration_ms': 4810.0, 'tokens': 1234},
      );
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: EventRow(
              event: event,
              traceStartedAt: '2026-04-26T10:00:00Z',
            ),
          ),
        ),
      );
      expect(find.textContaining('1234'), findsOneWidget);
    });

    testWidgets('renders guardrail_check with verdict color', (tester) async {
      const event = CrewEvent(
        traceId: 'a',
        eventType: 'crew_guardrail_check',
        details: {'task_id': 't1', 'verdict': 'fail', 'retry_count': 1},
      );
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(body: EventRow(event: event, traceStartedAt: null)),
        ),
      );
      expect(find.textContaining('fail'), findsOneWidget);
    });
  });
}
