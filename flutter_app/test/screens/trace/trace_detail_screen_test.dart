import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/screens/trace/trace_detail_screen.dart';
import 'package:cognithor_ui/services/trace_service.dart';

class _MockTraceService extends Mock implements TraceService {}

void main() {
  setUpAll(() {
    registerFallbackValue((CrewEvent _) {});
  });

  testWidgets('renders timeline events from provider', (tester) async {
    final svc = _MockTraceService();
    when(() => svc.fetchTrace(any())).thenAnswer(
      (_) async => [
        const CrewEvent(
          traceId: 'abc',
          eventType: 'crew_kickoff_started',
          timestamp: '2026-04-26T10:00:00Z',
          details: {'n_tasks': 2},
        ),
        const CrewEvent(
          traceId: 'abc',
          eventType: 'crew_task_started',
          timestamp: '2026-04-26T10:00:01Z',
          details: {'task_id': 't1', 'agent_role': 'researcher'},
        ),
      ],
    );
    when(() => svc.fetchTraceStats(any())).thenAnswer(
      (_) async => const CrewTraceStats(
        totalTokens: 0,
        agentBreakdown: {},
        guardrailPass: 0,
        guardrailFail: 0,
        guardrailRetries: 0,
      ),
    );
    when(() => svc.subscribeToTrace(any(), any())).thenAnswer((_) {});
    when(() => svc.unsubscribeFromTrace(any())).thenAnswer((_) {});

    final provider = TraceProvider(traceService: svc);

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider<TraceProvider>.value(
          value: provider,
          child: const TraceDetailScreen(traceId: 'abc'),
        ),
      ),
    );

    await tester.pumpAndSettle();
    expect(find.textContaining('crew_kickoff_started'), findsOneWidget);
    expect(find.textContaining('researcher'), findsOneWidget);
  });
}
