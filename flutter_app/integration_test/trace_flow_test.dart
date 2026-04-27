import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/screens/trace/trace_list_screen.dart';
import 'package:cognithor_ui/services/trace_service.dart';

class _MockTraceService extends Mock implements TraceService {}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() {
    registerFallbackValue((CrewEvent _) {});
  });

  testWidgets('list → tap card → detail view shows events', (tester) async {
    final svc = _MockTraceService();
    when(
      () => svc.listTraces(
        status: any(named: 'status'),
        limit: any(named: 'limit'),
      ),
    ).thenAnswer(
      (_) async => [
        const CrewTraceMeta(
          traceId: 'abc-1',
          status: TraceStatus.running,
          nTasks: 1,
          totalTokens: 50,
          agentCount: 1,
          nFailedGuardrails: 0,
        ),
      ],
    );
    when(() => svc.subscribeToLifecycle(any())).thenAnswer((_) {});
    when(() => svc.unsubscribeFromLifecycle()).thenAnswer((_) {});
    when(() => svc.fetchTrace('abc-1')).thenAnswer(
      (_) async => [
        const CrewEvent(
          traceId: 'abc-1',
          eventType: 'crew_kickoff_started',
          details: {'n_tasks': 1},
        ),
      ],
    );
    when(() => svc.fetchTraceStats('abc-1')).thenAnswer(
      (_) async => const CrewTraceStats(
        totalTokens: 50,
        agentBreakdown: {'researcher': 50},
        guardrailPass: 1,
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
          child: const TraceListScreen(),
        ),
      ),
    );
    await tester.pumpAndSettle();

    // Verify list rendered.
    expect(find.textContaining('abc-1'), findsOneWidget);

    // Tap card → detail view.
    await tester.tap(find.textContaining('abc-1'));
    await tester.pumpAndSettle();

    // Detail view shows the kickoff event.
    expect(find.textContaining('crew_kickoff_started'), findsOneWidget);
  });
}
