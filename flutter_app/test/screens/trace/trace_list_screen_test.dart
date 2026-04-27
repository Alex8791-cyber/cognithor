import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/screens/trace/trace_list_screen.dart';
import 'package:cognithor_ui/services/trace_service.dart';

class _MockTraceService extends Mock implements TraceService {}

void main() {
  setUpAll(() {
    registerFallbackValue((CrewEvent _) {});
  });

  testWidgets('renders trace cards from provider state', (tester) async {
    final svc = _MockTraceService();
    when(
      () => svc.listTraces(
        status: any(named: 'status'),
        limit: any(named: 'limit'),
      ),
    ).thenAnswer(
      (_) async => [
        CrewTraceMeta(
          traceId: 'trace-1',
          status: TraceStatus.running,
          nTasks: 2,
          totalTokens: 100,
          agentCount: 1,
          nFailedGuardrails: 0,
        ),
        CrewTraceMeta(
          traceId: 'trace-2',
          status: TraceStatus.completed,
          nTasks: 4,
          totalTokens: 500,
          agentCount: 2,
          nFailedGuardrails: 1,
        ),
      ],
    );
    when(() => svc.subscribeToLifecycle(any())).thenAnswer((_) {});
    when(() => svc.unsubscribeFromLifecycle()).thenAnswer((_) {});

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
    expect(find.textContaining('trace-1'), findsOneWidget);
    expect(find.textContaining('trace-2'), findsOneWidget);
  });
}
