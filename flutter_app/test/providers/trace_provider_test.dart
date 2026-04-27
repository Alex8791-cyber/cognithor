import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/services/trace_service.dart';

class _MockTraceService extends Mock implements TraceService {}

void main() {
  setUpAll(() {
    registerFallbackValue((CrewEvent _) {});
  });

  group('TraceProvider', () {
    late _MockTraceService svc;
    late TraceProvider provider;

    setUp(() {
      svc = _MockTraceService();
      provider = TraceProvider(traceService: svc);
    });

    test('initial state is empty', () {
      expect(provider.traces, isEmpty);
      expect(provider.pinnedTraceId, isNull);
      expect(provider.isLoading, false);
    });

    test('loadTraces sets traces from service', () async {
      when(
        () => svc.listTraces(
          status: any(named: 'status'),
          limit: any(named: 'limit'),
        ),
      ).thenAnswer(
        (_) async => [
          const CrewTraceMeta(
            traceId: 'a',
            status: TraceStatus.running,
            nTasks: 1,
            totalTokens: 0,
            agentCount: 1,
            nFailedGuardrails: 0,
          ),
        ],
      );

      await provider.loadTraces();
      expect(provider.traces.length, 1);
      expect(provider.traces.first.traceId, 'a');
      expect(provider.isLoading, false);
    });

    test('pinTrace fetches events + subscribes', () async {
      when(() => svc.fetchTrace(any())).thenAnswer(
        (_) async => [
          const CrewEvent(traceId: 'a', eventType: 'crew_kickoff_started'),
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

      await provider.pinTrace('a');
      expect(provider.pinnedTraceId, 'a');
      expect(provider.pinnedEvents.length, 1);
      verify(() => svc.subscribeToTrace('a', any())).called(1);
    });

    test('unpinTrace clears state + unsubscribes', () async {
      when(() => svc.fetchTrace(any())).thenAnswer((_) async => []);
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

      await provider.pinTrace('a');
      provider.unpinTrace();

      expect(provider.pinnedTraceId, isNull);
      expect(provider.pinnedEvents, isEmpty);
      verify(() => svc.unsubscribeFromTrace('a')).called(1);
    });
  });
}
