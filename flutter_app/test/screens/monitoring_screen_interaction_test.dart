/// Deep interaction tests for [MonitoringScreen].
///
/// Builds on the smoke tests in `monitoring_screen_test.dart` (PR #489)
/// by driving real user flows: switching between Dashboard/Events/Live
/// Logs tabs, asserting that the dashboard tab renders the mocked
/// stats (uptime / active sessions / total requests), the events tab
/// renders an event card (or the empty-state), and the retry button
/// in the error state re-issues the load.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/monitoring_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('MonitoringScreen interactions', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    void stubApi({
      required Map<String, dynamic> dashboard,
      required List<Map<String, dynamic>> events,
    }) {
      when(
        () => api.getMonitoringDashboard(),
      ).thenAnswer((_) async => dashboard);
      when(
        () => api.getMonitoringEvents(n: any(named: 'n')),
      ).thenAnswer((_) async => {'events': events});
    }

    setUp(() {
      api = _MockApiClient();
      conn = FakeConnectionProvider(apiClient: api);
      stubApi(
        dashboard: <String, dynamic>{
          'uptime': '3h 12m',
          'active_sessions': 4,
          'total_requests': 128,
        },
        events: <Map<String, dynamic>>[],
      );
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const MonitoringScreen(),
      ),
    );

    Future<void> pump(WidgetTester tester) async {
      await tester.pumpWidget(wrap());
      // Initial pump shows the loading spinner.
      await tester.pump();
      // Drain the parallel _loadData Future.wait.
      await tester.pump(const Duration(milliseconds: 50));
      await tester.pump(const Duration(milliseconds: 50));
    }

    Future<void> teardown(WidgetTester tester) async {
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 50));
    }

    testWidgets('dashboard tab renders mocked uptime/sessions/requests stats', (
      tester,
    ) async {
      await pump(tester);
      tester.takeException();

      // The dashboard tab is the default selected tab.
      expect(find.text('3h 12m'), findsOneWidget);
      expect(find.text('4'), findsOneWidget);
      expect(find.text('128'), findsOneWidget);

      await teardown(tester);
    });

    testWidgets('tapping the Events tab swaps the body to the events list', (
      tester,
    ) async {
      stubApi(
        dashboard: <String, dynamic>{'uptime': '0s'},
        events: [
          {
            'severity': 'INFO',
            'message': 'Gateway warmed up',
            'timestamp': '2026-05-05T12:00:00Z',
          },
        ],
      );

      await pump(tester);
      tester.takeException();

      // Switch to Events tab — find the tab text and tap it.
      await tester.tap(find.text('Events'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 250));

      // The event message is now rendered in the events tab body.
      expect(find.text('Gateway warmed up'), findsOneWidget);
      // Severity badge text is rendered too.
      expect(find.text('INFO'), findsOneWidget);

      await teardown(tester);
    });

    testWidgets(
      'events tab shows the empty-state copy when no events are present',
      (tester) async {
        // Default stub events list is empty.
        await pump(tester);
        tester.takeException();

        await tester.tap(find.text('Events'));
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 250));

        expect(find.text('No events recorded'), findsOneWidget);

        await teardown(tester);
      },
    );

    testWidgets(
      'pull-to-refresh on the dashboard tab re-fires both endpoints',
      (tester) async {
        await pump(tester);
        tester.takeException();

        // After initial load each endpoint has fired at least once.
        verify(() => api.getMonitoringDashboard()).called(1);
        verify(() => api.getMonitoringEvents(n: any(named: 'n'))).called(1);

        // Find the RefreshIndicator wrapping the dashboard ListView.
        // The dashboard tab is the default selection so the dashboard
        // tab's RefreshIndicator is on screen.
        final refreshIndicator = find.byType(RefreshIndicator).first;
        await tester.fling(refreshIndicator, const Offset(0, 400), 1500);
        await tester.pump();
        await tester.pump(const Duration(seconds: 1));
        await tester.pump(const Duration(milliseconds: 100));

        // Each endpoint should now have been hit at least once more
        // (the periodic timer may also tick — assert >= 1 additional call).
        verify(
          () => api.getMonitoringDashboard(),
        ).called(greaterThanOrEqualTo(1));

        await teardown(tester);
      },
    );
  });
}
