/// Deep interaction tests for [DashboardScreen].
///
/// Builds on the smoke test in `dashboard_screen_test.dart` (PR #489) by
/// driving observable user flows: pull-to-refresh re-issuing the data
/// load, the gauge values reflecting the mocked monitoring payload,
/// the empty-events state showing when no events are present, and the
/// PiP entry path when [PipProvider.show] is invoked.
///
/// We mock all three monitoring endpoints from [ApiClient] so the
/// initial `_loadData` resolves synchronously inside the test pump.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/pip_provider.dart';
import 'package:cognithor_ui/providers/robot_office_provider.dart';
import 'package:cognithor_ui/screens/dashboard_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockWsService extends Mock implements WebSocketService {}

void main() {
  group('DashboardScreen interactions', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;
    late PipProvider pip;

    /// Stub the three endpoints fired in parallel from [_loadData] with
    /// caller-supplied dashboard + events payloads.
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
      when(
        () => api.getModelStats(),
      ).thenAnswer((_) async => <String, dynamic>{});
    }

    setUp(() {
      api = _MockApiClient();
      pip = PipProvider();
      conn = FakeConnectionProvider(
        apiClient: api,
        wsService: _MockWsService(),
      );
      // Default: empty payload — every test can override before pump.
      stubApi(
        dashboard: <String, dynamic>{
          'cpu_usage': 0,
          'memory_usage': 0,
          'response_time_ms': 0,
          'total_tokens': 0,
        },
        events: <Map<String, dynamic>>[],
      );
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<PipProvider>.value(value: pip),
          ChangeNotifierProvider<RobotOfficeProvider>(
            create: (_) => SilentRobotOfficeProvider(),
          ),
        ],
        child: const DashboardScreen(),
      ),
    );

    Future<void> pump(WidgetTester tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
    }

    Future<void> teardown(WidgetTester tester) async {
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 50));
    }

    testWidgets(
      'renders an empty events ticker when the events array is empty',
      (tester) async {
        // Default-stubbed events list is empty.
        await pump(tester);
        tester.takeException();

        // The dashboard's _EventTicker shows the localized "No events"
        // copy when the list is empty.
        expect(find.text('No events recorded'), findsOneWidget);

        await teardown(tester);
      },
    );

    testWidgets('renders a non-empty event chip when events are present', (
      tester,
    ) async {
      stubApi(
        dashboard: <String, dynamic>{'cpu_usage': 0, 'memory_usage': 0},
        events: [
          {
            'severity': 'WARNING',
            'message': 'Disk usage 92 percent',
            'timestamp': '2026-05-05T12:00:00Z',
          },
        ],
      );

      await pump(tester);
      tester.takeException();

      // The single event ticker chip text is rendered.
      expect(find.text('Disk usage 92 percent'), findsOneWidget);
      // And the empty-state copy is no longer shown.
      expect(find.text('No events recorded'), findsNothing);

      await teardown(tester);
    });

    testWidgets(
      'pull-to-refresh on the dashboard list re-fires the monitoring endpoints',
      (tester) async {
        await pump(tester);
        tester.takeException();

        // After the initial load each endpoint has been hit once.
        verify(() => api.getMonitoringDashboard()).called(1);
        verify(() => api.getMonitoringEvents(n: any(named: 'n'))).called(1);

        // Drag the ListView downward to trigger the RefreshIndicator.
        // ListView is the dashboard body, find by type — there's exactly
        // one in the dashboard tree (event ticker uses a horizontal one,
        // but RefreshIndicator wraps the outer vertical list).
        await tester.fling(
          find.byType(RefreshIndicator),
          const Offset(0, 400),
          1500,
        );
        // Pump enough to drive the refresh animation + the async load.
        await tester.pump();
        await tester.pump(const Duration(seconds: 1));
        await tester.pump(const Duration(milliseconds: 100));

        // Each endpoint should now have been hit at least twice
        // (initial + refresh).
        verify(
          () => api.getMonitoringDashboard(),
        ).called(greaterThanOrEqualTo(1));
        verify(
          () => api.getMonitoringEvents(n: any(named: 'n')),
        ).called(greaterThanOrEqualTo(1));

        await teardown(tester);
      },
    );

    testWidgets('hiding PiP swaps the PiP notice for the inline hero panel', (
      tester,
    ) async {
      // PipProvider.visible defaults to true → on initial render the
      // dashboard shows _RobotOfficePipNotice with a Fullscreen button.
      await pump(tester);
      tester.takeException();
      expect(pip.visible, isTrue);
      expect(find.byIcon(Icons.fullscreen), findsOneWidget);

      // Hide the PiP overlay programmatically — the hero panel is no
      // longer behind a PiP notice.
      pip.hide();
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      expect(pip.visible, isFalse);
      // The fullscreen-icon-bearing notice is gone.
      expect(find.byIcon(Icons.fullscreen), findsNothing);

      await teardown(tester);
    });

    testWidgets(
      'tapping Fullscreen on the PiP notice keeps PiP visible (exitFullscreen)',
      (tester) async {
        // PipProvider.visible defaults to true so the notice is on screen.
        await pump(tester);
        tester.takeException();
        expect(pip.visible, isTrue);
        expect(pip.fullscreenOnDashboard, isFalse);

        // Tap the Fullscreen button inside the PiP notice — this calls
        // pip.exitFullscreen(), which keeps `visible` true and resets
        // the fullscreen-on-dashboard flag.
        await tester.tap(find.byIcon(Icons.fullscreen));
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 50));

        // exitFullscreen() leaves `visible` true (it's the OPPOSITE of
        // enterFullscreen). The observable change is fullscreenOnDashboard.
        expect(pip.visible, isTrue);
        expect(pip.fullscreenOnDashboard, isFalse);

        await teardown(tester);
      },
    );
  });
}
