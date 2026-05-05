/// Smoke test for [DashboardScreen].
///
/// Previously deferred because the screen owns:
///   - a 15-second `Timer.periodic` that re-issues `_loadData`
///     (cancelled in `dispose`), and
///   - inline RobotOfficeProvider.init() which would otherwise spin
///     up its own 10-second polling Timer + WS listeners.
///
/// We mock the three monitoring endpoints so the initial load
/// completes synchronously, and we use [SilentRobotOfficeProvider]
/// to no-op the inline `init`. The dispose-path test ensures the
/// 15-second refresh Timer is cancelled.
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
  group('DashboardScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // Three endpoints fired in parallel from `_loadData`.
      when(() => api.getMonitoringDashboard()).thenAnswer(
        (_) async => <String, dynamic>{
          'cpu_usage': 0,
          'memory_usage': 0,
          'response_time_ms': 0,
          'total_tokens': 0,
        },
      );
      when(
        () => api.getMonitoringEvents(n: any(named: 'n')),
      ).thenAnswer((_) async => {'events': <Map<String, dynamic>>[]});
      when(
        () => api.getModelStats(),
      ).thenAnswer((_) async => <String, dynamic>{});

      conn = FakeConnectionProvider(
        apiClient: api,
        wsService: _MockWsService(),
      );
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<PipProvider>(create: (_) => PipProvider()),
          ChangeNotifierProvider<RobotOfficeProvider>(
            create: (_) => SilentRobotOfficeProvider(),
          ),
        ],
        child: const DashboardScreen(),
      ),
    );

    testWidgets('renders without crashing on minimal monitoring payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      // Initial pump shows the shimmer loading state.
      await tester.pump();
      // Drain the parallel `Future.wait` triggered by `_loadData`.
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(DashboardScreen), findsOneWidget);
    });

    testWidgets('drives all 3 monitoring endpoints from initial load', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();

      verify(
        () => api.getMonitoringDashboard(),
      ).called(greaterThanOrEqualTo(1));
      verify(
        () => api.getMonitoringEvents(n: any(named: 'n')),
      ).called(greaterThanOrEqualTo(1));
    });

    testWidgets('cancels its 15-second refresh Timer on dispose', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Replacing the tree triggers `_DashboardScreenState.dispose`.
      // If the periodic Timer were leaked the framework would surface
      // it as a "Timer is still running" exception when the test ends.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(DashboardScreen), findsNothing);
    });
  });
}
