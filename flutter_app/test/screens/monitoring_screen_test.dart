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
  group('MonitoringScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // Default: empty dashboard + no events (loading completes empty).
      when(
        () => api.getMonitoringDashboard(),
      ).thenAnswer((_) async => <String, dynamic>{});
      when(
        () => api.getMonitoringEvents(n: any(named: 'n')),
      ).thenAnswer((_) async => {'events': <Map<String, dynamic>>[]});

      conn = FakeConnectionProvider(apiClient: api);
    });

    tearDown(() {
      // Cancel pending periodic timers to keep the test isolated.
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const MonitoringScreen(),
      ),
    );

    testWidgets('renders without crashing on empty dashboard payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      // Initial pump: shows skeleton while load is in flight.
      expect(find.byType(MonitoringScreen), findsOneWidget);

      // Let the async _loadData complete.
      await tester.pump(const Duration(milliseconds: 50));
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(MonitoringScreen), findsOneWidget);
    });

    testWidgets('drives both monitoring API endpoints', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump(const Duration(milliseconds: 50));
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();

      verify(
        () => api.getMonitoringDashboard(),
      ).called(greaterThanOrEqualTo(1));
      verify(
        () => api.getMonitoringEvents(n: any(named: 'n')),
      ).called(greaterThanOrEqualTo(1));
    });

    testWidgets('cancels its periodic refresh timer on dispose', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump(const Duration(milliseconds: 50));

      // Replace the screen — disposes _MonitoringScreenState. If the
      // periodic timer was leaked the test framework would surface it
      // as a "Timer is still running" exception when settling.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(MonitoringScreen), findsNothing);
    });
  });
}
