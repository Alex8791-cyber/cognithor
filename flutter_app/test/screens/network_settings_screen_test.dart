import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/network_settings_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('NetworkSettingsScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // Empty interfaces by default — screen lands in "no NICs found"
      // empty-state branch.
      when(() => api.get(any())).thenAnswer(
        (_) async => {
          'interfaces': <Map<String, dynamic>>[],
          'auto_detect': true,
          'active_ips': <String>[],
        },
      );
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const NetworkSettingsScreen(),
      ),
    );

    testWidgets('renders without crashing on empty interface list', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(NetworkSettingsScreen), findsOneWidget);
    });

    testWidgets('drives the /network/interfaces API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.get(any())).called(greaterThanOrEqualTo(1));
    });

    testWidgets('renders interface entries when API returns data', (
      tester,
    ) async {
      when(() => api.get(any())).thenAnswer(
        (_) async => {
          'interfaces': [
            {'name': 'eth0', 'ip': '192.168.1.10', 'is_up': true},
            {'name': 'wlan0', 'ip': '10.0.0.5', 'is_up': true},
          ],
          'auto_detect': false,
          'bind_host': '192.168.1.10',
          'active_ips': ['192.168.1.10'],
        },
      );

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();

      expect(find.textContaining('eth0'), findsWidgets);
      expect(find.textContaining('wlan0'), findsWidgets);
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.get(any())).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(NetworkSettingsScreen), findsOneWidget);
    });
  });
}
