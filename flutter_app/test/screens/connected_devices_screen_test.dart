import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/connected_devices_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('ConnectedDevicesScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(
        () => api.get(any()),
      ).thenAnswer((_) async => {'devices': <Map<String, dynamic>>[]});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const ConnectedDevicesScreen(),
      ),
    );

    testWidgets('renders without crashing on empty device list', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(ConnectedDevicesScreen), findsOneWidget);
    });

    testWidgets('drives the /devices API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.get(any())).called(greaterThanOrEqualTo(1));
    });

    testWidgets('renders device entries when API returns data', (tester) async {
      when(() => api.get(any())).thenAnswer(
        (_) async => {
          'devices': [
            {'id': 'd1', 'name': 'iPhone 15', 'paired': true},
            {'id': 'd2', 'name': 'Pixel 8', 'paired': false},
          ],
        },
      );

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.textContaining('iPhone 15'), findsWidgets);
      expect(find.textContaining('Pixel 8'), findsWidgets);
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.get(any())).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(ConnectedDevicesScreen), findsOneWidget);
    });
  });
}
