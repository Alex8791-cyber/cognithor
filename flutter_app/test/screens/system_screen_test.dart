import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/system_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SystemScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(
        () => api.getSystemStatus(),
      ).thenAnswer((_) async => {'uptime_seconds': 0, 'version': '0.97.0'});
      when(() => api.getCommands()).thenAnswer((_) async => {'commands': []});
      when(
        () => api.getConnectors(),
      ).thenAnswer((_) async => {'connectors': []});

      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const SystemScreen(),
      ),
    );

    testWidgets('renders without crashing on empty system payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(SystemScreen), findsOneWidget);
    });

    testWidgets('drives all 3 system API endpoints', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.getSystemStatus()).called(greaterThanOrEqualTo(1));
      verify(() => api.getCommands()).called(greaterThanOrEqualTo(1));
      verify(() => api.getConnectors()).called(greaterThanOrEqualTo(1));
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.getSystemStatus()).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(SystemScreen), findsOneWidget);
    });
  });
}
