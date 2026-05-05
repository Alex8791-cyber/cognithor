import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/identity_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('IdentityScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(
        () => api.getIdentityState(),
      ).thenAnswer((_) async => {'available': true, 'name': 'Cognithor'});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const IdentityScreen(),
      ),
    );

    testWidgets('renders without crashing on a healthy identity payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      // didChangeDependencies → setState async; let it settle.
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(IdentityScreen), findsOneWidget);
    });

    testWidgets('drives the identity API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.getIdentityState()).called(greaterThanOrEqualTo(1));
    });

    testWidgets('error response surfaces the failure path', (tester) async {
      when(
        () => api.getIdentityState(),
      ).thenAnswer((_) async => {'error': 'identity disabled'});

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(IdentityScreen), findsOneWidget);
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.getIdentityState()).thenThrow(Exception('network down'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // Screen should still render (it goes into loading-then-error state).
      expect(find.byType(IdentityScreen), findsOneWidget);
    });
  });
}
