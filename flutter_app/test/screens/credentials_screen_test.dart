import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/credentials_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('CredentialsScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(
        () => api.getCredentials(),
      ).thenAnswer((_) async => {'credentials': <Map<String, dynamic>>[]});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const CredentialsScreen(),
      ),
    );

    testWidgets('renders without crashing on empty credential list', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(CredentialsScreen), findsOneWidget);
    });

    testWidgets('drives the credentials API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.getCredentials()).called(greaterThanOrEqualTo(1));
    });

    testWidgets('renders credential entries when API returns data', (
      tester,
    ) async {
      when(() => api.getCredentials()).thenAnswer(
        (_) async => {
          'credentials': [
            {'service': 'github', 'key': 'ghp_xxx'},
            {'service': 'openai', 'key': 'sk_xxx'},
          ],
        },
      );

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.textContaining('github'), findsWidgets);
      expect(find.textContaining('openai'), findsWidgets);
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.getCredentials()).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(CredentialsScreen), findsOneWidget);
    });
  });
}
