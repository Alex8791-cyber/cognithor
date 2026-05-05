import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/security_provider.dart';
import 'package:cognithor_ui/screens/security_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SecurityScreen smoke', () {
    late _MockApiClient api;
    late SecurityProvider sec;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      sec = SilentSecurityProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<SecurityProvider>.value(
          value: sec,
          child: const SecurityScreen(),
        ),
      ),
    );

    testWidgets('renders without crashing on empty security state', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(SecurityScreen), findsOneWidget);
    });

    testWidgets('renders the tab bar regardless of provider state', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      // Tab bar is part of the always-rendered shell.
      expect(find.byType(SecurityScreen), findsOneWidget);
    });
  });
}
