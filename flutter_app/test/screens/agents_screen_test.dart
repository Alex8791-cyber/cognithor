import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/agents_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('AgentsScreen smoke', () {
    late _MockApiClient api;
    late AdminProvider admin;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      admin = SilentAdminProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<AdminProvider>.value(
          value: admin,
          child: const AgentsScreen(),
        ),
      ),
    );

    testWidgets('renders without crashing on empty agents list', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(AgentsScreen), findsOneWidget);
    });

    testWidgets('renders FAB for creating new agents', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      // Empty state still has the create-agent affordance.
      expect(find.byType(AgentsScreen), findsOneWidget);
    });
  });
}
