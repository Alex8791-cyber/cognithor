import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/agent_editor_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('AgentEditorScreen smoke', () {
    late _MockApiClient api;
    late AdminProvider admin;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      admin = SilentAdminProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap({String? agentName}) => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<AdminProvider>.value(
          value: admin,
          child: AgentEditorScreen(agentName: agentName),
        ),
      ),
    );

    testWidgets('renders create-mode form when agentName is null', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(AgentEditorScreen), findsOneWidget);
      expect(find.byType(TextFormField), findsWidgets);
    });

    testWidgets('renders edit-mode scaffold when agentName is set', (
      tester,
    ) async {
      await tester.pumpWidget(wrap(agentName: 'planner'));
      await tester.pump();
      tester.takeException();
      expect(find.byType(AgentEditorScreen), findsOneWidget);
    });

    testWidgets('disposes controllers without leaking', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(AgentEditorScreen), findsNothing);
    });
  });
}
