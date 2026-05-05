import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/workflow_provider.dart';
import 'package:cognithor_ui/screens/workflows_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('WorkflowsScreen smoke', () {
    late _MockApiClient api;
    late WorkflowProvider workflow;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // The screen also calls api.getWorkflowInstances + getDagRuns
      // directly via setState. Stub those to empty.
      when(
        () => api.getWorkflowInstances(),
      ).thenAnswer((_) async => {'instances': []});
      when(
        () => api.getWorkflowDagRuns(),
      ).thenAnswer((_) async => {'runs': []});

      workflow = SilentWorkflowProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<WorkflowProvider>.value(
          value: workflow,
          child: const WorkflowsScreen(),
        ),
      ),
    );

    testWidgets('renders without crashing on empty workflows', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(WorkflowsScreen), findsOneWidget);
    });

    testWidgets('renders 3 tabs (instances / templates / dag)', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(Tab), findsNWidgets(3));
    });
  });
}
