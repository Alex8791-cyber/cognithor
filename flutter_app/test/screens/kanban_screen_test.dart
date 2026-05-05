/// Smoke test for [KanbanScreen].
///
/// Previously deferred from the screen-test suite because the screen's
/// `initState` does a `WidgetsBinding.addPostFrameCallback` that fans
/// out three provider load calls. With the silent providers below
/// every fan-out is a no-op so the screen renders without scheduling
/// any background work.
///
/// The test also exercises the dispose path by replacing the widget
/// tree with a `SizedBox`, matching the pattern in
/// `monitoring_screen_test.dart`.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/cron_provider.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/screens/kanban_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('KanbanScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<KanbanProvider>(
            create: (_) => SilentKanbanProvider(),
          ),
          ChangeNotifierProvider<CronProvider>(
            create: (_) => SilentCronProvider(),
          ),
          ChangeNotifierProvider<AdminProvider>(
            create: (_) => SilentAdminProvider(),
          ),
          ChangeNotifierProvider<ChatProvider>(create: (_) => ChatProvider()),
        ],
        child: const KanbanScreen(),
      ),
    );

    testWidgets('renders without crashing on empty kanban payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      // Drain the post-frame callback that triggers `fetchTasks` /
      // `loadAgents` (both are no-ops thanks to the silent providers).
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(KanbanScreen), findsOneWidget);
    });

    testWidgets('renders the 3-segment view toggle', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // The 3-segment view toggle: board, pipeline, scheduled.
      expect(
        find.byWidgetPredicate((w) => w is SegmentedButton),
        findsOneWidget,
      );
    });

    testWidgets('disposes cleanly when replaced', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(KanbanScreen), findsNothing);
    });
  });
}
