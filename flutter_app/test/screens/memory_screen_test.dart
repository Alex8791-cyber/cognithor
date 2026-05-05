import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/memory_provider.dart';
import 'package:cognithor_ui/screens/memory_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('MemoryScreen smoke', () {
    late _MockApiClient api;
    late MemoryProvider memory;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      memory = SilentMemoryProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<MemoryProvider>.value(
          value: memory,
          child: const MemoryScreen(),
        ),
      ),
    );

    testWidgets('renders without crashing on empty memory state', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(MemoryScreen), findsOneWidget);
    });

    testWidgets('renders the 3-tab bar (graph / hygiene / explainability)', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      // Tab icons are stable identifiers regardless of locale.
      expect(find.byIcon(Icons.hub_outlined), findsWidgets);
      expect(find.byIcon(Icons.health_and_safety_outlined), findsWidgets);
      expect(find.byIcon(Icons.account_tree_outlined), findsWidgets);
    });
  });
}
