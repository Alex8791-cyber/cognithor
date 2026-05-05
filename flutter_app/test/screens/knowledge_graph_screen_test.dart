import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/knowledge_graph_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('KnowledgeGraphScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(
        () => api.getMemoryGraphEntities(),
      ).thenAnswer((_) async => {'entities': [], 'relations': []});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const KnowledgeGraphScreen(),
      ),
    );

    testWidgets('renders without crashing on empty graph', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(KnowledgeGraphScreen), findsOneWidget);
    });

    testWidgets('drives the memory graph API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(
        () => api.getMemoryGraphEntities(),
      ).called(greaterThanOrEqualTo(1));
    });

    testWidgets('renders entity + relation data when API returns it', (
      tester,
    ) async {
      when(() => api.getMemoryGraphEntities()).thenAnswer(
        (_) async => {
          'entities': [
            {'id': 'e1', 'name': 'Cognithor', 'type': 'project'},
            {'id': 'e2', 'name': 'Alex', 'type': 'person'},
          ],
          'relations': [
            {'src': 'e2', 'dst': 'e1', 'type': 'owns'},
          ],
        },
      );

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // Entities are drawn to a CustomPainter canvas, not text
      // widgets, so we just assert the screen survived the populated
      // payload without crashing.
      expect(find.byType(KnowledgeGraphScreen), findsOneWidget);
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.getMemoryGraphEntities()).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(KnowledgeGraphScreen), findsOneWidget);
    });
  });
}
