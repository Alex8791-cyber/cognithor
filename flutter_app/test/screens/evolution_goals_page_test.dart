import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/evolution_provider.dart';
import 'package:cognithor_ui/screens/evolution_goals_page.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('EvolutionGoalsPage smoke', () {
    late _MockApiClient api;
    late EvolutionProvider evo;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // The provider's fetchAll() fans out to 4 GETs.
      when(() => api.get(any())).thenAnswer((_) async => <String, dynamic>{});

      evo = EvolutionProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    tearDown(() {
      evo.dispose();
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<EvolutionProvider>.value(
          value: evo,
          child: const EvolutionGoalsPage(),
        ),
      ),
    );

    testWidgets('renders 3 tabs in the TabController', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(EvolutionGoalsPage), findsOneWidget);
      // The page has length=3 TabController; expect Tab widgets.
      expect(find.byType(Tab), findsNWidgets(3));
    });

    testWidgets('drives evolution endpoints via provider', (tester) async {
      await tester.pumpWidget(wrap());
      // postFrameCallback fires after first frame; pump several times.
      await tester.pump();
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // fetchAll fans out to goals + plans + journal + stats.
      verify(() => api.get(any())).called(greaterThanOrEqualTo(4));
    });

    testWidgets('exception during fetchAll is captured cleanly', (
      tester,
    ) async {
      when(() => api.get(any())).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(EvolutionGoalsPage), findsOneWidget);
    });
  });
}
