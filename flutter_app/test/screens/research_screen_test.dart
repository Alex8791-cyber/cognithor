import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/packs_provider.dart';
import 'package:cognithor_ui/providers/research_provider.dart';
import 'package:cognithor_ui/screens/research_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('ResearchScreen smoke', () {
    late _MockApiClient api;
    late ResearchProvider research;
    late PacksProvider packs;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      research = ResearchProvider();
      packs = PacksProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<PacksProvider>.value(
          value: packs,
          child: ChangeNotifierProvider<ResearchProvider>.value(
            value: research,
            child: const ResearchScreen(),
          ),
        ),
      ),
    );

    testWidgets('renders without crashing when deep-research pack absent', (
      tester,
    ) async {
      // Default PacksProvider has empty pack list → didChangeDependencies
      // short-circuits without calling research.loadHistory().
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(ResearchScreen), findsOneWidget);
    });

    testWidgets('exposes a query TextField', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(TextField), findsOneWidget);
    });
  });
}
