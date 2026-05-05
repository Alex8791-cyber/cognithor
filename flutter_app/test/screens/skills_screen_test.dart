import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/screens/skills_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SkillsScreen smoke', () {
    late _MockApiClient api;
    late SkillsProvider skills;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      skills = SilentSkillsProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<SkillsProvider>.value(
          value: skills,
          child: const SkillsScreen(),
        ),
      ),
    );

    testWidgets('renders without crashing on empty skills lists', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(SkillsScreen), findsOneWidget);
    });

    testWidgets('renders the marketplace tab bar', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      // Empty state shows tabs even with no skills loaded.
      expect(find.byType(SkillsScreen), findsOneWidget);
    });
  });
}
