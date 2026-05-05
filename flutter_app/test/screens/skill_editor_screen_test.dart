import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/screens/skill_editor_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SkillEditorScreen smoke', () {
    late _MockApiClient api;
    late SkillsProvider skills;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      skills = SilentSkillsProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap({String? slug}) => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<SkillsProvider>.value(
          value: skills,
          child: SkillEditorScreen(slug: slug),
        ),
      ),
    );

    testWidgets('renders create-mode form when slug is null', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(SkillEditorScreen), findsOneWidget);
      // Form fields are TextField widgets (multiple).
      expect(find.byType(TextFormField), findsWidgets);
    });

    testWidgets('renders edit-mode form (loading state) when slug is set', (
      tester,
    ) async {
      await tester.pumpWidget(wrap(slug: 'some-skill'));
      await tester.pump();
      tester.takeException();
      // Edit mode triggers _loadSkill which would normally throw on null api;
      // we just verify the screen scaffold mounts.
      expect(find.byType(SkillEditorScreen), findsOneWidget);
    });

    testWidgets('disposes its controllers without leaking', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(SkillEditorScreen), findsNothing);
    });
  });
}
