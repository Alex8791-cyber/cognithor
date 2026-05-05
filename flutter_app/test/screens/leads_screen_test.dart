import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/screens/leads_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('LeadsScreen smoke', () {
    late _MockApiClient api;
    late RedditLeadsProvider leads;
    late SourcesProvider sources;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      leads = SilentRedditLeadsProvider();
      sources = SilentSourcesProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<RedditLeadsProvider>.value(
          value: leads,
          child: ChangeNotifierProvider<SourcesProvider>.value(
            value: sources,
            child: const LeadsScreen(),
          ),
        ),
      ),
    );

    testWidgets('renders without crashing on empty leads list', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(LeadsScreen), findsOneWidget);
    });
  });
}
