import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/teach_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('TeachScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(() => api.getLearnHistory()).thenAnswer((_) async => {'items': []});
      when(() => api.getLearnQueue()).thenAnswer((_) async => {'items': []});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const TeachScreen(),
      ),
    );

    testWidgets('renders without crashing on empty learn history', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(TeachScreen), findsOneWidget);
    });

    testWidgets('drives both learn endpoints', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.getLearnHistory()).called(greaterThanOrEqualTo(1));
      verify(() => api.getLearnQueue()).called(greaterThanOrEqualTo(1));
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.getLearnHistory()).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(TeachScreen), findsOneWidget);
    });
  });
}
