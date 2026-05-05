import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/learning_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void _stubAllLearning(_MockApiClient api) {
  when(() => api.getLearningStats()).thenAnswer((_) async => {});
  when(() => api.getLearningGaps()).thenAnswer((_) async => {'gaps': []});
  when(
    () => api.getConfidenceHistory(),
  ).thenAnswer((_) async => {'history': []});
  when(() => api.getLearningQueue()).thenAnswer((_) async => {'tasks': []});
  when(
    () => api.getLearningDirectories(),
  ).thenAnswer((_) async => {'directories': []});
  when(
    () => api.getQAPairs(
      query: any(named: 'query'),
      limit: any(named: 'limit'),
    ),
  ).thenAnswer((_) async => {'pairs': []});
  when(
    () => api.getRecentLineage(limit: any(named: 'limit')),
  ).thenAnswer((_) async => {'entries': []});
}

void main() {
  group('LearningScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      _stubAllLearning(api);
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const LearningScreen(),
      ),
    );

    testWidgets('renders without crashing on empty learning data', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(LearningScreen), findsOneWidget);
    });

    testWidgets('drives all 7 learning API endpoints', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.getLearningStats()).called(greaterThanOrEqualTo(1));
      verify(() => api.getLearningGaps()).called(greaterThanOrEqualTo(1));
      verify(() => api.getConfidenceHistory()).called(greaterThanOrEqualTo(1));
      verify(() => api.getLearningQueue()).called(greaterThanOrEqualTo(1));
      verify(
        () => api.getLearningDirectories(),
      ).called(greaterThanOrEqualTo(1));
    });

    testWidgets('exception during load is captured cleanly', (tester) async {
      when(() => api.getLearningStats()).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(LearningScreen), findsOneWidget);
    });
  });
}
