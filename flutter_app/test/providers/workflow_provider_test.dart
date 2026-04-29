import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/workflow_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('WorkflowProvider', () {
    late _MockApiClient api;
    late WorkflowProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = WorkflowProvider()..setApi(api);
    });

    test('initial state is empty', () {
      expect(provider.categories, isEmpty);
      expect(provider.isLoading, false);
      expect(provider.error, isNull);
    });

    test('loadCategories populates categories from API response', () async {
      when(() => api.getWorkflowCategories()).thenAnswer(
        (_) async => {
          'categories': [
            {'id': 'a', 'name': 'Alpha'},
            {'id': 'b', 'name': 'Beta'},
          ],
        },
      );

      await provider.loadCategories();

      expect(provider.categories.length, 2);
      expect(provider.isLoading, false);
      expect(provider.error, isNull);
    });

    test('loadCategories sets error on failure', () async {
      when(
        () => api.getWorkflowCategories(),
      ).thenThrow(Exception('network down'));

      await provider.loadCategories();

      expect(provider.categories, isEmpty);
      expect(provider.error, contains('network down'));
      expect(provider.isLoading, false);
    });

    test('loadCategories no-ops when API not set', () async {
      final unboundProvider = WorkflowProvider();

      await unboundProvider.loadCategories();

      expect(unboundProvider.isLoading, false);
      expect(unboundProvider.categories, isEmpty);
      verifyZeroInteractions(api);
    });

    test(
      'startWorkflow forwards templateId and clears error on success',
      () async {
        when(() => api.startWorkflow('tpl-1')).thenAnswer((_) async => {});

        await provider.startWorkflow('tpl-1');

        verify(() => api.startWorkflow('tpl-1')).called(1);
        expect(provider.error, isNull);
        expect(provider.isLoading, false);
      },
    );

    test('startWorkflow sets error on failure', () async {
      when(() => api.startWorkflow(any())).thenThrow(Exception('boom'));

      await provider.startWorkflow('tpl-2');

      expect(provider.error, contains('boom'));
      expect(provider.isLoading, false);
    });
  });
}
