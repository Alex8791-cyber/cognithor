import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/memory_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('MemoryProvider', () {
    late _MockApiClient api;
    late MemoryProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = MemoryProvider()..setApi(api);
    });

    test('initial state is empty', () {
      expect(provider.graphStats, isNull);
      expect(provider.entities, isEmpty);
      expect(provider.hygieneStats, isNull);
      expect(provider.quarantined, isEmpty);
      expect(provider.explainabilityStats, isNull);
      expect(provider.trails, isEmpty);
      expect(provider.lowTrustTrails, isEmpty);
      expect(provider.isLoading, false);
      expect(provider.error, isNull);
    });

    test('loadGraphStats stores result', () async {
      when(
        () => api.getMemoryGraphStats(),
      ).thenAnswer((_) async => {'nodes': 42, 'edges': 99});

      await provider.loadGraphStats();

      expect(provider.graphStats, {'nodes': 42, 'edges': 99});
      expect(provider.error, isNull);
      expect(provider.isLoading, false);
    });

    test('loadEntities populates entities list', () async {
      when(() => api.getMemoryGraphEntities()).thenAnswer(
        (_) async => {
          'entities': [
            {'id': 'e1'},
            {'id': 'e2'},
          ],
        },
      );

      await provider.loadEntities();

      expect(provider.entities.length, 2);
      expect(provider.error, isNull);
    });

    test('loadHygieneStats sets error on failure', () async {
      when(() => api.getHygieneStats()).thenThrow(Exception('boom'));

      await provider.loadHygieneStats();

      expect(provider.hygieneStats, isNull);
      expect(provider.error, contains('boom'));
      expect(provider.isLoading, false);
    });

    test('loadQuarantine populates quarantined list', () async {
      when(() => api.getQuarantine()).thenAnswer(
        (_) async => {
          'items': [
            {'id': 'q1'},
          ],
        },
      );

      await provider.loadQuarantine();

      expect(provider.quarantined.length, 1);
    });

    test('loadTrails populates trails list', () async {
      when(() => api.getExplainabilityTrails()).thenAnswer(
        (_) async => {
          'trails': [
            {'id': 't1'},
            {'id': 't2'},
            {'id': 't3'},
          ],
        },
      );

      await provider.loadTrails();

      expect(provider.trails.length, 3);
    });

    test('loadLowTrustTrails populates lowTrustTrails list', () async {
      when(() => api.getLowTrustTrails()).thenAnswer(
        (_) async => {
          'trails': [
            {'id': 'lt1'},
          ],
        },
      );

      await provider.loadLowTrustTrails();

      expect(provider.lowTrustTrails.length, 1);
    });

    test('loaders no-op when API not set', () async {
      final unbound = MemoryProvider();
      await unbound.loadGraphStats();
      await unbound.loadEntities();
      await unbound.loadHygieneStats();
      expect(unbound.isLoading, false);
      verifyZeroInteractions(api);
    });
  });
}
