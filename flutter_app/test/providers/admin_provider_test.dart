import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('AdminProvider', () {
    late _MockApiClient api;
    late AdminProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = AdminProvider()..setApi(api);
    });

    test('initial state', () {
      final fresh = AdminProvider();
      expect(fresh.systemStatus, isNull);
      expect(fresh.agents, isEmpty);
      expect(fresh.models, isNull);
      expect(fresh.modelStats, isNull);
      expect(fresh.vaultStats, isNull);
      expect(fresh.vaultAgents, isEmpty);
      expect(fresh.credentials, isEmpty);
      expect(fresh.bindings, isEmpty);
      expect(fresh.commands, isEmpty);
      expect(fresh.connectors, isEmpty);
      expect(fresh.isolationStats, isNull);
      expect(fresh.circles, isNull);
      expect(fresh.isLoading, isFalse);
      expect(fresh.error, isNull);
    });

    group('without API client (no setApi)', () {
      late AdminProvider noApi;

      setUp(() {
        noApi = AdminProvider();
      });

      test('loadSystemStatus is a no-op', () async {
        await noApi.loadSystemStatus();
        expect(noApi.systemStatus, isNull);
        expect(noApi.isLoading, isFalse);
      });

      test('loadAgents is a no-op', () async {
        await noApi.loadAgents();
        expect(noApi.agents, isEmpty);
      });

      test('createAgent returns false', () async {
        final ok = await noApi.createAgent({'name': 'x'});
        expect(ok, isFalse);
      });

      test('getAgent returns null', () async {
        final res = await noApi.getAgent('x');
        expect(res, isNull);
      });
    });

    group('loadSystemStatus', () {
      test('populates systemStatus on success', () async {
        when(
          () => api.getSystemStatus(),
        ).thenAnswer((_) async => {'status': 'ok', 'uptime': 1234});

        await provider.loadSystemStatus();

        expect(provider.systemStatus, {'status': 'ok', 'uptime': 1234});
        expect(provider.isLoading, isFalse);
        expect(provider.error, isNull);
      });

      test('captures exception in error field', () async {
        when(() => api.getSystemStatus()).thenThrow(Exception('boom'));

        await provider.loadSystemStatus();

        expect(provider.error, contains('boom'));
        expect(provider.isLoading, isFalse);
      });
    });

    group('loadAgents', () {
      test('populates agents from response', () async {
        when(() => api.getAgents()).thenAnswer(
          (_) async => {
            'agents': [
              {'name': 'planner'},
              {'name': 'executor'},
            ],
          },
        );

        await provider.loadAgents();

        expect(provider.agents.length, 2);
        expect(
          (provider.agents.first as Map<String, dynamic>)['name'],
          'planner',
        );
        expect(provider.error, isNull);
      });

      test('falls back to empty list when key missing', () async {
        when(() => api.getAgents()).thenAnswer((_) async => {});

        await provider.loadAgents();

        expect(provider.agents, isEmpty);
      });
    });

    group('createAgent', () {
      test('returns true and triggers loadAgents on success', () async {
        when(
          () => api.createAgent(any()),
        ).thenAnswer((_) async => {'ok': true});
        when(() => api.getAgents()).thenAnswer(
          (_) async => {
            'agents': [
              {'name': 'planner'},
            ],
          },
        );

        final ok = await provider.createAgent({'name': 'planner'});

        expect(ok, isTrue);
        expect(provider.agents.length, 1);
        verify(() => api.createAgent({'name': 'planner'})).called(1);
        verify(() => api.getAgents()).called(1);
      });

      test('returns false and surfaces server error', () async {
        when(
          () => api.createAgent(any()),
        ).thenAnswer((_) async => {'error': 'duplicate'});

        final ok = await provider.createAgent({'name': 'planner'});

        expect(ok, isFalse);
        expect(provider.error, 'duplicate');
      });

      test('returns false on exception', () async {
        when(() => api.createAgent(any())).thenThrow(Exception('network'));

        final ok = await provider.createAgent({'name': 'planner'});

        expect(ok, isFalse);
        expect(provider.error, contains('network'));
      });
    });

    group('updateAgent', () {
      test('returns true and reloads on success', () async {
        when(
          () => api.updateAgent(any(), any()),
        ).thenAnswer((_) async => {'ok': true});
        when(() => api.getAgents()).thenAnswer((_) async => {'agents': []});

        final ok = await provider.updateAgent('planner', {'temperature': 0.5});

        expect(ok, isTrue);
        verify(
          () => api.updateAgent('planner', {'temperature': 0.5}),
        ).called(1);
      });

      test('returns false and surfaces server error', () async {
        when(
          () => api.updateAgent(any(), any()),
        ).thenAnswer((_) async => {'error': 'not_found'});

        final ok = await provider.updateAgent('planner', {});

        expect(ok, isFalse);
        expect(provider.error, 'not_found');
      });
    });

    group('deleteAgent', () {
      test('returns true and reloads on success', () async {
        when(() => api.deleteAgent(any())).thenAnswer((_) async => {});
        when(() => api.getAgents()).thenAnswer((_) async => {'agents': []});

        final ok = await provider.deleteAgent('planner');

        expect(ok, isTrue);
      });

      test('returns false on server error', () async {
        when(
          () => api.deleteAgent(any()),
        ).thenAnswer((_) async => {'error': 'cannot delete'});

        final ok = await provider.deleteAgent('planner');

        expect(ok, isFalse);
        expect(provider.error, 'cannot delete');
      });
    });

    group('getAgent', () {
      test('returns map when API returns valid data', () async {
        when(
          () => api.getAgent('planner'),
        ).thenAnswer((_) async => {'name': 'planner', 'model': 'qwen3:8b'});

        final res = await provider.getAgent('planner');

        expect(res, isNotNull);
        expect(res!['name'], 'planner');
      });

      test('returns null when API responds with error', () async {
        when(
          () => api.getAgent('missing'),
        ).thenAnswer((_) async => {'error': 'not_found'});

        final res = await provider.getAgent('missing');

        expect(res, isNull);
      });

      test('returns null on exception and stores error', () async {
        when(() => api.getAgent('boom')).thenThrow(Exception('bad'));

        final res = await provider.getAgent('boom');

        expect(res, isNull);
        expect(provider.error, contains('bad'));
      });
    });
  });
}
