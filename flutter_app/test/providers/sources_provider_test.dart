import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SourcesProvider', () {
    late _MockApiClient api;
    late SourcesProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = SourcesProvider()..setApi(api);
    });

    test('initial state is empty + not loading', () {
      expect(provider.sources, isEmpty);
      expect(provider.isEmpty, true);
      expect(provider.loading, false);
      expect(provider.error, isNull);
      expect(provider.hasSource('reddit'), false);
    });

    test('refresh populates sources from API response', () async {
      when(() => api.get('/api/v1/leads/sources')).thenAnswer(
        (_) async => {
          'sources': [
            {
              'source_id': 'reddit',
              'display_name': 'Reddit',
              'icon': 'reddit',
              'color': '#FF4500',
              'capabilities': ['monitor_keywords', 'reply'],
            },
            {
              'source_id': 'twitter',
              'display_name': 'Twitter',
              'icon': 'twitter',
              'color': '#1DA1F2',
              'capabilities': ['monitor_keywords'],
            },
          ],
        },
      );

      await provider.refresh();

      expect(provider.sources.length, 2);
      expect(provider.isEmpty, false);
      expect(provider.hasSource('reddit'), true);
      expect(provider.sources.first.capabilities, contains('reply'));
      expect(provider.loading, false);
      expect(provider.error, isNull);
    });

    test('refresh sets error on failure', () async {
      when(
        () => api.get('/api/v1/leads/sources'),
      ).thenThrow(Exception('unreachable'));

      await provider.refresh();

      expect(provider.sources, isEmpty);
      expect(provider.error, contains('unreachable'));
      expect(provider.loading, false);
    });

    test('refresh handles missing "sources" key gracefully', () async {
      when(() => api.get('/api/v1/leads/sources')).thenAnswer((_) async => {});

      await provider.refresh();

      expect(provider.sources, isEmpty);
      expect(provider.error, isNull);
    });

    test('refresh no-ops when API not set', () async {
      final unbound = SourcesProvider();

      await unbound.refresh();

      expect(unbound.loading, false);
      expect(unbound.sources, isEmpty);
      verifyZeroInteractions(api);
    });

    test('LeadSourceInfo.fromJson uses defaults for missing fields', () {
      final src = LeadSourceInfo.fromJson({});

      expect(src.sourceId, '');
      expect(src.displayName, '');
      expect(src.icon, '');
      expect(src.color, '');
      expect(src.capabilities, isEmpty);
    });
  });
}
