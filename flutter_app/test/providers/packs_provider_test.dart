import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/packs_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('PacksProvider', () {
    late _MockApiClient api;
    late PacksProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = PacksProvider()..setApi(api);
    });

    test('initial state has no packs', () {
      expect(provider.packs, isEmpty);
      expect(provider.hasPackLoaded('any.pack'), false);
    });

    test('refresh populates packs from API response', () async {
      when(() => api.get('/api/v1/packs/loaded')).thenAnswer(
        (_) async => {
          'packs': [
            {
              'qualified_id': 'cognithor.deep_research',
              'version': '1.0.0',
              'display_name': 'Deep Research',
              'tools': ['research_one', 'research_two'],
            },
            {
              'qualified_id': 'cognithor.reddit',
              'version': '0.5.0',
              'display_name': 'Reddit Lead Source',
              'tools': [],
            },
          ],
        },
      );

      await provider.refresh();

      expect(provider.packs.length, 2);
      expect(provider.packs.first.qualifiedId, 'cognithor.deep_research');
      expect(provider.packs.first.tools, ['research_one', 'research_two']);
      expect(provider.hasPackLoaded('cognithor.reddit'), true);
      expect(provider.hasPackLoaded('cognithor.unknown'), false);
    });

    test('refresh swallows errors and leaves packs unchanged', () async {
      when(
        () => api.get('/api/v1/packs/loaded'),
      ).thenThrow(Exception('offline'));

      await provider.refresh();

      expect(provider.packs, isEmpty);
    });

    test('refresh handles missing "packs" key gracefully', () async {
      when(() => api.get('/api/v1/packs/loaded')).thenAnswer((_) async => {});

      await provider.refresh();

      expect(provider.packs, isEmpty);
    });

    test('refresh notifies listeners', () async {
      when(
        () => api.get('/api/v1/packs/loaded'),
      ).thenAnswer((_) async => {'packs': []});

      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      await provider.refresh();

      expect(notifyCount, greaterThanOrEqualTo(1));
    });

    test('LoadedPack.fromJson uses defaults for missing fields', () {
      final pack = LoadedPack.fromJson({});

      expect(pack.qualifiedId, '');
      expect(pack.version, '');
      expect(pack.displayName, '');
      expect(pack.tools, isEmpty);
    });
  });
}
