import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  setUpAll(() {
    registerFallbackValue(<String, dynamic>{});
  });

  group('ConfigProvider', () {
    late _MockApiClient api;
    late ConfigProvider p;

    setUp(() {
      api = _MockApiClient();
      p = ConfigProvider()..setApi(api);
    });

    test('initial state has defaults injected after loadAll', () async {
      when(() => api.get(any())).thenAnswer((_) async => {});

      await p.loadAll();

      // Defaults are merged in, so cfg should NOT be empty.
      expect(p.cfg, isNotEmpty);
      expect(p.cfg['language'], 'de');
      expect(p.cfg['llm_backend_type'], 'ollama');
      expect(p.loading, isFalse);
      expect(p.error, isNull);
    });

    test(
      'loadAll surfaces error from /config but still merges defaults',
      () async {
        when(() => api.get(any())).thenAnswer((_) async => {});
        when(() => api.get('config')).thenThrow(Exception('boom'));

        await p.loadAll();

        expect(p.error, contains('boom'));
        expect(p.cfg['language'], 'de'); // defaults still applied
        expect(p.loading, isFalse);
      },
    );

    test('loadAll honours overlay values from server', () async {
      when(() => api.get(any())).thenAnswer((_) async => {});
      when(() => api.get('config')).thenAnswer(
        (_) async => {
          'owner_name': 'Alex',
          'language': 'en',
          'ollama': {'base_url': 'http://other:11434'},
        },
      );

      await p.loadAll();

      expect(p.cfg['owner_name'], 'Alex');
      expect(p.cfg['language'], 'en');
      // Deep merge: overlay key wins, untouched defaults preserved.
      final ollama = p.cfg['ollama'] as Map<String, dynamic>;
      expect(ollama['base_url'], 'http://other:11434');
      expect(ollama['timeout_seconds'], 120); // from defaults
    });

    test('getPath / set traverse and mutate via dot-path', () {
      p.set('models.planner.name', 'qwen3:14b');

      expect(p.getPath('models.planner.name'), 'qwen3:14b');
      expect(p.getPath('does.not.exist'), isNull);
    });

    test('hasChanges flips when cfg mutated after a snapshot', () async {
      when(() => api.get(any())).thenAnswer((_) async => {});
      when(
        () => api.get('config'),
      ).thenAnswer((_) async => {'owner_name': 'A'});

      await p.loadAll();
      expect(p.hasChanges, isFalse);

      p.set('owner_name', 'B');
      expect(p.hasChanges, isTrue);
    });

    test('discard restores last saved snapshot', () async {
      when(() => api.get(any())).thenAnswer((_) async => {});
      when(
        () => api.get('config'),
      ).thenAnswer((_) async => {'owner_name': 'A'});

      await p.loadAll();
      p.set('owner_name', 'modified');
      expect(p.cfg['owner_name'], 'modified');

      p.discard();
      expect(p.cfg['owner_name'], 'A');
      expect(p.hasChanges, isFalse);
    });

    test('addAgent / updateAgent / removeAgent mutate list and notify', () {
      var notified = 0;
      p.addListener(() => notified++);

      p.addAgent({'name': 'planner'});
      expect(p.agents, hasLength(1));

      p.updateAgent(0, {'name': 'planner', 'temperature': 0.5});
      expect(p.agents.first['temperature'], 0.5);

      p.removeAgent(0);
      expect(p.agents, isEmpty);

      // Out-of-range guards: must not throw.
      p.updateAgent(99, {});
      p.removeAgent(99);

      expect(notified, greaterThanOrEqualTo(3));
    });

    test('exportJson + importJson round-trips state', () async {
      when(() => api.get(any())).thenAnswer((_) async => {});
      when(
        () => api.get('config'),
      ).thenAnswer((_) async => {'owner_name': 'A'});
      await p.loadAll();
      p.addAgent({'name': 'planner'});

      final exported = p.exportJson();
      final fresh = ConfigProvider()..setApi(api);
      await fresh.importJson(exported);

      expect(fresh.cfg['owner_name'], 'A');
      expect(fresh.agents.single['name'], 'planner');
    });

    test('importJson with malformed JSON sets an error', () async {
      await p.importJson('not json {');
      expect(p.error, contains('Invalid JSON'));
    });

    test('leadsEngineEnabled reflects social toggles', () {
      // Default cfg has no `social` map → false
      expect(p.leadsEngineEnabled, isFalse);

      p.set('social.reddit_scan_enabled', true);
      expect(p.leadsEngineEnabled, isTrue);

      p.set('social.reddit_scan_enabled', false);
      p.set('social.rss_enabled', true);
      expect(p.leadsEngineEnabled, isTrue);
    });

    test(
      'importJson accepts a bare config blob (no top-level wrapper)',
      () async {
        final raw = jsonEncode({'owner_name': 'Bare', 'language': 'fr'});
        await p.importJson(raw);
        expect(p.cfg['owner_name'], 'Bare');
        expect(p.cfg['language'], 'fr');
        // Defaults still merged
        expect(p.cfg['llm_backend_type'], 'ollama');
      },
    );
  });
}
