import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  setUpAll(() {
    registerFallbackValue(<String, dynamic>{});
  });

  group('SkillsProvider', () {
    late _MockApiClient api;
    late SkillsProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = SkillsProvider()..setApi(api);
    });

    test('initial state is empty', () {
      expect(provider.featured, isEmpty);
      expect(provider.trending, isEmpty);
      expect(provider.categories, isEmpty);
      expect(provider.searchResults, isEmpty);
      expect(provider.installed, isEmpty);
      expect(provider.searchQuery, '');
      expect(provider.isLoading, false);
      expect(provider.error, isNull);
    });

    test(
      'loadFeatured prefers "featured" key, falls back to "skills"',
      () async {
        when(() => api.getMarketplaceFeatured()).thenAnswer(
          (_) async => {
            'skills': [
              {'id': 's1'},
              {'id': 's2'},
            ],
          },
        );
        await provider.loadFeatured();
        expect(provider.featured.length, 2);

        when(() => api.getMarketplaceFeatured()).thenAnswer(
          (_) async => {
            'featured': [
              {'id': 'f1'},
            ],
            'skills': [
              {'id': 's-extra'},
            ],
          },
        );
        await provider.loadFeatured();
        expect(provider.featured.length, 1);
        expect(provider.featured.first['id'], 'f1');
      },
    );

    test('loadTrending sets error on failure', () async {
      when(
        () => api.getMarketplaceTrending(),
      ).thenThrow(Exception('upstream-down'));
      await provider.loadTrending();
      expect(provider.error, contains('upstream-down'));
      expect(provider.isLoading, false);
    });

    test('loadCategories handles non-list response gracefully', () async {
      when(
        () => api.getMarketplaceCategories(),
      ).thenAnswer((_) async => {'categories': 'not-a-list'});
      await provider.loadCategories();
      expect(provider.categories, isEmpty);
    });

    test('search updates searchQuery + populates searchResults', () async {
      when(() => api.searchMarketplace('foo')).thenAnswer(
        (_) async => {
          'results': [
            {'id': 'r1'},
          ],
        },
      );
      await provider.search('foo');
      expect(provider.searchQuery, 'foo');
      expect(provider.searchResults.length, 1);
    });

    test(
      'loadInstalled prefers "installed" key, falls back to "skills"',
      () async {
        when(() => api.getInstalledSkills()).thenAnswer(
          (_) async => {
            'skills': [
              {'id': 'i1'},
            ],
          },
        );
        await provider.loadInstalled();
        expect(provider.installed.length, 1);
      },
    );

    test('installSkill calls API + reloads installed list', () async {
      when(() => api.installSkill('skill-1')).thenAnswer((_) async => {});
      when(() => api.getInstalledSkills()).thenAnswer(
        (_) async => {
          'installed': [
            {'id': 'skill-1'},
          ],
        },
      );

      await provider.installSkill('skill-1');

      verify(() => api.installSkill('skill-1')).called(1);
      expect(provider.installed.length, 1);
      expect(provider.error, isNull);
    });

    test('uninstallSkill calls API + reloads installed list', () async {
      when(() => api.uninstallSkill('skill-1')).thenAnswer((_) async => {});
      when(
        () => api.getInstalledSkills(),
      ).thenAnswer((_) async => {'installed': []});

      await provider.uninstallSkill('skill-1');

      verify(() => api.uninstallSkill('skill-1')).called(1);
      expect(provider.installed, isEmpty);
    });

    test('createSkill returns true on success', () async {
      when(
        () => api.post('skill-registry/create', any()),
      ).thenAnswer((_) async => {});
      when(
        () => api.getInstalledSkills(),
      ).thenAnswer((_) async => {'installed': []});

      final ok = await provider.createSkill({'name': 'x'});
      expect(ok, true);
    });

    test('createSkill returns false + sets error on failure', () async {
      when(
        () => api.post('skill-registry/create', any()),
      ).thenThrow(Exception('schema-invalid'));

      final ok = await provider.createSkill({'name': 'x'});
      expect(ok, false);
      expect(provider.error, contains('schema-invalid'));
    });

    test('updateSkill PUTs to slug-keyed endpoint', () async {
      when(
        () => api.put('skill-registry/my-skill', any()),
      ).thenAnswer((_) async => {});
      when(
        () => api.getInstalledSkills(),
      ).thenAnswer((_) async => {'installed': []});

      final ok = await provider.updateSkill('my-skill', {'name': 'updated'});
      expect(ok, true);
      verify(() => api.put('skill-registry/my-skill', any())).called(1);
    });

    test('deleteSkill DELETEs the slug-keyed endpoint', () async {
      when(() => api.delete('skill-registry/foo')).thenAnswer((_) async => {});
      when(
        () => api.getInstalledSkills(),
      ).thenAnswer((_) async => {'installed': []});

      final ok = await provider.deleteSkill('foo');
      expect(ok, true);
    });

    test('toggleSkill PUTs to /toggle endpoint with empty body', () async {
      when(
        () => api.put('skill-registry/foo/toggle', {}),
      ).thenAnswer((_) async => {});
      when(
        () => api.getInstalledSkills(),
      ).thenAnswer((_) async => {'installed': []});

      final ok = await provider.toggleSkill('foo');
      expect(ok, true);
    });

    test('exportSkill returns "skill_md" payload', () async {
      when(
        () => api.get('skill-registry/foo/export'),
      ).thenAnswer((_) async => {'skill_md': '# Skill foo\n'});

      final md = await provider.exportSkill('foo');
      expect(md, '# Skill foo\n');
    });

    test('exportSkill returns null on failure', () async {
      when(
        () => api.get('skill-registry/foo/export'),
      ).thenThrow(Exception('boom'));

      final md = await provider.exportSkill('foo');
      expect(md, isNull);
    });

    test('getSkillDetail returns map on success, null on failure', () async {
      when(
        () => api.get('skill-registry/x'),
      ).thenAnswer((_) async => {'name': 'x'});
      expect(await provider.getSkillDetail('x'), {'name': 'x'});

      when(() => api.get('skill-registry/missing')).thenThrow(Exception('404'));
      expect(await provider.getSkillDetail('missing'), isNull);
    });

    test('CRUD methods no-op when API is unset', () async {
      final unbound = SkillsProvider();
      expect(await unbound.getSkillDetail('x'), isNull);
      expect(await unbound.createSkill({}), false);
      expect(await unbound.updateSkill('x', {}), false);
      expect(await unbound.deleteSkill('x'), false);
      expect(await unbound.toggleSkill('x'), false);
      expect(await unbound.exportSkill('x'), isNull);
      verifyZeroInteractions(api);
    });
  });
}
