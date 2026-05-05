import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/sessions_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SessionsProvider', () {
    late _MockApiClient api;
    late SessionsProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = SessionsProvider()..setApi(api);
    });

    test('initial state', () {
      expect(provider.sessions, isEmpty);
      expect(provider.folders, isEmpty);
      expect(provider.activeSessionId, isNull);
      expect(provider.isLoading, false);
      expect(provider.error, isNull);
    });

    group('loadSessions', () {
      test('populates sessions from API response', () async {
        when(() => api.listSessions()).thenAnswer(
          (_) async => {
            'sessions': <Map<String, dynamic>>[
              {'id': 's1', 'title': 'first'},
              {'id': 's2', 'title': 'second'},
            ],
          },
        );

        await provider.loadSessions();

        expect(provider.sessions.length, 2);
        expect(provider.sessions.first['id'], 's1');
        expect(provider.isLoading, false);
        expect(provider.error, isNull);
      });

      test('captures API error response', () async {
        when(
          () => api.listSessions(),
        ).thenAnswer((_) async => {'error': 'offline'});

        await provider.loadSessions();

        expect(provider.error, 'offline');
        expect(provider.sessions, isEmpty);
      });

      test('catches thrown exceptions', () async {
        when(() => api.listSessions()).thenThrow(Exception('boom'));

        await provider.loadSessions();

        expect(provider.error, contains('boom'));
        expect(provider.isLoading, false);
      });

      test('returns immediately when api not set', () async {
        final bare = SessionsProvider();
        await bare.loadSessions();
        expect(bare.sessions, isEmpty);
        expect(bare.isLoading, false);
      });
    });

    group('createNewSession', () {
      test('sets activeSessionId and reloads', () async {
        when(
          () => api.createSession(),
        ).thenAnswer((_) async => {'session_id': 'new-id'});
        when(
          () => api.listSessions(),
        ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});

        final id = await provider.createNewSession();

        expect(id, 'new-id');
        expect(provider.activeSessionId, 'new-id');
        verify(() => api.listSessions()).called(1);
      });

      test('captures error from API', () async {
        when(
          () => api.createSession(),
        ).thenAnswer((_) async => {'error': 'rate-limit'});

        final id = await provider.createNewSession();

        expect(id, isNull);
        expect(provider.error, 'rate-limit');
      });

      test('returns null when api not set', () async {
        final bare = SessionsProvider();
        final id = await bare.createNewSession();
        expect(id, isNull);
      });
    });

    group('createIncognitoSession', () {
      test('sets activeSessionId and reloads', () async {
        when(
          () => api.createIncognitoSession(),
        ).thenAnswer((_) async => {'session_id': 'incog-1'});
        when(
          () => api.listSessions(),
        ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});

        final id = await provider.createIncognitoSession();

        expect(id, 'incog-1');
        expect(provider.activeSessionId, 'incog-1');
      });

      test('returns null on API error', () async {
        when(
          () => api.createIncognitoSession(),
        ).thenAnswer((_) async => {'error': 'denied'});

        final id = await provider.createIncognitoSession();

        expect(id, isNull);
        expect(provider.activeSessionId, isNull);
      });
    });

    group('loadHistory', () {
      test('returns messages and sets activeSessionId', () async {
        when(() => api.getSessionHistory('s1')).thenAnswer(
          (_) async => {
            'messages': <Map<String, dynamic>>[
              {'role': 'user', 'text': 'hi'},
            ],
          },
        );

        final result = await provider.loadHistory('s1');

        expect(result, isNotNull);
        expect(result!.length, 1);
        expect(result.first['text'], 'hi');
        expect(provider.activeSessionId, 's1');
      });

      test('returns null on error', () async {
        when(
          () => api.getSessionHistory('s1'),
        ).thenAnswer((_) async => {'error': 'gone'});

        final result = await provider.loadHistory('s1');

        expect(result, isNull);
        expect(provider.error, 'gone');
      });
    });

    group('deleteSession', () {
      test('clears activeSessionId when deleting active', () async {
        provider.activeSessionId = 's1';
        when(() => api.deleteSession('s1')).thenAnswer((_) async => {});
        when(
          () => api.listSessions(),
        ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});

        await provider.deleteSession('s1');

        expect(provider.activeSessionId, isNull);
      });

      test('keeps activeSessionId when deleting other', () async {
        provider.activeSessionId = 's1';
        when(() => api.deleteSession('s2')).thenAnswer((_) async => {});
        when(
          () => api.listSessions(),
        ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});

        await provider.deleteSession('s2');

        expect(provider.activeSessionId, 's1');
      });
    });

    group('searchChats', () {
      test('empty query clears results', () async {
        provider.searchResults = [
          {'id': 's1'},
        ];

        await provider.searchChats('   ');

        expect(provider.searchResults, isEmpty);
      });

      test('populates results from API', () async {
        when(() => api.searchSessions('foo')).thenAnswer(
          (_) async => {
            'results': [
              {'id': 's1', 'snippet': 'foo'},
            ],
          },
        );

        await provider.searchChats('foo');

        expect(provider.searchResults.length, 1);
        expect(provider.searchResults.first['id'], 's1');
      });

      test('clears results on exception', () async {
        when(() => api.searchSessions('boom')).thenThrow(Exception('x'));

        provider.searchResults = [
          {'id': 'old'},
        ];
        await provider.searchChats('boom');

        expect(provider.searchResults, isEmpty);
      });
    });

    group('sessionsByProject', () {
      test('groups by folder with Allgemein last', () {
        provider.sessions = [
          {'id': 'a', 'folder': 'Work'},
          {'id': 'b', 'folder': ''},
          {'id': 'c', 'folder': 'Personal'},
          {'id': 'd', 'folder': 'Work'},
        ];

        final grouped = provider.sessionsByProject;
        final keys = grouped.keys.toList();

        expect(keys, ['Personal', 'Work', 'Allgemein']);
        expect(grouped['Work']!.length, 2);
        expect(grouped['Allgemein']!.length, 1);
        expect(grouped['Allgemein']!.first['id'], 'b');
      });

      test('empty sessions yields empty map', () {
        expect(provider.sessionsByProject, isEmpty);
      });
    });

    test('renameSession reloads list on success', () async {
      when(
        () => api.renameSession('s1', 'new-title'),
      ).thenAnswer((_) async => {});
      when(
        () => api.listSessions(),
      ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});

      await provider.renameSession('s1', 'new-title');

      verify(() => api.listSessions()).called(1);
    });

    test('moveToFolder reloads sessions and folders', () async {
      when(
        () => api.moveSessionToFolder('s1', 'Personal'),
      ).thenAnswer((_) async => {});
      when(
        () => api.listSessions(),
      ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});
      when(() => api.listFolders()).thenAnswer(
        (_) async => {
          'folders': <String>['Personal'],
        },
      );

      await provider.moveToFolder('s1', 'Personal');

      verify(() => api.listSessions()).called(1);
      verify(() => api.listFolders()).called(1);
      expect(provider.folders, ['Personal']);
    });

    test('autoSessionOnStartup creates new when shouldNew=true', () async {
      when(
        () => api.shouldNewSession(timeoutMinutes: 30),
      ).thenAnswer((_) async => true);
      when(
        () => api.createSession(),
      ).thenAnswer((_) async => {'session_id': 'fresh'});
      when(
        () => api.listSessions(),
      ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});

      final id = await provider.autoSessionOnStartup();

      expect(id, 'fresh');
    });

    test(
      'autoSessionOnStartup resumes most recent when shouldNew=false',
      () async {
        when(
          () => api.shouldNewSession(timeoutMinutes: 30),
        ).thenAnswer((_) async => false);
        when(() => api.listSessions()).thenAnswer(
          (_) async => {
            'sessions': <Map<String, dynamic>>[
              {'id': 'recent'},
            ],
          },
        );

        final id = await provider.autoSessionOnStartup();

        expect(id, 'recent');
        expect(provider.activeSessionId, 'recent');
      },
    );

    test('autoSessionOnStartup creates new when no sessions exist', () async {
      when(
        () => api.shouldNewSession(timeoutMinutes: 30),
      ).thenAnswer((_) async => false);
      when(
        () => api.listSessions(),
      ).thenAnswer((_) async => {'sessions': <Map<String, dynamic>>[]});
      when(
        () => api.createSession(),
      ).thenAnswer((_) async => {'session_id': 'new-fallback'});

      final id = await provider.autoSessionOnStartup();

      expect(id, 'new-fallback');
    });
  });
}
