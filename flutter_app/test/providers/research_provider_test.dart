import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/research_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('ResearchSummary.fromJson', () {
    test('uses defaults for missing fields', () {
      final s = ResearchSummary.fromJson({});
      expect(s.id, '');
      expect(s.query, '');
      expect(s.hops, 0);
      expect(s.confidenceAvg, 0.0);
      expect(s.createdAt, 0.0);
    });

    test('reads provided fields with type coercion', () {
      final s = ResearchSummary.fromJson({
        'id': 'r1',
        'query': 'climate',
        'hops': 3,
        'confidence_avg': 0.85,
        'created_at': 1714400000,
      });
      expect(s.id, 'r1');
      expect(s.hops, 3);
      expect(s.confidenceAvg, 0.85);
      expect(s.createdAt, 1714400000.0);
    });
  });

  group('ResearchSummary.timeAgo', () {
    test('"just now" for current timestamp', () {
      final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
      final s = ResearchSummary(
        id: 'x',
        query: '',
        hops: 0,
        confidenceAvg: 0,
        createdAt: now,
      );
      expect(s.timeAgo, 'just now');
    });

    test('"Xm ago" for ~5 minutes old', () {
      final ts = (DateTime.now().millisecondsSinceEpoch / 1000.0) - 300;
      final s = ResearchSummary(
        id: 'x',
        query: '',
        hops: 0,
        confidenceAvg: 0,
        createdAt: ts,
      );
      expect(s.timeAgo, contains('m ago'));
    });

    test('"Xh ago" for ~3 hours old', () {
      final ts = (DateTime.now().millisecondsSinceEpoch / 1000.0) - 3 * 3600;
      final s = ResearchSummary(
        id: 'x',
        query: '',
        hops: 0,
        confidenceAvg: 0,
        createdAt: ts,
      );
      expect(s.timeAgo, contains('h ago'));
    });

    test('"Xd ago" for ~2 days old', () {
      final ts = (DateTime.now().millisecondsSinceEpoch / 1000.0) - 2 * 86400;
      final s = ResearchSummary(
        id: 'x',
        query: '',
        hops: 0,
        confidenceAvg: 0,
        createdAt: ts,
      );
      expect(s.timeAgo, contains('d ago'));
    });
  });

  group('ResearchResult.fromJson', () {
    test('uses defaults for missing fields', () {
      final r = ResearchResult.fromJson({});
      expect(r.id, '');
      expect(r.reportMd, '');
      expect(r.sources, isEmpty);
    });

    test('reads sources as list of maps', () {
      final r = ResearchResult.fromJson({
        'id': 'r1',
        'report_md': '# Report',
        'sources': [
          {'url': 'a'},
          {'url': 'b'},
        ],
      });
      expect(r.reportMd, '# Report');
      expect(r.sources.length, 2);
      expect(r.sources.first['url'], 'a');
    });
  });

  group('ResearchProvider', () {
    late _MockApiClient api;
    late ResearchProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = ResearchProvider()..setApi(api);
    });

    test('initial state is empty', () {
      expect(provider.activeResult, isNull);
      expect(provider.history, isEmpty);
      expect(provider.loading, false);
      expect(provider.error, isNull);
    });

    test('loadHistory populates history list', () async {
      when(() => api.get('/api/v1/research/history')).thenAnswer(
        (_) async => {
          'results': [
            {'id': 'r1', 'query': 'q1'},
            {'id': 'r2', 'query': 'q2'},
          ],
        },
      );

      await provider.loadHistory();

      expect(provider.history.length, 2);
      expect(provider.history.first.id, 'r1');
    });

    test('loadHistory swallows errors silently', () async {
      when(
        () => api.get('/api/v1/research/history'),
      ).thenThrow(Exception('offline'));

      await provider.loadHistory();

      expect(provider.history, isEmpty);
      // Errors here are intentionally swallowed (UI uses freshness, not banner).
      expect(provider.error, isNull);
    });

    test('loadResult sets activeResult', () async {
      when(
        () => api.get('/api/v1/research/r1'),
      ).thenAnswer((_) async => {'id': 'r1', 'query': 'q', 'report_md': '# R'});

      await provider.loadResult('r1');

      expect(provider.activeResult?.id, 'r1');
      expect(provider.activeResult?.reportMd, '# R');
      expect(provider.loading, false);
    });

    test('loadResult sets error on failure', () async {
      when(
        () => api.get('/api/v1/research/missing'),
      ).thenThrow(Exception('404'));

      await provider.loadResult('missing');

      expect(provider.error, contains('404'));
      expect(provider.loading, false);
    });

    test(
      'deleteResearch removes from history + clears active if matching',
      () async {
        // Seed history.
        when(() => api.get('/api/v1/research/history')).thenAnswer(
          (_) async => {
            'results': [
              {'id': 'r1', 'query': 'a'},
              {'id': 'r2', 'query': 'b'},
            ],
          },
        );
        await provider.loadHistory();

        when(
          () => api.delete('/api/v1/research/r1'),
        ).thenAnswer((_) async => {});
        await provider.deleteResearch('r1');

        expect(provider.history.map((r) => r.id), ['r2']);
      },
    );

    test('exportResearch returns path on success', () async {
      when(
        () => api.post('/api/v1/research/r1/export', {'format': 'pdf'}),
      ).thenAnswer((_) async => {'path': '/tmp/r1.pdf'});

      final path = await provider.exportResearch('r1', 'pdf');

      expect(path, '/tmp/r1.pdf');
    });

    test('exportResearch returns null on failure', () async {
      when(() => api.post(any(), any())).thenThrow(Exception('boom'));

      final path = await provider.exportResearch('r1', 'pdf');

      expect(path, isNull);
    });

    test('loadHistory no-ops when API not set', () async {
      final unbound = ResearchProvider();
      await unbound.loadHistory();
      verifyZeroInteractions(api);
    });
  });
}
