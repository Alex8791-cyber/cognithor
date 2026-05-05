import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void _stubGetters(_MockApiClient api) {
  when(
    () => api.getRedditLeads(
      status: any(named: 'status'),
      minScore: any(named: 'minScore'),
    ),
  ).thenAnswer((_) async => {'leads': <Map<String, dynamic>>[]});
  when(
    () => api.getRedditLeadStats(),
  ).thenAnswer((_) async => {'stats': <String, dynamic>{}});
}

void main() {
  group('RedditLead.fromJson', () {
    test('parses full payload', () {
      final lead = RedditLead.fromJson({
        'id': 'lead-1',
        'post_id': 'p1',
        'subreddit': 'flutter',
        'title': 'Need help',
        'body': 'How does X work?',
        'url': 'https://reddit.com/r/flutter/p1',
        'author': 'alice',
        'intent_score': 87,
        'score_reason': 'high intent',
        'reply_draft': 'draft',
        'reply_final': 'final',
        'status': 'reviewed',
        'upvotes': 12,
        'num_comments': 3,
        'detected_at': 1.0,
      });

      expect(lead.id, 'lead-1');
      expect(lead.postId, 'p1');
      expect(lead.subreddit, 'flutter');
      expect(lead.intentScore, 87);
      expect(lead.upvotes, 12);
      expect(lead.numComments, 3);
      expect(lead.status, 'reviewed');
    });

    test('falls back gracefully on missing fields', () {
      final lead = RedditLead.fromJson(<String, dynamic>{});
      expect(lead.id, '');
      expect(lead.intentScore, 0);
      expect(lead.upvotes, 0);
      expect(lead.status, 'new');
    });

    test('effectiveReply prefers replyFinal when set', () {
      final lead = RedditLead.fromJson({
        'reply_draft': 'draft text',
        'reply_final': 'final text',
      });
      expect(lead.effectiveReply, 'final text');
    });

    test('effectiveReply falls back to replyDraft when final is empty', () {
      final lead = RedditLead.fromJson({
        'reply_draft': 'draft text',
        'reply_final': '',
      });
      expect(lead.effectiveReply, 'draft text');
    });

    test('timeAgo handles "just now" boundary', () {
      final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
      final lead = RedditLead.fromJson({'detected_at': now});
      expect(lead.timeAgo, 'just now');
    });
  });

  group('RedditLeadsProvider', () {
    late _MockApiClient api;
    late RedditLeadsProvider provider;

    setUp(() {
      api = _MockApiClient();
      _stubGetters(api);
      provider = RedditLeadsProvider();
    });

    tearDown(() {
      provider.dispose();
    });

    test('initial state', () {
      final fresh = RedditLeadsProvider();
      expect(fresh.leads, isEmpty);
      expect(fresh.stats, isEmpty);
      expect(fresh.loading, isFalse);
      expect(fresh.scanning, isFalse);
      expect(fresh.error, isNull);
      expect(fresh.statusFilter, '');
      expect(fresh.minScoreFilter, 0);
      expect(fresh.newCount, 0);
      expect(fresh.reviewedCount, 0);
      expect(fresh.repliedCount, 0);
      fresh.dispose();
    });

    test('without API, fetchLeads is a no-op', () async {
      await provider.fetchLeads();
      expect(provider.leads, isEmpty);
      expect(provider.loading, isFalse);
    });

    test('without API, scanNow returns false', () async {
      final ok = await provider.scanNow();
      expect(ok, isFalse);
    });

    test('init populates leads + stats from API', () async {
      when(
        () => api.getRedditLeads(
          status: any(named: 'status'),
          minScore: any(named: 'minScore'),
        ),
      ).thenAnswer(
        (_) async => {
          'leads': [
            {'id': 'a', 'status': 'new', 'intent_score': 50},
            {'id': 'b', 'status': 'replied', 'intent_score': 90},
          ],
        },
      );
      when(() => api.getRedditLeadStats()).thenAnswer(
        (_) async => {
          'stats': {'total': 2},
        },
      );

      provider.init(api);
      // ChangeNotifier flushes the await tail of fetchLeads after a
      // microtask; let everything settle.
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.leads.length, 2);
      expect(provider.newCount, 1);
      expect(provider.repliedCount, 1);
      expect(provider.stats['total'], 2);
    });

    test('fetchLeads surfaces API error in error field', () async {
      when(
        () => api.getRedditLeads(
          status: any(named: 'status'),
          minScore: any(named: 'minScore'),
        ),
      ).thenAnswer((_) async => {'error': 'rate limited'});

      provider.init(api);
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.error, 'rate limited');
      expect(provider.loading, isFalse);
    });

    test('setStatusFilter updates and triggers fetch', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      provider.setStatusFilter('replied');
      expect(provider.statusFilter, 'replied');
    });

    test('setMinScoreFilter updates and triggers fetch', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      provider.setMinScoreFilter(50);
      expect(provider.minScoreFilter, 50);
    });

    test('scanNow returns true and reloads on success', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      when(() => api.scanRedditLeads()).thenAnswer((_) async => {'ok': true});

      final ok = await provider.scanNow();
      expect(ok, isTrue);
      expect(provider.scanning, isFalse);
    });

    test('scanNow returns false and stores error from server', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      when(
        () => api.scanRedditLeads(),
      ).thenAnswer((_) async => {'error': 'busy'});

      final ok = await provider.scanNow();
      expect(ok, isFalse);
      expect(provider.error, 'busy');
      expect(provider.scanning, isFalse);
    });

    test('updateLead reloads on success', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      when(
        () => api.updateRedditLead(any(), any()),
      ).thenAnswer((_) async => {'ok': true});

      final ok = await provider.updateLead('lead-1', status: 'replied');
      expect(ok, isTrue);
      verify(
        () => api.updateRedditLead('lead-1', {'status': 'replied'}),
      ).called(1);
    });

    test('updateLead returns false when API returns error', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      when(
        () => api.updateRedditLead(any(), any()),
      ).thenAnswer((_) async => {'error': 'forbidden'});

      final ok = await provider.updateLead('lead-1', status: 'replied');
      expect(ok, isFalse);
    });

    test('getPerformance caches by id', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      when(
        () => api.getRedditLeadPerformance('lead-1'),
      ).thenAnswer((_) async => {'views': 100});

      final first = await provider.getPerformance('lead-1');
      final second = await provider.getPerformance('lead-1');

      expect(first, {'views': 100});
      expect(second, {'views': 100});
      verify(() => api.getRedditLeadPerformance('lead-1')).called(1);
    });

    test('preloadPerformance dedupes calls per id', () async {
      provider.init(api);
      await Future<void>.delayed(Duration.zero);

      when(
        () => api.getRedditLeadPerformance('lead-1'),
      ).thenAnswer((_) async => {'views': 5});

      await provider.preloadPerformance('lead-1');
      await provider.preloadPerformance('lead-1');

      verify(() => api.getRedditLeadPerformance('lead-1')).called(1);
      expect(provider.getCachedPerformance('lead-1'), {'views': 5});
    });
  });
}
