import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/cron_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('CronJob.scheduleLabel', () {
    test('weekday schedule formatted', () {
      final job = CronJob(name: 'n', schedule: '30 9 * * 1-5', prompt: '');
      expect(job.scheduleLabel, 'Weekdays 9:30');
    });

    test('friday schedule formatted', () {
      final job = CronJob(name: 'n', schedule: '0 17 * * 5', prompt: '');
      expect(job.scheduleLabel, 'Fridays 17:00');
    });

    test('monthly first-of-month schedule formatted', () {
      final job = CronJob(name: 'n', schedule: '0 8 1 * *', prompt: '');
      expect(job.scheduleLabel, 'Monthly (1st) 8:00');
    });

    test('daily schedule formatted', () {
      final job = CronJob(name: 'n', schedule: '15 6 * * *', prompt: '');
      expect(job.scheduleLabel, 'Daily 6:15');
    });

    test('non-matching dow returned verbatim', () {
      // dow '2' (Tuesday only) has no special-case branch.
      final job = CronJob(name: 'n', schedule: '0 9 * * 2', prompt: '');
      expect(job.scheduleLabel, '0 9 * * 2');
    });

    test('malformed (too few parts) returned verbatim', () {
      final job = CronJob(name: 'n', schedule: '0 0', prompt: '');
      expect(job.scheduleLabel, '0 0');
    });
  });

  group('CronJob.fromJson', () {
    test('uses defaults for missing fields', () {
      final job = CronJob.fromJson({});
      expect(job.name, '');
      expect(job.channel, 'telegram');
      expect(job.model, 'qwen3:8b');
      expect(job.enabled, false);
      expect(job.nextRun, isNull);
    });

    test('reads provided fields', () {
      final job = CronJob.fromJson({
        'name': 'morning-news',
        'schedule': '0 7 * * *',
        'prompt': 'summarize news',
        'channel': 'discord',
        'model': 'qwen3:14b',
        'enabled': true,
        'agent': 'newsbot',
        'next_run': '2026-04-30T07:00:00Z',
      });
      expect(job.name, 'morning-news');
      expect(job.channel, 'discord');
      expect(job.enabled, true);
      expect(job.agent, 'newsbot');
      expect(job.nextRun, '2026-04-30T07:00:00Z');
    });
  });

  group('CronProvider', () {
    late _MockApiClient api;
    late CronProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = CronProvider();
    });

    tearDown(() => provider.dispose());

    test('initial state is empty + not loading', () {
      expect(provider.jobs, isEmpty);
      expect(provider.loading, false);
      expect(provider.error, isNull);
    });

    test('setApiClient triggers initial fetch', () async {
      when(() => api.get('/cron-jobs/enriched')).thenAnswer(
        (_) async => {
          'jobs': [
            {'name': 'a', 'schedule': '0 8 * * *', 'prompt': 'p'},
          ],
        },
      );

      provider.setApiClient(api);
      // Allow the awaited fetch in the timer-callback path to settle.
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.jobs.length, 1);
      expect(provider.jobs.first.name, 'a');
    });

    test('fetchJobs sets error on failure', () async {
      when(
        () => api.get('/cron-jobs/enriched'),
      ).thenThrow(Exception('cron-down'));
      provider.setApiClient(api);

      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.error, contains('cron-down'));
      expect(provider.loading, false);
    });
  });
}
