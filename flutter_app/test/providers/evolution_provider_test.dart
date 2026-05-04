import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/evolution_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('EvolutionGoal.fromJson', () {
    test('parses full payload', () {
      final goal = EvolutionGoal.fromJson({
        'id': 'g1',
        'title': 'Title',
        'description': 'Desc',
        'status': 'paused',
        'progress': 0.42,
        'priority': 1,
        'tags': ['alpha', 'beta'],
      });

      expect(goal.id, 'g1');
      expect(goal.title, 'Title');
      expect(goal.status, 'paused');
      expect(goal.progress, closeTo(0.42, 1e-9));
      expect(goal.priority, 1);
      expect(goal.tags, ['alpha', 'beta']);
    });

    test('uses defaults for missing fields', () {
      final goal = EvolutionGoal.fromJson(<String, dynamic>{});

      expect(goal.id, '');
      expect(goal.title, '');
      expect(goal.status, 'active');
      expect(goal.progress, 0.0);
      expect(goal.priority, 3);
      expect(goal.tags, isEmpty);
    });
  });

  group('EvolutionPlan.fromJson + completionPercent', () {
    test('parses full payload', () {
      final plan = EvolutionPlan.fromJson({
        'id': 'p1',
        'goal': 'do thing',
        'status': 'running',
        'sub_goals_total': 4,
        'sub_goals_passed': 2,
        'coverage_score': 0.5,
        'quality_score': 0.7,
        'cycle_state': 'active',
      });

      expect(plan.id, 'p1');
      expect(plan.goal, 'do thing');
      expect(plan.status, 'running');
      expect(plan.subGoalsTotal, 4);
      expect(plan.subGoalsPassed, 2);
      expect(plan.completionPercent, 0.5);
      expect(plan.cycleState, 'active');
    });

    test('completionPercent is 0 when no sub-goals', () {
      final plan = EvolutionPlan.fromJson(<String, dynamic>{});
      expect(plan.completionPercent, 0.0);
      expect(plan.cycleState, 'unknown');
    });
  });

  group('EvolutionProvider', () {
    late _MockApiClient api;
    late EvolutionProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = EvolutionProvider()..setApi(api);
    });

    test('initial state is empty', () {
      expect(provider.goals, isEmpty);
      expect(provider.plans, isEmpty);
      expect(provider.journal, '');
      expect(provider.stats, isEmpty);
      expect(provider.loading, false);
    });

    group('fetchGoals', () {
      test('populates goals list', () async {
        when(() => api.get('evolution/goals')).thenAnswer(
          (_) async => {
            'goals': <Map<String, dynamic>>[
              {'id': 'g1', 'title': 'A'},
              {'id': 'g2', 'title': 'B'},
            ],
          },
        );

        await provider.fetchGoals();

        expect(provider.goals.length, 2);
        expect(provider.goals.first.id, 'g1');
        expect(provider.loading, false);
      });

      test('treats missing goals key as empty list', () async {
        when(
          () => api.get('evolution/goals'),
        ).thenAnswer((_) async => <String, dynamic>{});

        await provider.fetchGoals();

        expect(provider.goals, isEmpty);
      });

      test('keeps state on API error response', () async {
        provider.goals.add(EvolutionGoal(id: 'pre', title: 'pre'));
        when(
          () => api.get('evolution/goals'),
        ).thenAnswer((_) async => {'error': 'down'});

        await provider.fetchGoals();

        // Error responses skip the assignment, keeping previous goals.
        expect(provider.goals.first.id, 'pre');
        expect(provider.loading, false);
      });

      test('catches thrown exceptions and clears loading', () async {
        when(() => api.get('evolution/goals')).thenThrow(Exception('boom'));

        await provider.fetchGoals();

        expect(provider.loading, false);
      });

      test('returns immediately when api not set', () async {
        final bare = EvolutionProvider();
        await bare.fetchGoals();
        expect(bare.goals, isEmpty);
        expect(bare.loading, false);
      });
    });

    group('fetchPlans', () {
      test('populates plans list', () async {
        when(() => api.get('evolution/plans')).thenAnswer(
          (_) async => {
            'plans': <Map<String, dynamic>>[
              {'id': 'p1', 'goal': 'do x'},
            ],
          },
        );

        await provider.fetchPlans();

        expect(provider.plans.length, 1);
        expect(provider.plans.first.id, 'p1');
      });

      test('handles missing plans key', () async {
        when(
          () => api.get('evolution/plans'),
        ).thenAnswer((_) async => <String, dynamic>{});

        await provider.fetchPlans();

        expect(provider.plans, isEmpty);
      });
    });

    group('fetchJournal', () {
      test('uses default 7 days', () async {
        when(
          () => api.get('evolution/journal?days=7'),
        ).thenAnswer((_) async => {'content': 'entry'});

        await provider.fetchJournal();

        expect(provider.journal, 'entry');
      });

      test('respects custom days param', () async {
        when(
          () => api.get('evolution/journal?days=30'),
        ).thenAnswer((_) async => {'content': 'long entry'});

        await provider.fetchJournal(days: 30);

        expect(provider.journal, 'long entry');
      });

      test('skips assignment on API error', () async {
        provider.fetchJournal();
        when(
          () => api.get('evolution/journal?days=7'),
        ).thenAnswer((_) async => {'error': 'gone'});

        await provider.fetchJournal();
        expect(provider.journal, '');
      });
    });

    group('fetchStats', () {
      test('captures full stats payload', () async {
        when(
          () => api.get('evolution/stats'),
        ).thenAnswer((_) async => {'goals': 5, 'plans': 2});

        await provider.fetchStats();

        expect(provider.stats['goals'], 5);
        expect(provider.stats['plans'], 2);
      });
    });

    group('createGoal', () {
      test('returns true and reloads on success', () async {
        when(
          () => api.post('evolution/goals', any()),
        ).thenAnswer((_) async => {'id': 'new'});
        when(
          () => api.get('evolution/goals'),
        ).thenAnswer((_) async => {'goals': <Map<String, dynamic>>[]});

        final ok = await provider.createGoal(title: 'New');

        expect(ok, true);
        verify(() => api.get('evolution/goals')).called(1);
      });

      test('returns false on API error', () async {
        when(
          () => api.post('evolution/goals', any()),
        ).thenAnswer((_) async => {'error': 'invalid'});

        final ok = await provider.createGoal(title: 'X');

        expect(ok, false);
      });

      test('returns false when api not set', () async {
        final bare = EvolutionProvider();
        final ok = await bare.createGoal(title: 'X');
        expect(ok, false);
      });
    });

    group('updateGoal', () {
      test('only includes provided fields', () async {
        when(
          () => api.patch('evolution/goals/g1', any()),
        ).thenAnswer((_) async => {});
        when(
          () => api.get('evolution/goals'),
        ).thenAnswer((_) async => {'goals': <Map<String, dynamic>>[]});

        final ok = await provider.updateGoal(
          'g1',
          title: 'Updated',
          priority: 5,
        );

        expect(ok, true);
        final captured =
            verify(
                  () => api.patch('evolution/goals/g1', captureAny()),
                ).captured.single
                as Map<String, dynamic>;
        expect(captured.keys, containsAll(['title', 'priority']));
        expect(captured.containsKey('description'), false);
        expect(captured.containsKey('status'), false);
      });

      test('returns false on error', () async {
        when(
          () => api.patch('evolution/goals/g1', any()),
        ).thenAnswer((_) async => {'error': 'boom'});

        final ok = await provider.updateGoal('g1', title: 'X');

        expect(ok, false);
      });
    });

    group('deleteGoal', () {
      test('returns true and reloads on success', () async {
        when(
          () => api.delete('evolution/goals/g1'),
        ).thenAnswer((_) async => {});
        when(
          () => api.get('evolution/goals'),
        ).thenAnswer((_) async => {'goals': <Map<String, dynamic>>[]});

        final ok = await provider.deleteGoal('g1');

        expect(ok, true);
      });

      test('returns false on API error', () async {
        when(
          () => api.delete('evolution/goals/g1'),
        ).thenAnswer((_) async => {'error': 'denied'});

        final ok = await provider.deleteGoal('g1');

        expect(ok, false);
      });
    });

    test('fetchAll calls every fetch method in parallel', () async {
      when(
        () => api.get('evolution/goals'),
      ).thenAnswer((_) async => {'goals': <Map<String, dynamic>>[]});
      when(
        () => api.get('evolution/plans'),
      ).thenAnswer((_) async => {'plans': <Map<String, dynamic>>[]});
      when(
        () => api.get('evolution/journal?days=7'),
      ).thenAnswer((_) async => {'content': ''});
      when(
        () => api.get('evolution/stats'),
      ).thenAnswer((_) async => <String, dynamic>{});

      await provider.fetchAll();

      verify(() => api.get('evolution/goals')).called(1);
      verify(() => api.get('evolution/plans')).called(1);
      verify(() => api.get('evolution/journal?days=7')).called(1);
      verify(() => api.get('evolution/stats')).called(1);
    });
  });
}
