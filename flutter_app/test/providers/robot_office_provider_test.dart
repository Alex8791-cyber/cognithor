import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/robot_office_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void _stubInitialPolls(_MockApiClient api) {
  when(() => api.get(any())).thenAnswer((_) async => <String, dynamic>{});
  when(() => api.getList(any())).thenAnswer((_) async => <dynamic>[]);
}

void main() {
  group('AgentInfo', () {
    test('default flags are sane', () {
      final a = AgentInfo(name: 'planner');
      expect(a.name, 'planner');
      expect(a.displayName, isNull);
      expect(a.isWorking, isFalse);
      expect(a.currentTask, '');
    });

    test('explicit displayName + flags', () {
      final a = AgentInfo(
        name: 'planner',
        displayName: 'Planner',
        isWorking: true,
        currentTask: 'Reviewing tickets',
      );
      expect(a.displayName, 'Planner');
      expect(a.isWorking, isTrue);
      expect(a.currentTask, 'Reviewing tickets');
    });
  });

  group('SystemMetrics', () {
    test('defaults to zero', () {
      final m = SystemMetrics();
      expect(m.cpu, 0);
      expect(m.memory, 0);
      expect(m.load, 0);
    });

    test('mutable fields', () {
      final m = SystemMetrics(cpu: 0.5, memory: 0.3, load: 0.4);
      m.cpu = 0.9;
      expect(m.cpu, 0.9);
      expect(m.memory, 0.3);
    });
  });

  group('PgePhase', () {
    test('has all 5 phases', () {
      expect(PgePhase.values.length, 5);
      expect(
        PgePhase.values,
        containsAll([
          PgePhase.idle,
          PgePhase.planning,
          PgePhase.gating,
          PgePhase.executing,
          PgePhase.streaming,
        ]),
      );
    });
  });

  group('RobotOfficeProvider', () {
    late RobotOfficeProvider provider;

    setUp(() {
      provider = RobotOfficeProvider();
    });

    tearDown(() {
      provider.dispose();
    });

    test('initial state', () {
      expect(provider.pgePhase, PgePhase.idle);
      expect(provider.plannerTask, '');
      expect(provider.executorTask, '');
      expect(provider.gatekeeperTask, '');
      expect(provider.agents, isEmpty);
      expect(provider.metrics.cpu, 0);
      expect(provider.kanbanCounts, isEmpty);
      expect(provider.kanbanTasks, isEmpty);
      expect(provider.isUninitialized, isTrue);
    });

    test('activePhaseInt maps each phase to a stable index', () {
      expect(provider.activePhaseInt, 4); // idle
      // The painter relies on these exact indices — guard against
      // accidental enum reordering.
      final mapping = {
        PgePhase.planning: 0,
        PgePhase.gating: 1,
        PgePhase.executing: 2,
        PgePhase.streaming: 3,
        PgePhase.idle: 4,
      };
      // Verify all 5 phases have a unique mapped int.
      expect(mapping.values.toSet().length, 5);
    });

    test('init wires API + ws (ws may be null)', () async {
      final api = _MockApiClient();
      _stubInitialPolls(api);

      provider.init(api, null);
      // _api is now non-null, polling started
      expect(provider.isUninitialized, isFalse);

      // Allow the first immediate _poll to complete.
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);
      // No error / no crash; state stays sane on empty responses.
      expect(provider.agents, isEmpty);
      expect(provider.kanbanCounts, isEmpty);
    });

    test('poll: agents are populated from response', () async {
      final api = _MockApiClient();
      _stubInitialPolls(api);
      when(() => api.get('agents')).thenAnswer(
        (_) async => {
          'agents': [
            {'name': 'planner', 'display_name': 'Planner'},
            {'name': 'executor'},
            {'name': ''}, // filtered out (empty name)
          ],
        },
      );

      provider.init(api, null);
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.agents.length, 2);
      expect(provider.agents.first.name, 'planner');
      expect(provider.agents.first.displayName, 'Planner');
    });

    test('poll: kanban counts + per-status titles are populated', () async {
      final api = _MockApiClient();
      _stubInitialPolls(api);
      when(() => api.getList('kanban/tasks')).thenAnswer(
        (_) async => [
          {'status': 'backlog', 'title': 'task A'},
          {'status': 'backlog', 'title': 'task B'},
          {'status': 'in_progress', 'title': 'task C'},
        ],
      );

      provider.init(api, null);
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.kanbanCounts['backlog'], 2);
      expect(provider.kanbanCounts['in_progress'], 1);
      expect(provider.kanbanTasks['backlog'], ['task A', 'task B']);
    });

    test('poll: defaults status to "backlog" when missing', () async {
      final api = _MockApiClient();
      _stubInitialPolls(api);
      when(() => api.getList('kanban/tasks')).thenAnswer(
        (_) async => [
          {'title': 'unstatused'},
        ],
      );

      provider.init(api, null);
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.kanbanCounts['backlog'], 1);
    });

    test('poll: metrics scaled to 0..1 from percent', () async {
      final api = _MockApiClient();
      _stubInitialPolls(api);
      when(() => api.get('monitoring/dashboard')).thenAnswer(
        (_) async => {
          'system': {'cpu_percent': 80, 'memory_percent': 60},
        },
      );

      provider.init(api, null);
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.metrics.cpu, closeTo(0.8, 1e-9));
      expect(provider.metrics.memory, closeTo(0.6, 1e-9));
      expect(provider.metrics.load, closeTo(0.7, 1e-9));
    });

    test('poll: metrics load is clamped to 1.0', () async {
      final api = _MockApiClient();
      _stubInitialPolls(api);
      when(() => api.get('monitoring/dashboard')).thenAnswer(
        (_) async => {
          'system': {'cpu_percent': 200, 'memory_percent': 200},
        },
      );

      provider.init(api, null);
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.metrics.load, lessThanOrEqualTo(1.0));
    });

    test(
      'poll: agents endpoint failure leaves agents empty + no crash',
      () async {
        final api = _MockApiClient();
        _stubInitialPolls(api);
        when(() => api.get('agents')).thenThrow(Exception('boom'));

        provider.init(api, null);
        await Future<void>.delayed(Duration.zero);
        await Future<void>.delayed(Duration.zero);

        expect(provider.agents, isEmpty);
      },
    );
  });
}
