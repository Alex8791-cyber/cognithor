import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('KanbanTask', () {
    test('fromJson parses full payload', () {
      final t = KanbanTask.fromJson({
        'id': 't1',
        'title': 'Title',
        'description': 'Desc',
        'status': 'in_progress',
        'priority': 'high',
        'assigned_agent': 'planner',
        'source': 'chat',
        'source_ref': 'msg42',
        'parent_id': 'parent-1',
        'labels': ['urgent', 'bug'],
        'sort_order': 7,
        'created_at': 't0',
        'updated_at': 't1',
        'completed_at': '',
        'created_by': 'user',
        'result_summary': 'ok',
      });

      expect(t.id, 't1');
      expect(t.title, 'Title');
      expect(t.status, 'in_progress');
      expect(t.priority, 'high');
      expect(t.assignedAgent, 'planner');
      expect(t.parentId, 'parent-1');
      expect(t.labels, ['urgent', 'bug']);
      expect(t.sortOrder, 7);
    });

    test('fromJson uses defaults for missing fields', () {
      final t = KanbanTask.fromJson(<String, dynamic>{});
      expect(t.id, '');
      expect(t.status, 'todo');
      expect(t.priority, 'medium');
      expect(t.source, 'manual');
      expect(t.createdBy, 'user');
      expect(t.subtasks, isEmpty);
    });

    test('fromJson recurses into subtasks', () {
      final t = KanbanTask.fromJson({
        'id': 'parent',
        'title': 'P',
        'subtasks': [
          {'id': 'child1', 'title': 'C1'},
          {'id': 'child2', 'title': 'C2'},
        ],
      });
      expect(t.subtasks.length, 2);
      expect(t.subtasks[0].id, 'child1');
    });

    test('statusDisplay maps known statuses', () {
      expect(
        KanbanTask(id: 'a', title: 't', status: 'todo').statusDisplay,
        'To Do',
      );
      expect(
        KanbanTask(id: 'a', title: 't', status: 'in_progress').statusDisplay,
        'In Progress',
      );
      expect(
        KanbanTask(id: 'a', title: 't', status: 'verifying').statusDisplay,
        'Verifying',
      );
      expect(
        KanbanTask(id: 'a', title: 't', status: 'cancelled').statusDisplay,
        'Cancelled',
      );
      expect(
        KanbanTask(id: 'a', title: 't', status: 'unknown').statusDisplay,
        'unknown',
      );
    });

    test('toJson returns minimal-create payload', () {
      final t = KanbanTask(
        id: 'x',
        title: 'New',
        description: 'd',
        priority: 'low',
        assignedAgent: 'a1',
        labels: ['l1'],
        parentId: 'p',
      );
      final json = t.toJson();
      expect(json['title'], 'New');
      expect(json['priority'], 'low');
      expect(json['assigned_agent'], 'a1');
      expect(json['labels'], ['l1']);
      expect(json['parent_id'], 'p');
    });
  });

  group('KanbanStats.fromJson', () {
    test('parses full payload', () {
      final s = KanbanStats.fromJson({
        'total': 10,
        'by_status': {'todo': 4, 'done': 6},
        'by_agent': {'planner': 3},
        'by_source': {'manual': 5, 'chat': 5},
      });
      expect(s.total, 10);
      expect(s.byStatus['todo'], 4);
      expect(s.byAgent['planner'], 3);
      expect(s.bySource['chat'], 5);
    });

    test('uses defaults for missing maps', () {
      final s = KanbanStats.fromJson(<String, dynamic>{});
      expect(s.total, 0);
      expect(s.byStatus, isEmpty);
    });
  });

  group('KanbanProvider', () {
    late _MockApiClient api;
    late KanbanProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = KanbanProvider()..setApiClient(api);
    });

    test('initial state', () {
      expect(provider.tasks, isEmpty);
      expect(provider.loading, false);
      expect(provider.error, isNull);
      expect(provider.pipelineMode, false);
      expect(provider.stats.total, 0);
    });

    test('togglePipelineMode flips and notifies', () {
      var notifyCount = 0;
      provider.addListener(() => notifyCount++);
      provider.togglePipelineMode();
      expect(provider.pipelineMode, true);
      provider.togglePipelineMode();
      expect(provider.pipelineMode, false);
      expect(notifyCount, 2);
    });

    group('tasksByStatus', () {
      test('groups into the 5 known columns', () {
        when(() => api.get('/kanban/tasks')).thenAnswer(
          (_) async => {
            'tasks': [
              {'id': 't1', 'title': 'a', 'status': 'todo'},
              {'id': 't2', 'title': 'b', 'status': 'in_progress'},
              {'id': 't3', 'title': 'c', 'status': 'done'},
              {'id': 't4', 'title': 'd', 'status': 'done'},
              {'id': 't5', 'title': 'e', 'status': 'blocked'},
            ],
          },
        );

        return provider.fetchTasks().then((_) {
          final grouped = provider.tasksByStatus;
          expect(grouped.keys, [
            'todo',
            'in_progress',
            'verifying',
            'done',
            'blocked',
          ]);
          expect(grouped['todo']!.length, 1);
          expect(grouped['done']!.length, 2);
          expect(grouped['verifying']!, isEmpty);
        });
      });
    });

    group('fetchTasks', () {
      test('populates from tasks key', () async {
        when(() => api.get('/kanban/tasks')).thenAnswer(
          (_) async => {
            'tasks': [
              {'id': 't1', 'title': 'a'},
              {'id': 't2', 'title': 'b'},
            ],
          },
        );

        await provider.fetchTasks();

        expect(provider.tasks.length, 2);
        expect(provider.error, isNull);
        expect(provider.loading, false);
      });

      test('falls back to items key', () async {
        when(() => api.get('/kanban/tasks')).thenAnswer(
          (_) async => {
            'items': [
              {'id': 't1', 'title': 'only-via-items'},
            ],
          },
        );

        await provider.fetchTasks();

        expect(provider.tasks.length, 1);
        expect(provider.tasks.first.id, 't1');
      });

      test('captures API error', () async {
        when(
          () => api.get('/kanban/tasks'),
        ).thenAnswer((_) async => {'error': 'down'});

        await provider.fetchTasks();

        expect(provider.error, contains('down'));
        expect(provider.tasks, isEmpty);
      });

      test('catches network exceptions', () async {
        when(() => api.get('/kanban/tasks')).thenThrow(Exception('boom'));

        await provider.fetchTasks();

        expect(provider.error, contains('boom'));
        expect(provider.loading, false);
      });

      test('builds query from filters', () async {
        when(
          () => api.get('/kanban/tasks?status=done&agent=planner'),
        ).thenAnswer((_) async => {'tasks': <Map<String, dynamic>>[]});

        await provider.fetchTasks(status: 'done', agent: 'planner');

        verify(
          () => api.get('/kanban/tasks?status=done&agent=planner'),
        ).called(1);
      });

      test('returns immediately when api not set', () async {
        final bare = KanbanProvider();
        await bare.fetchTasks();
        expect(bare.tasks, isEmpty);
        expect(bare.loading, false);
      });
    });

    test('fetchStats populates stats on success', () async {
      when(() => api.get('/kanban/stats')).thenAnswer(
        (_) async => {
          'total': 3,
          'by_status': {'todo': 3},
        },
      );

      await provider.fetchStats();

      expect(provider.stats.total, 3);
      expect(provider.stats.byStatus['todo'], 3);
    });

    group('createTask', () {
      test('inserts new task at front on success', () async {
        when(() => api.post('/kanban/tasks', any())).thenAnswer(
          (_) async => {'id': 'new', 'title': 'Created', 'status': 'todo'},
        );

        final t = await provider.createTask(title: 'Created');

        expect(t, isNotNull);
        expect(provider.tasks.first.id, 'new');
      });

      test('returns null on API error', () async {
        when(
          () => api.post('/kanban/tasks', any()),
        ).thenAnswer((_) async => {'error': 'denied'});

        final t = await provider.createTask(title: 'X');

        expect(t, isNull);
        expect(provider.tasks, isEmpty);
      });
    });

    group('moveTask', () {
      test('optimistically updates task status', () async {
        provider.tasks.add(KanbanTask(id: 't1', title: 'a', status: 'todo'));
        when(
          () => api.post('/kanban/tasks/t1/move', any()),
        ).thenAnswer((_) async => {'ok': true});
        when(
          () => api.get('/kanban/tasks'),
        ).thenAnswer((_) async => {'tasks': <Map<String, dynamic>>[]});

        final ok = await provider.moveTask('t1', 'in_progress');

        expect(ok, true);
      });

      test('returns false on API error and reloads', () async {
        provider.tasks.add(KanbanTask(id: 't1', title: 'a'));
        when(
          () => api.post('/kanban/tasks/t1/move', any()),
        ).thenAnswer((_) async => {'error': 'fail'});
        when(
          () => api.get('/kanban/tasks'),
        ).thenAnswer((_) async => {'tasks': <Map<String, dynamic>>[]});

        final ok = await provider.moveTask('t1', 'done');

        expect(ok, false);
        verify(() => api.get('/kanban/tasks')).called(1);
      });
    });

    test('updateTask returns true on success', () async {
      when(
        () => api.patch('/kanban/tasks/t1', any()),
      ).thenAnswer((_) async => {'ok': true});
      when(
        () => api.get('/kanban/tasks'),
      ).thenAnswer((_) async => {'tasks': <Map<String, dynamic>>[]});

      final ok = await provider.updateTask('t1', {'title': 'New'});
      expect(ok, true);
    });

    test('deleteTask removes task locally on success', () async {
      provider.tasks.add(KanbanTask(id: 't1', title: 'a'));
      when(
        () => api.delete('/kanban/tasks/t1'),
      ).thenAnswer((_) async => {'ok': true});

      final ok = await provider.deleteTask('t1');

      expect(ok, true);
      expect(provider.tasks, isEmpty);
    });

    test('getHistory returns list on success', () async {
      when(() => api.get('/kanban/tasks/t1/history')).thenAnswer(
        (_) async => {
          'history': [
            {'event': 'created'},
          ],
        },
      );

      final hist = await provider.getHistory('t1');

      expect(hist.length, 1);
      expect(hist.first['event'], 'created');
    });

    test('getHistory returns empty list on error', () async {
      when(
        () => api.get('/kanban/tasks/t1/history'),
      ).thenAnswer((_) async => {'error': 'gone'});

      final hist = await provider.getHistory('t1');

      expect(hist, isEmpty);
    });

    group('onKanbanUpdate', () {
      test('inserts task on created action', () {
        provider.onKanbanUpdate({
          'action': 'created',
          'task': {'id': 'new', 'title': 'X', 'status': 'todo'},
        });
        expect(provider.tasks.first.id, 'new');
      });

      test('removes task on deleted action', () {
        provider.tasks.add(KanbanTask(id: 't1', title: 'a'));
        provider.tasks.add(KanbanTask(id: 't2', title: 'b'));

        provider.onKanbanUpdate({'action': 'deleted', 'task_id': 't1'});

        expect(provider.tasks.length, 1);
        expect(provider.tasks.first.id, 't2');
      });

      test('updated/moved triggers fetch', () async {
        when(
          () => api.get('/kanban/tasks'),
        ).thenAnswer((_) async => {'tasks': <Map<String, dynamic>>[]});

        provider.onKanbanUpdate({'action': 'updated'});
        await Future<void>.delayed(Duration.zero);
        verify(() => api.get('/kanban/tasks')).called(1);
      });
    });
  });
}
