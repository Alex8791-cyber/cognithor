import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/models/chat_node.dart';
import 'package:cognithor_ui/providers/tree_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

ChatNode _node({required String id, String? parent, int branch = 0}) =>
    ChatNode(
      id: id,
      conversationId: 'c1',
      parentId: parent,
      role: 'user',
      text: id,
      branchIndex: branch,
    );

void main() {
  group('TreeProvider', () {
    late TreeProvider provider;
    late _MockApiClient api;

    setUp(() {
      api = _MockApiClient();
      provider = TreeProvider()..setApi(api);
    });

    test('initial state has no tree, no branches', () {
      expect(provider.hasTree, false);
      expect(provider.hasBranches, false);
      expect(provider.nodes, isEmpty);
      expect(provider.activePath, isEmpty);
    });

    test('addNode appends to activePath + tracks fork when sibling exists', () {
      provider.addNode(_node(id: 'root'));
      provider.addNode(_node(id: 'a', parent: 'root', branch: 0));
      provider.addNode(_node(id: 'b', parent: 'root', branch: 1));

      expect(provider.hasTree, true);
      expect(provider.activePath, ['root', 'a', 'b']);
      // 'root' now has 2 children → fork point.
      expect(provider.isForkPoint('root'), true);
      expect(provider.getChildCount('root'), 2);
    });

    test('isForkPoint returns false for single-child node', () {
      provider.addNode(_node(id: 'root'));
      provider.addNode(_node(id: 'a', parent: 'root'));
      expect(provider.isForkPoint('root'), false);
    });

    test('clear resets all state + notifies listeners', () {
      provider.addNode(_node(id: 'root'));
      provider.addNode(_node(id: 'a', parent: 'root'));
      provider.addNode(_node(id: 'b', parent: 'root', branch: 1));

      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      provider.clear();

      expect(provider.nodes, isEmpty);
      expect(provider.activePath, isEmpty);
      expect(provider.hasBranches, false);
      expect(provider.conversationId, isNull);
      expect(notifyCount, greaterThanOrEqualTo(1));
    });

    test('loadTree builds nodes + activePath from API response', () async {
      when(() => api.get('chat/tree/c1')).thenAnswer(
        (_) async => {
          'nodes': [
            {
              'id': 'root',
              'conversation_id': 'c1',
              'parent_id': null,
              'role': 'user',
              'text': 'hi',
              'branch_index': 0,
            },
            {
              'id': 'a',
              'conversation_id': 'c1',
              'parent_id': 'root',
              'role': 'assistant',
              'text': 'hello',
              'branch_index': 0,
            },
          ],
          'fork_points': <String, dynamic>{},
          'active_leaf_id': 'a',
        },
      );

      await provider.loadTree('c1');

      expect(provider.nodes.length, 2);
      expect(provider.activePath, ['root', 'a']);
      expect(provider.conversationId, 'c1');
    });

    test('loadTree records fork_points', () async {
      when(() => api.get('chat/tree/c1')).thenAnswer(
        (_) async => {
          'nodes': [
            {
              'id': 'root',
              'conversation_id': 'c1',
              'role': 'user',
              'text': 'q',
            },
            {
              'id': 'a',
              'conversation_id': 'c1',
              'parent_id': 'root',
              'role': 'assistant',
              'text': 'r1',
              'branch_index': 0,
            },
            {
              'id': 'b',
              'conversation_id': 'c1',
              'parent_id': 'root',
              'role': 'assistant',
              'text': 'r2',
              'branch_index': 1,
            },
          ],
          'fork_points': {'root': 2},
          'active_leaf_id': 'a',
        },
      );

      await provider.loadTree('c1');

      expect(provider.hasBranches, true);
      expect(provider.getChildCount('root'), 2);
      expect(provider.getActiveChildIndex('root'), 0);
    });

    test('switchBranch sends ws message + updates activePath', () async {
      // Seed a tree with two branches off `root`.
      when(() => api.get('chat/tree/c1')).thenAnswer(
        (_) async => {
          'nodes': [
            {
              'id': 'root',
              'conversation_id': 'c1',
              'role': 'user',
              'text': 'q',
            },
            {
              'id': 'a',
              'conversation_id': 'c1',
              'parent_id': 'root',
              'role': 'assistant',
              'text': 'r1',
              'branch_index': 0,
            },
            {
              'id': 'b',
              'conversation_id': 'c1',
              'parent_id': 'root',
              'role': 'assistant',
              'text': 'r2',
              'branch_index': 1,
            },
          ],
          'fork_points': {'root': 2},
          'active_leaf_id': 'a',
        },
      );
      await provider.loadTree('c1');

      Map<String, dynamic>? sent;
      provider.setWsSend((msg) => sent = msg);

      await provider.switchBranch('root', 1);

      expect(provider.activePath, ['root', 'b']);
      expect(provider.getActiveChildIndex('root'), 1);
      expect(sent, isNotNull);
      expect(sent!['type'], 'branch_switch');
      expect(sent!['leaf_id'], 'b');
    });

    test('switchBranch is a no-op when index is out of range', () async {
      when(() => api.get('chat/tree/c1')).thenAnswer(
        (_) async => {
          'nodes': [
            {
              'id': 'root',
              'conversation_id': 'c1',
              'role': 'user',
              'text': 'q',
            },
            {
              'id': 'a',
              'conversation_id': 'c1',
              'parent_id': 'root',
              'role': 'assistant',
              'text': 'r',
              'branch_index': 0,
            },
          ],
          'fork_points': {'root': 1},
          'active_leaf_id': 'a',
        },
      );
      await provider.loadTree('c1');

      Map<String, dynamic>? sent;
      provider.setWsSend((msg) => sent = msg);

      await provider.switchBranch('root', 99);

      expect(sent, isNull);
    });

    test('refreshFromSession with sessionId uses query parameter', () async {
      when(
        () => api.get('chat/tree/latest?session_id=s1'),
      ).thenAnswer((_) async => {'conversation_id': 'cX'});
      when(() => api.get('chat/tree/cX')).thenAnswer(
        (_) async => <String, dynamic>{
          'nodes': <Map<String, dynamic>>[],
          'fork_points': <String, dynamic>{},
        },
      );

      await provider.refreshFromSession(api, sessionId: 's1');

      expect(provider.conversationId, 'cX');
    });
  });
}
