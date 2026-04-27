/// Conversation tree state management.
///
/// Manages nodes, active path, fork points, and branch switching.
/// Integrates with ChatProvider for message rendering and
/// WebSocketService for branch_switch commands.
library;

import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/models/chat_node.dart';
import 'package:cognithor_ui/services/api_client.dart';

/// Callback type for sending WebSocket messages.
typedef WsSendCallback = void Function(Map<String, dynamic> message);

class TreeProvider extends ChangeNotifier {
  ApiClient? _api;
  WsSendCallback? _wsSend;

  String? conversationId;
  final Map<String, ChatNode> nodes = {};
  List<String> activePath = []; // Node IDs from root to current leaf
  Map<String, int> forkPoints = {}; // nodeId -> childCount
  Map<String, int> activeChildAtFork =
      {}; // nodeId -> which child is active (0-based)

  bool get hasTree => nodes.isNotEmpty;
  bool get hasBranches => forkPoints.isNotEmpty;

  void setApi(ApiClient api) => _api = api;
  void setWsSend(WsSendCallback wsSend) => _wsSend = wsSend;

  /// Load full tree from backend for a conversation.
  Future<void> loadTree(String convId) async {
    if (_api == null) return;
    conversationId = convId;
    try {
      final data = await _api!.get('chat/tree/$convId');
      nodes.clear();
      forkPoints.clear();

      final nodeList = data['nodes'] as List? ?? [];
      for (final n in nodeList) {
        final node = ChatNode.fromJson(n as Map<String, dynamic>);
        nodes[node.id] = node;
      }

      // Fork points
      final fps = data['fork_points'] as Map<String, dynamic>? ?? {};
      forkPoints = fps.map((k, v) => MapEntry(k, v as int));

      // Active path
      final leafId = data['active_leaf_id'] as String?;
      if (leafId != null) {
        _computeActivePath(leafId);
      }

      // Compute active child indices at each fork
      _computeActiveChildIndices();

      notifyListeners();
    } catch (e) {
      debugPrint('[Tree] loadTree failed: $e');
    }
  }

  /// Add a node locally (called when user sends message or assistant responds).
  void addNode(ChatNode node) {
    nodes[node.id] = node;
    activePath.add(node.id);

    // Check if parent now has multiple children (new fork point)
    if (node.parentId != null) {
      final siblings = nodes.values
          .where((n) => n.parentId == node.parentId)
          .length;
      if (siblings > 1) {
        forkPoints[node.parentId!] = siblings;
      }
    }

    notifyListeners();
  }

  /// Switch to a different branch at a fork point.
  Future<void> switchBranch(String forkNodeId, int newChildIndex) async {
    if (conversationId == null) return;

    // Find the children of the fork node
    final children =
        nodes.values.where((n) => n.parentId == forkNodeId).toList()
          ..sort((a, b) => a.branchIndex.compareTo(b.branchIndex));

    if (newChildIndex < 0 || newChildIndex >= children.length) return;

    final targetChild = children[newChildIndex];

    // Find the deepest leaf in this branch
    String leafId = targetChild.id;
    while (true) {
      final childrenOfLeaf = nodes.values
          .where((n) => n.parentId == leafId)
          .toList();
      if (childrenOfLeaf.isEmpty) break;
      // Follow first child (default branch)
      childrenOfLeaf.sort((a, b) => a.branchIndex.compareTo(b.branchIndex));
      leafId = childrenOfLeaf.first.id;
    }

    // Update active path
    _computeActivePath(leafId);
    activeChildAtFork[forkNodeId] = newChildIndex;

    // Notify backend to replay this branch's history into WorkingMemory
    if (_wsSend != null) {
      _wsSend!({
        'type': 'branch_switch',
        'conversation_id': conversationId,
        'leaf_id': leafId,
      });
    }

    notifyListeners();
  }

  /// Get children count at a fork point.
  int getChildCount(String nodeId) => forkPoints[nodeId] ?? 0;

  /// Get active child index at a fork point.
  int getActiveChildIndex(String nodeId) => activeChildAtFork[nodeId] ?? 0;

  /// Check if a node is a fork point (has multiple children).
  bool isForkPoint(String nodeId) => (forkPoints[nodeId] ?? 0) > 1;

  /// Refresh tree by loading the latest conversation from backend.
  ///
  /// If [sessionId] is provided it is sent as query parameter so the
  /// backend returns the conversation linked to that session instead of
  /// the globally most-recent one.
  Future<void> refreshFromSession(ApiClient api, {String? sessionId}) async {
    _api ??= api;
    try {
      final path = sessionId != null && sessionId.isNotEmpty
          ? 'chat/tree/latest?session_id=$sessionId'
          : 'chat/tree/latest';
      final data = await api.get(path);
      final convId = data['conversation_id'] as String?;
      if (convId != null && convId.isNotEmpty) {
        await loadTree(convId);
      }
    } catch (e) {
      debugPrint('[Tree] refreshFromSession failed: $e');
    }
  }

  /// Clear tree state (on session change).
  void clear() {
    conversationId = null;
    nodes.clear();
    activePath.clear();
    forkPoints.clear();
    activeChildAtFork.clear();
    notifyListeners();
  }

  // -- Internal ---------------------------------------------------------------

  void _computeActivePath(String leafId) {
    activePath.clear();
    String? currentId = leafId;
    while (currentId != null) {
      activePath.insert(0, currentId);
      currentId = nodes[currentId]?.parentId;
    }
  }

  void _computeActiveChildIndices() {
    activeChildAtFork.clear();
    final activeSet = activePath.toSet();
    for (final forkId in forkPoints.keys) {
      final children = nodes.values.where((n) => n.parentId == forkId).toList()
        ..sort((a, b) => a.branchIndex.compareTo(b.branchIndex));
      for (var i = 0; i < children.length; i++) {
        if (activeSet.contains(children[i].id)) {
          activeChildAtFork[forkId] = i;
          break;
        }
      }
    }
  }
}
