import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

/// Task model matching backend Task.to_dict() output.
class KanbanTask {
  final String id;
  String title;
  String description;
  String status;
  String priority;
  String assignedAgent;
  String source;
  String sourceRef;
  String parentId;
  List<String> labels;
  int sortOrder;
  String createdAt;
  String updatedAt;
  String completedAt;
  String createdBy;
  String resultSummary;
  List<KanbanTask> subtasks;

  KanbanTask({
    required this.id,
    required this.title,
    this.description = '',
    this.status = 'todo',
    this.priority = 'medium',
    this.assignedAgent = '',
    this.source = 'manual',
    this.sourceRef = '',
    this.parentId = '',
    this.labels = const [],
    this.sortOrder = 0,
    this.createdAt = '',
    this.updatedAt = '',
    this.completedAt = '',
    this.createdBy = 'user',
    this.resultSummary = '',
    this.subtasks = const [],
  });

  factory KanbanTask.fromJson(Map<String, dynamic> json) {
    return KanbanTask(
      id: json['id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      description: json['description'] as String? ?? '',
      status: json['status'] as String? ?? 'todo',
      priority: json['priority'] as String? ?? 'medium',
      assignedAgent: json['assigned_agent'] as String? ?? '',
      source: json['source'] as String? ?? 'manual',
      sourceRef: json['source_ref'] as String? ?? '',
      parentId: json['parent_id'] as String? ?? '',
      labels: (json['labels'] as List<dynamic>?)?.cast<String>() ?? [],
      sortOrder: json['sort_order'] as int? ?? 0,
      createdAt: json['created_at'] as String? ?? '',
      updatedAt: json['updated_at'] as String? ?? '',
      completedAt: json['completed_at'] as String? ?? '',
      createdBy: json['created_by'] as String? ?? 'user',
      resultSummary: json['result_summary'] as String? ?? '',
      subtasks:
          (json['subtasks'] as List<dynamic>?)
              ?.map((s) => KanbanTask.fromJson(s as Map<String, dynamic>))
              .toList() ??
          [],
    );
  }

  Map<String, dynamic> toJson() => {
    'title': title,
    'description': description,
    'priority': priority,
    'assigned_agent': assignedAgent,
    'labels': labels,
    'parent_id': parentId,
  };

  String get statusDisplay {
    switch (status) {
      case 'todo':
        return 'To Do';
      case 'in_progress':
        return 'In Progress';
      case 'verifying':
        return 'Verifying';
      case 'done':
        return 'Done';
      case 'blocked':
        return 'Blocked';
      case 'cancelled':
        return 'Cancelled';
      default:
        return status;
    }
  }
}

/// Board statistics from /api/v1/kanban/stats
class KanbanStats {
  final int total;
  final Map<String, int> byStatus;
  final Map<String, int> byAgent;
  final Map<String, int> bySource;

  KanbanStats({
    this.total = 0,
    this.byStatus = const {},
    this.byAgent = const {},
    this.bySource = const {},
  });

  factory KanbanStats.fromJson(Map<String, dynamic> json) {
    return KanbanStats(
      total: json['total'] as int? ?? 0,
      byStatus:
          (json['by_status'] as Map<String, dynamic>?)?.map(
            (k, v) => MapEntry(k, v as int),
          ) ??
          {},
      byAgent:
          (json['by_agent'] as Map<String, dynamic>?)?.map(
            (k, v) => MapEntry(k, v as int),
          ) ??
          {},
      bySource:
          (json['by_source'] as Map<String, dynamic>?)?.map(
            (k, v) => MapEntry(k, v as int),
          ) ??
          {},
    );
  }
}

/// Provider managing Kanban board state with REST API and WebSocket updates.
///
/// Requires an [ApiClient] instance (from [ConnectionProvider]).
class KanbanProvider extends ChangeNotifier {
  ApiClient? _api;

  List<KanbanTask> _tasks = [];
  KanbanStats _stats = KanbanStats();
  bool _loading = false;
  String? _error;
  bool _pipelineMode = false;

  List<KanbanTask> get tasks => _tasks;
  KanbanStats get stats => _stats;
  bool get loading => _loading;
  String? get error => _error;
  bool get pipelineMode => _pipelineMode;

  /// Inject the API client (called when ConnectionProvider connects).
  void setApiClient(ApiClient api) {
    _api = api;
  }

  /// Tasks grouped by status for the board columns.
  Map<String, List<KanbanTask>> get tasksByStatus {
    final map = <String, List<KanbanTask>>{};
    for (final status in [
      'todo',
      'in_progress',
      'verifying',
      'done',
      'blocked',
    ]) {
      map[status] = _tasks.where((t) => t.status == status).toList();
    }
    return map;
  }

  void togglePipelineMode() {
    _pipelineMode = !_pipelineMode;
    notifyListeners();
  }

  /// Check if [resp] indicates an API error.
  bool _isError(Map<String, dynamic> resp) => resp.containsKey('error');

  // ---------------------------------------------------------------------------
  // REST API
  // ---------------------------------------------------------------------------

  Future<void> fetchTasks({String? status, String? agent}) async {
    if (_api == null) return;
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final params = <String, String>{};
      if (status != null) params['status'] = status;
      if (agent != null) params['agent'] = agent;

      final query = params.entries.map((e) => '${e.key}=${e.value}').join('&');
      final path = '/kanban/tasks${query.isNotEmpty ? '?$query' : ''}';
      final resp = await _api!.get(path);

      if (!_isError(resp)) {
        // Backend returns {"tasks": [...]} or the list directly
        final items =
            resp['tasks'] as List<dynamic>? ??
            resp['items'] as List<dynamic>? ??
            [];
        _tasks = items
            .map((j) => KanbanTask.fromJson(j as Map<String, dynamic>))
            .toList();
      } else {
        _error = 'Failed to load tasks: ${resp['error']}';
      }
    } catch (e) {
      _error = 'Network error: $e';
    }

    _loading = false;
    notifyListeners();
  }

  Future<void> fetchStats() async {
    if (_api == null) return;
    try {
      final resp = await _api!.get('/kanban/stats');
      if (!_isError(resp)) {
        _stats = KanbanStats.fromJson(resp);
        notifyListeners();
      }
    } catch (_) {}
  }

  Future<KanbanTask?> createTask({
    required String title,
    String description = '',
    String priority = 'medium',
    String assignedAgent = '',
    List<String> labels = const [],
    String parentId = '',
  }) async {
    if (_api == null) return null;
    try {
      final resp = await _api!.post('/kanban/tasks', {
        'title': title,
        'description': description,
        'priority': priority,
        'assigned_agent': assignedAgent,
        'labels': labels,
        'parent_id': parentId,
      });
      if (!_isError(resp)) {
        final task = KanbanTask.fromJson(resp);
        _tasks.insert(0, task);
        notifyListeners();
        return task;
      }
    } catch (_) {}
    return null;
  }

  Future<bool> moveTask(
    String taskId,
    String newStatus, {
    int sortOrder = 0,
  }) async {
    if (_api == null) return false;
    // Optimistic update
    final idx = _tasks.indexWhere((t) => t.id == taskId);
    if (idx >= 0) {
      _tasks[idx].status = newStatus;
      _tasks[idx].sortOrder = sortOrder;
      notifyListeners();
    }

    try {
      final resp = await _api!.post('/kanban/tasks/$taskId/move', {
        'status': newStatus,
        'sort_order': sortOrder,
      });
      if (!_isError(resp)) {
        await fetchTasks();
        return true;
      } else {
        await fetchTasks();
        return false;
      }
    } catch (_) {
      await fetchTasks();
      return false;
    }
  }

  Future<bool> updateTask(String taskId, Map<String, dynamic> fields) async {
    if (_api == null) return false;
    try {
      final resp = await _api!.patch('/kanban/tasks/$taskId', fields);
      if (!_isError(resp)) {
        await fetchTasks();
        return true;
      }
    } catch (_) {}
    return false;
  }

  Future<bool> deleteTask(String taskId) async {
    if (_api == null) return false;
    try {
      final resp = await _api!.delete('/kanban/tasks/$taskId');
      if (!_isError(resp)) {
        _tasks.removeWhere((t) => t.id == taskId);
        notifyListeners();
        return true;
      }
    } catch (_) {}
    return false;
  }

  Future<List<Map<String, dynamic>>> getHistory(String taskId) async {
    if (_api == null) return [];
    try {
      final resp = await _api!.get('/kanban/tasks/$taskId/history');
      if (!_isError(resp)) {
        final items = resp['history'] as List<dynamic>? ?? [];
        return items.cast<Map<String, dynamic>>();
      }
    } catch (_) {}
    return [];
  }

  Future<void> reorderTasks(List<Map<String, dynamic>> batch) async {
    if (_api == null || batch.isEmpty) return;
    // Optimistic: already updated in-memory by caller
    notifyListeners();
    try {
      await _api!.post('/kanban/tasks/reorder', {'items': batch});
    } catch (_) {
      await fetchTasks();
    }
  }

  // ---------------------------------------------------------------------------
  // WebSocket handler
  // ---------------------------------------------------------------------------

  void onKanbanUpdate(Map<String, dynamic> msg) {
    final action = msg['action'] as String? ?? '';
    switch (action) {
      case 'created':
        if (msg['task'] != null) {
          _tasks.insert(
            0,
            KanbanTask.fromJson(msg['task'] as Map<String, dynamic>),
          );
        }
        break;
      case 'updated':
      case 'moved':
        fetchTasks();
        return;
      case 'deleted':
        final taskId = msg['task_id'] as String? ?? '';
        _tasks.removeWhere((t) => t.id == taskId);
        break;
    }
    notifyListeners();
  }
}
