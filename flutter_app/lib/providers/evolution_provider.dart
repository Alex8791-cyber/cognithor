import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

class EvolutionGoal {
  final String id;
  final String title;
  final String description;
  final String status;
  final double progress;
  final int priority;
  final List<String> tags;

  EvolutionGoal({
    required this.id,
    required this.title,
    this.description = '',
    this.status = 'active',
    this.progress = 0.0,
    this.priority = 3,
    this.tags = const [],
  });

  factory EvolutionGoal.fromJson(Map<String, dynamic> json) {
    return EvolutionGoal(
      id: json['id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      description: json['description'] as String? ?? '',
      status: json['status'] as String? ?? 'active',
      progress: (json['progress'] as num?)?.toDouble() ?? 0.0,
      priority: json['priority'] as int? ?? 3,
      tags: (json['tags'] as List<dynamic>?)?.cast<String>() ?? [],
    );
  }
}

class EvolutionPlan {
  final String id;
  final String goal;
  final String status;
  final int subGoalsTotal;
  final int subGoalsPassed;
  final double coverageScore;
  final double qualityScore;
  final String cycleState;

  EvolutionPlan({
    required this.id,
    required this.goal,
    this.status = '',
    this.subGoalsTotal = 0,
    this.subGoalsPassed = 0,
    this.coverageScore = 0.0,
    this.qualityScore = 0.0,
    this.cycleState = 'unknown',
  });

  factory EvolutionPlan.fromJson(Map<String, dynamic> json) {
    return EvolutionPlan(
      id: json['id'] as String? ?? '',
      goal: json['goal'] as String? ?? '',
      status: json['status'] as String? ?? '',
      subGoalsTotal: json['sub_goals_total'] as int? ?? 0,
      subGoalsPassed: json['sub_goals_passed'] as int? ?? 0,
      coverageScore: (json['coverage_score'] as num?)?.toDouble() ?? 0.0,
      qualityScore: (json['quality_score'] as num?)?.toDouble() ?? 0.0,
      cycleState: json['cycle_state'] as String? ?? 'unknown',
    );
  }

  double get completionPercent =>
      subGoalsTotal > 0 ? subGoalsPassed / subGoalsTotal : 0.0;
}

class EvolutionProvider extends ChangeNotifier {
  ApiClient? _api;

  List<EvolutionGoal> _goals = [];
  List<EvolutionPlan> _plans = [];
  String _journal = '';
  Map<String, dynamic> _stats = {};
  bool _loading = false;

  List<EvolutionGoal> get goals => _goals;
  List<EvolutionPlan> get plans => _plans;
  String get journal => _journal;
  Map<String, dynamic> get stats => _stats;
  bool get loading => _loading;

  void setApi(ApiClient api) {
    _api = api;
  }

  Future<void> fetchGoals() async {
    if (_api == null) return;
    _loading = true;
    notifyListeners();
    try {
      final data = await _api!.get('evolution/goals');
      if (data['error'] == null) {
        // Handle both {"goals": [...]} and raw-list responses.
        final rawList = data.containsKey('goals')
            ? data['goals'] as List<dynamic>
            : <dynamic>[];
        _goals = rawList
            .map((j) => EvolutionGoal.fromJson(j as Map<String, dynamic>))
            .toList();
      }
    } catch (_) {}
    _loading = false;
    notifyListeners();
  }

  Future<void> fetchPlans() async {
    if (_api == null) return;
    try {
      final data = await _api!.get('evolution/plans');
      if (data['error'] == null) {
        final rawList = data.containsKey('plans')
            ? data['plans'] as List<dynamic>
            : <dynamic>[];
        _plans = rawList
            .map((j) => EvolutionPlan.fromJson(j as Map<String, dynamic>))
            .toList();
        notifyListeners();
      }
    } catch (_) {}
  }

  Future<void> fetchJournal({int days = 7}) async {
    if (_api == null) return;
    try {
      final data = await _api!.get('evolution/journal?days=$days');
      if (data['error'] == null) {
        _journal = data['content'] as String? ?? '';
        notifyListeners();
      }
    } catch (_) {}
  }

  Future<void> fetchStats() async {
    if (_api == null) return;
    try {
      final data = await _api!.get('evolution/stats');
      if (data['error'] == null) {
        _stats = data;
        notifyListeners();
      }
    } catch (_) {}
  }

  Future<bool> createGoal({
    required String title,
    String description = '',
    int priority = 3,
  }) async {
    if (_api == null) return false;
    try {
      final data = await _api!.post('evolution/goals', {
        'title': title,
        'description': description,
        'priority': priority,
      });
      if (data['error'] == null) {
        await fetchGoals();
        return true;
      }
    } catch (_) {}
    return false;
  }

  Future<bool> updateGoal(
    String goalId, {
    String? title,
    String? description,
    String? status,
    int? priority,
  }) async {
    if (_api == null) return false;
    try {
      final body = <String, dynamic>{};
      if (title != null) body['title'] = title;
      if (description != null) body['description'] = description;
      if (status != null) body['status'] = status;
      if (priority != null) body['priority'] = priority;
      final data = await _api!.patch('evolution/goals/$goalId', body);
      if (data['error'] == null) {
        await fetchGoals();
        return true;
      }
    } catch (_) {}
    return false;
  }

  Future<bool> deleteGoal(String goalId) async {
    if (_api == null) return false;
    try {
      final data = await _api!.delete('evolution/goals/$goalId');
      if (data['error'] == null) {
        await fetchGoals();
        return true;
      }
    } catch (_) {}
    return false;
  }

  Future<void> fetchAll() async {
    await Future.wait([
      fetchGoals(),
      fetchPlans(),
      fetchJournal(),
      fetchStats(),
    ]);
  }
}
