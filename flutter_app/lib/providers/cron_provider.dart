import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

class CronJob {
  String name;
  String schedule;
  String prompt;
  String channel;
  String model;
  bool enabled;
  String agent;
  String? nextRun;

  CronJob({
    required this.name,
    required this.schedule,
    required this.prompt,
    this.channel = 'telegram',
    this.model = 'qwen3:8b',
    this.enabled = false,
    this.agent = '',
    this.nextRun,
  });

  factory CronJob.fromJson(Map<String, dynamic> j) => CronJob(
    name: j['name'] as String? ?? '',
    schedule: j['schedule'] as String? ?? '',
    prompt: j['prompt'] as String? ?? '',
    channel: j['channel'] as String? ?? 'telegram',
    model: j['model'] as String? ?? 'qwen3:8b',
    enabled: j['enabled'] as bool? ?? false,
    agent: j['agent'] as String? ?? '',
    nextRun: j['next_run'] as String?,
  );

  /// Human-readable schedule description.
  String get scheduleLabel {
    final parts = schedule.split(' ');
    if (parts.length < 5) return schedule;
    final minute = parts[0];
    final hour = parts[1];
    final dow = parts[4];
    if (dow == '1-5') return 'Weekdays $hour:${minute.padLeft(2, '0')}';
    if (dow == '5') return 'Fridays $hour:${minute.padLeft(2, '0')}';
    if (parts[2] == '1' && parts[3] == '*') {
      return 'Monthly (1st) $hour:${minute.padLeft(2, '0')}';
    }
    if (dow == '*' && parts[2] == '*') {
      return 'Daily $hour:${minute.padLeft(2, '0')}';
    }
    return schedule;
  }
}

class CronProvider extends ChangeNotifier {
  ApiClient? _api;
  List<CronJob> _jobs = [];
  bool _loading = false;
  String? _error;
  Timer? _refreshTimer;

  List<CronJob> get jobs => _jobs;
  bool get loading => _loading;
  String? get error => _error;

  void setApiClient(ApiClient? api) {
    _api = api;
    if (api != null) {
      fetchJobs();
      // Refresh every 30s for next_run updates
      _refreshTimer?.cancel();
      _refreshTimer = Timer.periodic(
        const Duration(seconds: 30),
        (_) => fetchJobs(),
      );
    }
  }

  Future<void> fetchJobs() async {
    if (_api == null) return;
    _loading = _jobs.isEmpty;
    notifyListeners();

    try {
      final resp = await _api!.get('/cron-jobs/enriched');
      final items = resp['jobs'] as List<dynamic>? ?? [];
      _jobs = items
          .map((j) => CronJob.fromJson(j as Map<String, dynamic>))
          .toList();
      _error = null;
    } catch (e) {
      _error = '$e';
    }

    _loading = false;
    notifyListeners();
  }

  Future<bool> toggleJob(String name) async {
    if (_api == null) return false;
    // Optimistic toggle
    final idx = _jobs.indexWhere((j) => j.name == name);
    if (idx >= 0) {
      _jobs[idx].enabled = !_jobs[idx].enabled;
      notifyListeners();
    }
    try {
      final resp = await _api!.patch('/cron-jobs/$name/toggle', {});
      if (resp.containsKey('error') && resp['status'] != null) {
        // Revert on failure
        if (idx >= 0) {
          _jobs[idx].enabled = !_jobs[idx].enabled;
          notifyListeners();
        }
        return false;
      }
      await fetchJobs();
      return true;
    } catch (_) {
      if (idx >= 0) {
        _jobs[idx].enabled = !_jobs[idx].enabled;
        notifyListeners();
      }
      return false;
    }
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }
}
