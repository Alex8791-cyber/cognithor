import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

class ResearchSummary {
  final String id;
  final String query;
  final int hops;
  final double confidenceAvg;
  final double createdAt;

  const ResearchSummary({
    required this.id,
    required this.query,
    required this.hops,
    required this.confidenceAvg,
    required this.createdAt,
  });

  factory ResearchSummary.fromJson(Map<String, dynamic> json) {
    return ResearchSummary(
      id: json['id'] as String? ?? '',
      query: json['query'] as String? ?? '',
      hops: json['hops'] as int? ?? 0,
      confidenceAvg: (json['confidence_avg'] as num?)?.toDouble() ?? 0.0,
      createdAt: (json['created_at'] as num?)?.toDouble() ?? 0.0,
    );
  }

  String get timeAgo {
    final diff = DateTime.now().difference(
      DateTime.fromMillisecondsSinceEpoch((createdAt * 1000).toInt()),
    );
    if (diff.inDays > 0) return '${diff.inDays}d ago';
    if (diff.inHours > 0) return '${diff.inHours}h ago';
    if (diff.inMinutes > 0) return '${diff.inMinutes}m ago';
    return 'just now';
  }
}

class ResearchResult {
  final String id;
  final String query;
  final String reportMd;
  final int hops;
  final double confidenceAvg;
  final List<Map<String, dynamic>> sources;

  const ResearchResult({
    required this.id,
    required this.query,
    required this.reportMd,
    required this.hops,
    required this.confidenceAvg,
    required this.sources,
  });

  factory ResearchResult.fromJson(Map<String, dynamic> json) {
    return ResearchResult(
      id: json['id'] as String? ?? '',
      query: json['query'] as String? ?? '',
      reportMd: json['report_md'] as String? ?? '',
      hops: json['hops'] as int? ?? 0,
      confidenceAvg: (json['confidence_avg'] as num?)?.toDouble() ?? 0.0,
      sources: List<Map<String, dynamic>>.from(json['sources'] as List? ?? []),
    );
  }
}

class ResearchProvider extends ChangeNotifier {
  ApiClient? _api;
  ResearchResult? _activeResult;
  List<ResearchSummary> _history = [];
  bool _loading = false;
  String? _error;
  String? _activeResearchId;

  ResearchResult? get activeResult => _activeResult;
  List<ResearchSummary> get history => _history;
  bool get loading => _loading;
  String? get error => _error;

  void setApi(ApiClient api) {
    _api = api;
  }

  Future<void> startResearch(String query) async {
    if (_api == null) return;
    _loading = true;
    _error = null;
    _activeResult = null;
    notifyListeners();
    try {
      final resp = await _api!.post('/api/v1/research/query', {'query': query});
      _activeResearchId = resp['id'] as String?;
      if (_activeResearchId != null) {
        await _pollResult(_activeResearchId!);
      }
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> _pollResult(String id) async {
    // Research is async on backend — poll until complete
    for (int i = 0; i < 60; i++) {
      await Future<void>.delayed(const Duration(seconds: 2));
      try {
        final resp = await _api!.get('/api/v1/research/$id');
        if (resp['status'] == 'complete' || resp['report_md'] != null) {
          _activeResult = ResearchResult.fromJson(resp);
          notifyListeners();
          return;
        }
      } catch (_) {}
    }
    _error = 'Research timed out';
  }

  Future<void> loadHistory() async {
    if (_api == null) return;
    try {
      final resp = await _api!.get('/api/v1/research/history');
      final raw = resp['results'] as List? ?? [];
      _history = raw
          .map((e) => ResearchSummary.fromJson(e as Map<String, dynamic>))
          .toList();
      notifyListeners();
    } catch (_) {}
  }

  Future<void> loadResult(String id) async {
    if (_api == null) return;
    _loading = true;
    notifyListeners();
    try {
      final resp = await _api!.get('/api/v1/research/$id');
      _activeResult = ResearchResult.fromJson(resp);
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> deleteResearch(String id) async {
    if (_api == null) return;
    try {
      await _api!.delete('/api/v1/research/$id');
      _history.removeWhere((r) => r.id == id);
      if (_activeResult?.id == id) _activeResult = null;
      notifyListeners();
    } catch (_) {}
  }

  Future<String?> exportResearch(String id, String format) async {
    if (_api == null) return null;
    try {
      final resp = await _api!.post('/api/v1/research/$id/export', {
        'format': format,
      });
      return resp['path'] as String?;
    } catch (_) {
      return null;
    }
  }
}
