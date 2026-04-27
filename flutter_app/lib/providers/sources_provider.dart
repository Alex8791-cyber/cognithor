import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

class LeadSourceInfo {
  final String sourceId;
  final String displayName;
  final String icon;
  final String color;
  final Set<String> capabilities;

  const LeadSourceInfo({
    required this.sourceId,
    required this.displayName,
    required this.icon,
    required this.color,
    required this.capabilities,
  });

  factory LeadSourceInfo.fromJson(Map<String, dynamic> json) {
    return LeadSourceInfo(
      sourceId: json['source_id'] as String? ?? '',
      displayName: json['display_name'] as String? ?? '',
      icon: json['icon'] as String? ?? '',
      color: json['color'] as String? ?? '',
      capabilities: Set<String>.from(
        (json['capabilities'] as List?)?.cast<String>() ?? [],
      ),
    );
  }
}

class SourcesProvider extends ChangeNotifier {
  ApiClient? _api;
  List<LeadSourceInfo> _sources = [];
  bool _loading = false;
  String? _error;

  List<LeadSourceInfo> get sources => _sources;
  bool get loading => _loading;
  String? get error => _error;
  bool get isEmpty => _sources.isEmpty;
  bool hasSource(String sourceId) =>
      _sources.any((s) => s.sourceId == sourceId);

  void setApi(ApiClient api) {
    _api = api;
  }

  Future<void> refresh() async {
    if (_api == null) return;
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      final resp = await _api!.get('/api/v1/leads/sources');
      final raw = resp['sources'] as List? ?? [];
      _sources = raw
          .map((e) => LeadSourceInfo.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }
}
