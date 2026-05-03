import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

class LoadedPack {
  final String qualifiedId;
  final String version;
  final String displayName;
  final List<String> tools;

  const LoadedPack({
    required this.qualifiedId,
    required this.version,
    required this.displayName,
    required this.tools,
  });

  factory LoadedPack.fromJson(Map<String, dynamic> json) {
    return LoadedPack(
      qualifiedId: json['qualified_id'] as String? ?? '',
      version: json['version'] as String? ?? '',
      displayName: json['display_name'] as String? ?? '',
      tools: List<String>.from(json['tools'] as List? ?? []),
    );
  }
}

class PacksProvider extends ChangeNotifier {
  ApiClient? _api;
  List<LoadedPack> _packs = [];

  List<LoadedPack> get packs => _packs;

  bool hasPackLoaded(String qualifiedId) =>
      _packs.any((p) => p.qualifiedId == qualifiedId);

  void setApi(ApiClient api) {
    _api = api;
  }

  Future<void> refresh() async {
    if (_api == null) return;
    try {
      final resp = await _api!.get('packs/loaded');
      final raw = resp['packs'] as List? ?? [];
      _packs = raw
          .map((e) => LoadedPack.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {}
    notifyListeners();
  }
}
