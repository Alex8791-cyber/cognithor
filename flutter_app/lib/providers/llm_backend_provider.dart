import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class HardwareInfo {
  final String gpuName;
  final int vramGb;
  final String computeCapability;

  HardwareInfo({
    required this.gpuName,
    required this.vramGb,
    required this.computeCapability,
  });

  factory HardwareInfo.fromJson(Map<String, dynamic> j) => HardwareInfo(
    gpuName: j['gpu_name'] as String,
    vramGb: j['vram_gb'] as int,
    computeCapability: j['compute_capability'] as String,
  );
}

class VLLMStatus {
  final bool hardwareOk;
  final HardwareInfo? hardwareInfo;
  final bool dockerOk;
  final bool imagePulled;
  final bool containerRunning;
  final String? currentModel;
  final String? lastError;

  VLLMStatus({
    required this.hardwareOk,
    required this.hardwareInfo,
    required this.dockerOk,
    required this.imagePulled,
    required this.containerRunning,
    required this.currentModel,
    required this.lastError,
  });

  factory VLLMStatus.fromJson(Map<String, dynamic> j) => VLLMStatus(
    hardwareOk: j['hardware_ok'] as bool,
    hardwareInfo: j['hardware_info'] == null
        ? null
        : HardwareInfo.fromJson(j['hardware_info'] as Map<String, dynamic>),
    dockerOk: j['docker_ok'] as bool,
    imagePulled: j['image_pulled'] as bool,
    containerRunning: j['container_running'] as bool,
    currentModel: j['current_model'] as String?,
    lastError: j['last_error'] as String?,
  );
}

class BackendEntry {
  final String name;
  final bool enabled;
  final String status;

  BackendEntry({
    required this.name,
    required this.enabled,
    required this.status,
  });

  factory BackendEntry.fromJson(Map<String, dynamic> j) => BackendEntry(
    name: j['name'] as String,
    enabled: j['enabled'] as bool,
    status: j['status'] as String,
  );
}

class LlmBackendProvider extends ChangeNotifier {
  final String apiBaseUrl;
  final http.Client _http;
  Timer? _pollTimer;

  List<BackendEntry> backends = [];
  String active = 'ollama';
  VLLMStatus? vllmStatus;
  String? error;
  List<Map<String, dynamic>> availableModels = [];
  String? recommendedModelId;

  // VLM-Router quality default (Sprint-VLM-Router 2026-05-07).
  // ``null`` means: let the router decide per-request via heuristic.
  String? vlmQualityDefault;
  List<Map<String, dynamic>> vlmProfiles = [];

  bool get isPolling => _pollTimer != null;

  LlmBackendProvider({required this.apiBaseUrl, http.Client? httpClient})
    : _http = httpClient ?? http.Client();

  String? _authToken;

  /// Set the API auth token (Bearer). Wired from ConnectionProvider in
  /// main.dart — the /api/backends/* endpoints are token-gated, so without
  /// this every request returns 401 and the vLLM screen shows empty defaults.
  void setAuthToken(String? token) {
    _authToken = token;
  }

  Map<String, String> _headers({bool json = false}) {
    final h = <String, String>{};
    if (json) h['content-type'] = 'application/json';
    final t = _authToken;
    if (t != null && t.isNotEmpty) h['Authorization'] = 'Bearer $t';
    return h;
  }

  Future<void> refreshList() async {
    try {
      final r = await _http.get(
        Uri.parse('$apiBaseUrl/api/backends'),
        headers: _headers(),
      );
      if (r.statusCode != 200) return;
      final body = jsonDecode(r.body) as Map<String, dynamic>;
      active = body['active'] as String;
      backends = (body['backends'] as List)
          .map((b) => BackendEntry.fromJson(b as Map<String, dynamic>))
          .toList();
      notifyListeners();
    } catch (e) {
      error = e.toString();
      notifyListeners();
    }
  }

  Future<void> refreshVllmStatus() async {
    try {
      final r = await _http.get(
        Uri.parse('$apiBaseUrl/api/backends/vllm/status'),
        headers: _headers(),
      );
      if (r.statusCode != 200) return;
      vllmStatus = VLLMStatus.fromJson(
        jsonDecode(r.body) as Map<String, dynamic>,
      );
      notifyListeners();
    } catch (e) {
      error = e.toString();
      notifyListeners();
    }
  }

  /// Start polling `/api/backends/vllm/status` every 2 seconds.
  /// Call from VllmSetupScreen.initState, stop in dispose.
  void startPolling() {
    stopPolling();
    refreshVllmStatus();
    _pollTimer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => refreshVllmStatus(),
    );
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  @override
  void dispose() {
    stopPolling();
    _http.close();
    super.dispose();
  }

  /// Kick off docker-pull and yield progress events as parsed maps.
  /// Events: {"status":"Downloading","progressDetail":{"current":N,"total":M},"id":"layer..."}
  Stream<Map<String, dynamic>> pullImage() async* {
    final uri = Uri.parse('$apiBaseUrl/api/backends/vllm/pull-image');
    final request = http.Request('POST', uri);
    request.headers.addAll(_headers());
    final streamed = await _http.send(request);
    if (streamed.statusCode != 200) {
      throw Exception('Pull failed: HTTP ${streamed.statusCode}');
    }
    String buffer = '';
    await for (final chunk in streamed.stream.transform(utf8.decoder)) {
      buffer += chunk;
      while (true) {
        final sep = buffer.indexOf('\n\n');
        if (sep == -1) break;
        final block = buffer.substring(0, sep);
        buffer = buffer.substring(sep + 2);
        for (final line in block.split('\n')) {
          if (line.startsWith('data:')) {
            final payload = line.substring(5).trim();
            if (payload.isEmpty) continue;
            try {
              yield jsonDecode(payload) as Map<String, dynamic>;
            } catch (_) {
              // Ignore malformed events
            }
          }
        }
      }
    }
    // Refresh full status after pull completes
    await refreshVllmStatus();
  }

  /// POST /api/backends/vllm/start — start a container for the given model.
  Future<void> startContainer(String model) async {
    final r = await _http.post(
      Uri.parse('$apiBaseUrl/api/backends/vllm/start'),
      headers: _headers(json: true),
      body: jsonEncode({'model': model}),
    );
    if (r.statusCode != 200) {
      final body = jsonDecode(r.body);
      throw Exception(body['detail']?['message'] ?? 'Start failed');
    }
    await refreshVllmStatus();
  }

  Future<void> setActive(String backend) async {
    final r = await _http.post(
      Uri.parse('$apiBaseUrl/api/backends/active'),
      headers: _headers(json: true),
      body: jsonEncode({'backend': backend}),
    );
    if (r.statusCode == 200) {
      active = backend;
      notifyListeners();
    } else {
      throw Exception('Backend switch failed: ${r.statusCode}');
    }
  }

  Future<void> fetchAvailableModels() async {
    final r = await _http.get(
      Uri.parse('$apiBaseUrl/api/backends/vllm/available-models'),
      headers: _headers(),
    );
    if (r.statusCode != 200) return;
    final body = jsonDecode(r.body) as Map<String, dynamic>;
    recommendedModelId = body['recommended_id'] as String?;
    availableModels = (body['models'] as List).cast<Map<String, dynamic>>();
    notifyListeners();
  }

  /// Read the VLM-Router quality override + the available tiers from the
  /// backend. Pulls both in one round-trip so the dropdown can render
  /// rich rows immediately.
  Future<void> fetchVlmQualityDefault() async {
    final r = await _http.get(
      Uri.parse('$apiBaseUrl/api/backends/vllm/quality-default'),
      headers: _headers(),
    );
    if (r.statusCode != 200) return;
    final body = jsonDecode(r.body) as Map<String, dynamic>;
    vlmQualityDefault = body['current'] as String?;
    vlmProfiles = (body['profiles'] as List).cast<Map<String, dynamic>>();
    notifyListeners();
  }

  /// Persist a VLM-Router quality override (or clear it with ``null``).
  /// Throws on non-200 so the UI can surface a SnackBar.
  Future<void> setVlmQualityDefault(String? quality) async {
    final r = await _http.post(
      Uri.parse('$apiBaseUrl/api/backends/vllm/quality-default'),
      headers: _headers(json: true),
      body: jsonEncode({'quality': quality}),
    );
    if (r.statusCode == 200) {
      vlmQualityDefault = quality;
      notifyListeners();
    } else {
      throw Exception('Set VLM quality failed: ${r.statusCode} — ${r.body}');
    }
  }
}
