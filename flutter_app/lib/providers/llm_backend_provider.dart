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
            : HardwareInfo.fromJson(
                j['hardware_info'] as Map<String, dynamic>),
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

  bool get isPolling => _pollTimer != null;

  LlmBackendProvider({required this.apiBaseUrl, http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  Future<void> refreshList() async {
    try {
      final r = await _http.get(Uri.parse('$apiBaseUrl/api/backends'));
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
      final r =
          await _http.get(Uri.parse('$apiBaseUrl/api/backends/vllm/status'));
      if (r.statusCode != 200) return;
      vllmStatus =
          VLLMStatus.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
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

  Future<void> setActive(String backend) async {
    final r = await _http.post(
      Uri.parse('$apiBaseUrl/api/backends/active'),
      headers: {'content-type': 'application/json'},
      body: jsonEncode({'backend': backend}),
    );
    if (r.statusCode == 200) {
      active = backend;
      notifyListeners();
    } else {
      throw Exception('Backend switch failed: ${r.statusCode}');
    }
  }
}
