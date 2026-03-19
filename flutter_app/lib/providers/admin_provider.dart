/// Admin / system state provider.
library;

import 'package:flutter/foundation.dart';
import 'package:jarvis_ui/services/api_client.dart';

class AdminProvider extends ChangeNotifier {
  ApiClient? _api;

  void setApi(ApiClient? api) {
    _api = api;
  }

  Map<String, dynamic>? systemStatus;
  List<dynamic> agents = [];
  Map<String, dynamic>? models;
  Map<String, dynamic>? modelStats;
  Map<String, dynamic>? vaultStats;
  List<dynamic> vaultAgents = [];
  List<dynamic> credentials = [];
  List<dynamic> bindings = [];
  List<dynamic> commands = [];
  List<dynamic> connectors = [];
  Map<String, dynamic>? isolationStats;
  Map<String, dynamic>? circles;
  bool isLoading = false;
  String? error;

  Future<void> loadSystemStatus() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      systemStatus = await _api!.getSystemStatus();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadAgents() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getAgents();
      agents = data['agents'] as List<dynamic>? ?? [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<Map<String, dynamic>?> getAgent(String name) async {
    if (_api == null) return null;
    try {
      final data = await _api!.getAgent(name);
      if (data.containsKey('error')) return null;
      return data;
    } catch (e) {
      error = e.toString();
      return null;
    }
  }

  Future<bool> createAgent(Map<String, dynamic> body) async {
    if (_api == null) return false;
    error = null;
    try {
      final res = await _api!.createAgent(body);
      if (res.containsKey('error')) {
        error = res['error'].toString();
        return false;
      }
      await loadAgents();
      return true;
    } catch (e) {
      error = e.toString();
      return false;
    }
  }

  Future<bool> updateAgent(String name, Map<String, dynamic> body) async {
    if (_api == null) return false;
    error = null;
    try {
      final res = await _api!.updateAgent(name, body);
      if (res.containsKey('error')) {
        error = res['error'].toString();
        return false;
      }
      await loadAgents();
      return true;
    } catch (e) {
      error = e.toString();
      return false;
    }
  }

  Future<bool> deleteAgent(String name) async {
    if (_api == null) return false;
    error = null;
    try {
      final res = await _api!.deleteAgent(name);
      if (res.containsKey('error')) {
        error = res['error'].toString();
        return false;
      }
      await loadAgents();
      return true;
    } catch (e) {
      error = e.toString();
      return false;
    }
  }

  Future<void> loadModels() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      models = await _api!.getModels();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadModelStats() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      modelStats = await _api!.getModelStats();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadVaultStats() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      vaultStats = await _api!.getVaultStats();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadVaultAgents() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getVaultAgents();
      vaultAgents = data['agents'] as List<dynamic>? ?? [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadCredentials() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getCredentials();
      credentials = data['credentials'] as List<dynamic>? ?? [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadBindings() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getBindings();
      bindings = data['bindings'] as List<dynamic>? ?? [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadCommands() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getCommands();
      commands = data['commands'] as List<dynamic>? ?? [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadConnectors() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getConnectors();
      connectors = data['connectors'] as List<dynamic>? ?? [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadIsolationStats() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      isolationStats = await _api!.getIsolationStats();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadCircles() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      circles = await _api!.getCircles();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> reloadConfig() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      await _api!.reloadConfig();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> shutdown() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      await _api!.shutdownServer();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }
}
