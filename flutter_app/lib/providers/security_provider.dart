/// Security & compliance state provider.
library;

import 'package:flutter/foundation.dart';
import 'package:jarvis_ui/services/api_client.dart';

class SecurityProvider extends ChangeNotifier {
  ApiClient? _api;

  void setApi(ApiClient? api) {
    _api = api;
  }

  Map<String, dynamic>? roles;
  Map<String, dynamic>? complianceReport;
  Map<String, dynamic>? complianceStats;
  Map<String, dynamic>? decisions;
  Map<String, dynamic>? remediations;
  Map<String, dynamic>? redteamStatus;
  List<dynamic> auditEntries = [];
  Map<String, dynamic>? authStats;
  bool isLoading = false;
  String? error;

  Future<void> loadRoles() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      roles = await _api!.getRbacRoles();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadComplianceReport() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      complianceReport = await _api!.getComplianceReport();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadComplianceStats() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      complianceStats = await _api!.getComplianceStats();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadDecisions() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      decisions = await _api!.getComplianceDecisions();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadRemediations() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      remediations = await _api!.getComplianceRemediations();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadRedteamStatus() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      redteamStatus = await _api!.getRedteamStatus();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> runRedteamScan(Map<String, dynamic> policy) async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      await _api!.runRedteamScan(policy);
      await loadRedteamStatus();
    } catch (e) {
      error = e.toString();
      isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loadAudit({String? action, String? severity}) async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      final data = await _api!.getMonitoringAudit(
        action: action,
        severity: severity,
      );
      auditEntries = data['entries'] is List ? data['entries'] as List : [];
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }

  Future<void> loadAuthStats() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      authStats = await _api!.getAuthStats();
    } catch (e) {
      error = e.toString();
    }
    isLoading = false;
    notifyListeners();
  }
}
