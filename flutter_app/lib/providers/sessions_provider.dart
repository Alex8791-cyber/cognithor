/// Session history management.
///
/// Manages chat sessions — listing, creating, loading history,
/// deleting, and renaming conversations.
library;

import 'package:flutter/foundation.dart' show ChangeNotifier, debugPrint, kDebugMode;
import 'package:jarvis_ui/services/api_client.dart';

void _log(String msg) {
  if (kDebugMode) debugPrint(msg);
}

class SessionsProvider extends ChangeNotifier {
  ApiClient? _api;

  List<Map<String, dynamic>> sessions = [];
  String? activeSessionId;
  bool isLoading = false;
  String? error;

  void setApi(ApiClient api) {
    _api = api;
  }

  Future<void> loadSessions() async {
    if (_api == null) return;
    isLoading = true;
    error = null;
    notifyListeners();

    try {
      final result = await _api!.listSessions();
      if (result.containsKey('error')) {
        error = result['error'].toString();
      } else {
        final raw = result['sessions'];
        if (raw is List) {
          sessions = raw.cast<Map<String, dynamic>>();
        }
      }
    } catch (e) {
      _log('[Sessions] loadSessions error: $e');
      error = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  Future<String?> createNewSession() async {
    if (_api == null) return null;
    try {
      final result = await _api!.createSession();
      if (result.containsKey('error')) {
        error = result['error'].toString();
        notifyListeners();
        return null;
      }
      final sessionId = result['session_id'] as String?;
      if (sessionId != null) {
        activeSessionId = sessionId;
        await loadSessions();
      }
      return sessionId;
    } catch (e) {
      _log('[Sessions] createNewSession error: $e');
      error = e.toString();
      notifyListeners();
      return null;
    }
  }

  Future<List<Map<String, dynamic>>?> loadHistory(String sessionId) async {
    if (_api == null) return null;
    try {
      final result = await _api!.getSessionHistory(sessionId);
      if (result.containsKey('error')) {
        error = result['error'].toString();
        notifyListeners();
        return null;
      }
      activeSessionId = sessionId;
      notifyListeners();
      final raw = result['messages'];
      if (raw is List) {
        return raw.cast<Map<String, dynamic>>();
      }
      return [];
    } catch (e) {
      _log('[Sessions] loadHistory error: $e');
      error = e.toString();
      notifyListeners();
      return null;
    }
  }

  Future<void> deleteSession(String sessionId) async {
    if (_api == null) return;
    try {
      await _api!.deleteSession(sessionId);
      if (activeSessionId == sessionId) {
        activeSessionId = null;
      }
      await loadSessions();
    } catch (e) {
      _log('[Sessions] deleteSession error: $e');
      error = e.toString();
      notifyListeners();
    }
  }

  Future<void> renameSession(String sessionId, String title) async {
    if (_api == null) return;
    try {
      await _api!.renameSession(sessionId, title);
      await loadSessions();
    } catch (e) {
      _log('[Sessions] renameSession error: $e');
      error = e.toString();
      notifyListeners();
    }
  }
}
