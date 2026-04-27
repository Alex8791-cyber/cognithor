/// Session history management.
///
/// Manages chat sessions — listing, creating, loading history,
/// deleting, and renaming conversations.
library;

import 'package:flutter/foundation.dart'
    show ChangeNotifier, debugPrint, kDebugMode;
import 'package:cognithor_ui/services/api_client.dart';

void _log(String msg) {
  if (kDebugMode) debugPrint(msg);
}

class SessionsProvider extends ChangeNotifier {
  ApiClient? _api;

  List<Map<String, dynamic>> sessions = [];
  List<String> folders = [];
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

  /// Create an incognito session (no memory enrichment, no persistence).
  Future<String?> createIncognitoSession() async {
    if (_api == null) return null;
    try {
      final data = await _api!.createIncognitoSession();
      if (data.containsKey('error')) return null;
      final id = data['session_id'] as String?;
      if (id != null) {
        activeSessionId = id;
        await loadSessions();
        notifyListeners();
      }
      return id;
    } catch (_) {
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

  List<Map<String, dynamic>> searchResults = [];

  Future<void> searchChats(String query) async {
    if (_api == null || query.trim().isEmpty) {
      searchResults = [];
      notifyListeners();
      return;
    }
    try {
      final data = await _api!.searchSessions(query);
      searchResults = List<Map<String, dynamic>>.from(data['results'] ?? []);
    } catch (_) {
      searchResults = [];
    }
    notifyListeners();
  }

  Future<void> loadFolders() async {
    if (_api == null) return;
    try {
      final result = await _api!.listFolders();
      if (result.containsKey('error')) {
        _log('[Sessions] loadFolders error: ${result['error']}');
        return;
      }
      final raw = result['folders'];
      if (raw is List) {
        folders = raw.cast<String>();
        notifyListeners();
      }
    } catch (e) {
      _log('[Sessions] loadFolders error: $e');
    }
  }

  /// Check if we should auto-create a new session on app open.
  /// Returns the session ID to use (new or existing).
  Future<String?> autoSessionOnStartup({int timeoutMinutes = 30}) async {
    if (_api == null) return null;
    try {
      final shouldNew = await _api!.shouldNewSession(
        timeoutMinutes: timeoutMinutes,
      );
      if (shouldNew) {
        return createNewSession();
      }
      // Resume most recent session
      await loadSessions();
      if (sessions.isNotEmpty) {
        final mostRecent = sessions.first;
        activeSessionId = mostRecent['id'] as String?;
        return activeSessionId;
      }
      return createNewSession();
    } catch (_) {
      return null;
    }
  }

  /// Sessions grouped by folder/project for sidebar display.
  Map<String, List<Map<String, dynamic>>> get sessionsByProject {
    final grouped = <String, List<Map<String, dynamic>>>{};
    for (final s in sessions) {
      final folder = (s['folder'] as String?) ?? '';
      final key = folder.isEmpty ? 'Allgemein' : folder;
      grouped.putIfAbsent(key, () => []).add(s);
    }
    // Sort: 'Allgemein' last, rest alphabetical
    final sorted = Map<String, List<Map<String, dynamic>>>.fromEntries(
      grouped.entries.toList()..sort((a, b) {
        if (a.key == 'Allgemein') return 1;
        if (b.key == 'Allgemein') return -1;
        return a.key.compareTo(b.key);
      }),
    );
    return sorted;
  }

  Future<void> moveToFolder(String sessionId, String folder) async {
    if (_api == null) return;
    try {
      await _api!.moveSessionToFolder(sessionId, folder);
      await loadSessions();
      await loadFolders();
    } catch (e) {
      _log('[Sessions] moveToFolder error: $e');
      error = e.toString();
      notifyListeners();
    }
  }
}
