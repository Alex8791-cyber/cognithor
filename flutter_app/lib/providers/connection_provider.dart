/// Connection state — manages backend URL, health, and connectivity.
library;

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

enum CognithorConnectionState { disconnected, connecting, connected, error }

/// Frontend version — must match backend `__version__` (major.minor).
/// See issue #111: version mismatch between Flutter and Python backend
/// should block entry to the app.
const String kFrontendVersion = '0.92.4';

/// Extracts "major.minor" from a semver-ish string like "0.91.0" or "0.91.0+1".
String? _majorMinor(String? v) {
  if (v == null || v.isEmpty) return null;
  final core = v.split('+').first.split('-').first;
  final parts = core.split('.');
  if (parts.length < 2) return null;
  return '${parts[0]}.${parts[1]}';
}

class ConnectionProvider extends ChangeNotifier {
  ConnectionProvider();

  static const _serverUrlKey = 'jarvis_server_url';

  /// Default URL: on web, use the host the page was loaded from.
  /// On native apps, default to localhost.
  static String get _defaultUrl {
    try {
      // ignore: avoid_web_libraries_in_flutter
      final uri = Uri.base; // works on web: gives the page URL
      if (uri.host.isNotEmpty && uri.host != 'localhost' && uri.host != '127.0.0.1') {
        return '${uri.scheme}://${uri.host}:${uri.port}';
      }
    } catch (_) {}
    return 'http://localhost:8741';
  }

  CognithorConnectionState state = CognithorConnectionState.disconnected;
  String serverUrl = _defaultUrl;
  String? errorMessage;
  String? backendVersion;

  /// True when backend major.minor does not match [kFrontendVersion] major.minor.
  /// When true, [state] is forced to [CognithorConnectionState.error] and the
  /// app refuses to enter the main shell (see SplashScreen / ConnectionGuard).
  bool versionMismatch = false;

  /// Frontend version exposed for UI display.
  String get frontendVersion => kFrontendVersion;

  /// Whether initial connection has ever succeeded (guards health polling).
  bool _wasConnected = false;

  /// Public accessor so [ConnectionGuard] can distinguish initial vs. lost.
  bool get wasConnected => _wasConnected;

  Timer? _healthTimer;

  ApiClient? _api;
  WebSocketService? _ws;

  ApiClient get api => _api!;
  WebSocketService get ws => _ws!;

  /// Load saved server URL and connect.
  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    serverUrl = prefs.getString(_serverUrlKey) ?? _defaultUrl;
    await connect();
  }

  /// Change server URL and reconnect.
  Future<void> setServerUrl(String url) async {
    final clean = url.trimRight().replaceAll(RegExp(r'/+$'), '');
    if (clean == serverUrl) return;
    serverUrl = clean;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_serverUrlKey, serverUrl);
    await connect();
  }

  /// Connect to the backend (health check + bootstrap).
  Future<void> connect() async {
    state = CognithorConnectionState.connecting;
    errorMessage = null;
    versionMismatch = false;
    notifyListeners();

    _ws?.disconnect();
    _api = ApiClient(baseUrl: serverUrl);

    try {
      // Health check with 10s timeout
      final health = await _api!.get('/health').timeout(
        const Duration(seconds: 10),
        onTimeout: () =>
            throw TimeoutException('Backend nicht erreichbar ($serverUrl)'),
      );
      if (health.containsKey('error')) {
        throw Exception(health['error']);
      }
      backendVersion = health['version'] as String?;

      // Version compatibility check (major.minor only, patch allowed to differ).
      // If backend doesn't report a version, assume OK (fallback).
      final feMM = _majorMinor(kFrontendVersion);
      final beMM = _majorMinor(backendVersion);
      if (beMM != null && feMM != null && beMM != feMM) {
        versionMismatch = true;
        state = CognithorConnectionState.error;
        errorMessage =
            'Version mismatch: Frontend v$kFrontendVersion, Backend v$backendVersion. '
            'Please update one side.';
        notifyListeners();
        return;
      }
      versionMismatch = false;

      // Bootstrap token
      final token = await _api!.bootstrap();
      if (token == null) {
        throw Exception('Bootstrap fehlgeschlagen - kein Token erhalten');
      }

      // WebSocket
      final wsUrl = serverUrl
          .replaceFirst('https://', 'wss://')
          .replaceFirst('http://', 'ws://');
      _ws = WebSocketService(apiClient: _api!, wsBaseUrl: wsUrl);

      state = CognithorConnectionState.connected;
      _wasConnected = true;
      _startHealthPolling();
    } on TimeoutException catch (e) {
      state = CognithorConnectionState.error;
      errorMessage = e.message ?? 'Backend nicht erreichbar ($serverUrl)';
    } catch (e) {
      state = CognithorConnectionState.error;
      errorMessage = 'Backend nicht erreichbar ($serverUrl)';
    }
    notifyListeners();
  }

  /// Starts periodic health checks (every 15s).
  void _startHealthPolling() {
    _healthTimer?.cancel();
    _healthTimer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => _checkHealth(),
    );
  }

  /// Pings /health and transitions state on failure/recovery.
  Future<void> _checkHealth() async {
    if (_api == null) return;
    try {
      final resp = await _api!.get('/health').timeout(
            const Duration(seconds: 5),
          );
      if (resp['status'] != 'ok' &&
          state == CognithorConnectionState.connected) {
        state = CognithorConnectionState.error;
        errorMessage = 'Backend health check failed';
        notifyListeners();
        _scheduleReconnect();
      } else if (resp['status'] == 'ok' &&
          state != CognithorConnectionState.connected) {
        state = CognithorConnectionState.connected;
        errorMessage = null;
        notifyListeners();
      }
    } catch (_) {
      if (state == CognithorConnectionState.connected) {
        state = CognithorConnectionState.error;
        errorMessage = 'Backend nicht erreichbar';
        notifyListeners();
        _scheduleReconnect();
      }
    }
  }

  /// Schedules a reconnection attempt after 5 seconds.
  void _scheduleReconnect() {
    Future.delayed(const Duration(seconds: 5), () {
      if (state != CognithorConnectionState.connected) {
        connect();
      }
    });
  }

  @override
  void dispose() {
    _healthTimer?.cancel();
    _ws?.disconnect();
    super.dispose();
  }
}
