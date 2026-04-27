/// Cognithor WebSocket service.
///
/// Implements the full 21-message-type protocol defined in
/// FLUTTER_API_CONTRACT.md.  Handles auth handshake, heartbeat,
/// reconnection with exponential back-off, and message dispatch.
library;

import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
import 'package:web_socket_channel/web_socket_channel.dart';

import 'package:cognithor_ui/services/api_client.dart';

void _log(String msg) {
  if (kDebugMode) debugPrint(msg);
}

// ---------------------------------------------------------------------------
// Message types (mirrors backend WSMessageType)
// ---------------------------------------------------------------------------

abstract final class WsType {
  // Client → Server
  static const auth = 'auth';
  static const userMessage = 'user_message';
  static const approvalResponse = 'approval_response';
  static const ping = 'ping';
  static const cancel = 'cancel';

  // Server → Client
  static const assistantMessage = 'assistant_message';
  static const streamToken = 'stream_token';
  static const streamEnd = 'stream_end';
  static const toolStart = 'tool_start';
  static const toolResult = 'tool_result';
  static const approvalRequest = 'approval_request';
  static const statusUpdate = 'status_update';
  static const pipelineEvent = 'pipeline_event';
  static const planDetail = 'plan_detail';
  static const canvasPush = 'canvas_push';
  static const canvasReset = 'canvas_reset';
  static const canvasEval = 'canvas_eval';
  static const transcription = 'transcription';
  static const error = 'error';
  static const pong = 'pong';
  static const identityState = 'identity_state';

  // Feedback
  static const feedback = 'feedback';
  static const feedbackComment = 'feedback_comment';
  static const feedbackAck = 'feedback_ack';
  static const feedbackFollowup = 'feedback_followup';

  // Kanban
  static const kanbanUpdate = 'kanban_update';

  // Crew / Trace
  static const String crewLifecycle = 'crew_lifecycle';
  static const String crewEvent = 'crew_event';
  static const String crewLifecycleSubscribe = 'crew_lifecycle_subscribe';
  static const String crewSubscribe = 'crew_subscribe';
  static const String crewUnsubscribe = 'crew_unsubscribe';
}

// ---------------------------------------------------------------------------
// Callback typedefs
// ---------------------------------------------------------------------------

typedef WsMessageCallback = void Function(Map<String, dynamic> message);

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

class WebSocketService {
  WebSocketService({required this.apiClient, required this.wsBaseUrl});

  final ApiClient apiClient;

  /// ws:// or wss:// base (e.g. `ws://localhost:8741`).
  final String wsBaseUrl;

  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _subscription;
  Timer? _heartbeat;
  Timer? _reconnectTimer;
  int _retries = 0;
  String? _sessionId;
  bool _disposed = false;

  static const _maxRetries = 10;
  static const _heartbeatInterval = Duration(seconds: 30);
  static const _baseReconnectDelay = Duration(seconds: 3);

  /// True when the WebSocket is connected and authenticated.
  bool get isConnected => _channel != null;
  String? get sessionId => _sessionId;

  // Per-type callback map.
  final Map<String, List<WsMessageCallback>> _listeners = {};

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Register a listener for a specific message type.
  void on(String type, WsMessageCallback cb) {
    _listeners.putIfAbsent(type, () => []).add(cb);
  }

  /// Remove a specific listener.
  void off(String type, WsMessageCallback cb) {
    _listeners[type]?.remove(cb);
  }

  /// Connect to the backend WebSocket.
  Future<void> connect(String sessionId) async {
    _sessionId = sessionId;
    _disposed = false;
    await _doConnect();
  }

  /// Disconnect and reconnect with a different session.
  Future<void> switchSession(String newSessionId) async {
    disconnect();
    // Brief delay to let the old connection close cleanly
    // before opening a new one (avoids ASGI race condition).
    await Future<void>.delayed(const Duration(milliseconds: 300));
    _disposed = false;
    await connect(newSessionId);
  }

  /// Disconnect gracefully.
  void disconnect() {
    _disposed = true;
    _heartbeat?.cancel();
    _reconnectTimer?.cancel();
    _subscription?.cancel();
    _channel?.sink.close();
    _channel = null;
  }

  /// Send a text message.
  void sendMessage(String text, {Map<String, dynamic>? metadata}) {
    _send({
      'type': WsType.userMessage,
      'text': text,
      'session_id': _sessionId,
      if (metadata != null) 'metadata': metadata,
    });
  }

  /// Send a file as base64.
  void sendFile(String fileName, String fileType, String base64Data) {
    _send({
      'type': WsType.userMessage,
      'text': '[File: $fileName]',
      'session_id': _sessionId,
      'metadata': {
        'file_name': fileName,
        'file_type': fileType,
        'file_base64': base64Data,
      },
    });
  }

  /// Send a voice recording as base64.
  void sendAudio(String base64Audio, {String mimeType = 'audio/webm'}) {
    _send({
      'type': WsType.userMessage,
      'text': '[Voice message]',
      'session_id': _sessionId,
      'metadata': {'audio_base64': base64Audio, 'audio_type': mimeType},
    });
  }

  /// Respond to an approval request. Returns `true` if the message was
  /// actually written to the socket, `false` if the send was dropped
  /// (e.g. channel not connected).
  bool respondApproval(String requestId, bool approved) {
    final ok = _send({
      'type': WsType.approvalResponse,
      'id': requestId,
      'approved': approved,
      'session_id': _sessionId,
    });
    if (ok) {
      _log('[WS] approval_response sent: id=$requestId approved=$approved');
    } else {
      _log(
        '[WS] approval_response DROPPED: id=$requestId approved=$approved (channel null)',
      );
    }
    return ok;
  }

  /// Cancel the current operation.
  void cancelOperation() {
    _send({'type': WsType.cancel, 'session_id': _sessionId});
  }

  /// Send thumbs up/down feedback for a message.
  void sendFeedback(
    int rating,
    String messageId, {
    String assistantResponse = '',
  }) {
    _send({
      'type': WsType.feedback,
      'rating': rating,
      'message_id': messageId,
      'session_id': _sessionId,
      'assistant_response': assistantResponse,
    });
  }

  /// Send a follow-up comment for previous negative feedback.
  void sendFeedbackComment(String feedbackId, String comment) {
    _send({
      'type': WsType.feedbackComment,
      'feedback_id': feedbackId,
      'comment': comment,
    });
  }

  /// Send an arbitrary outbound frame. Used by feature services
  /// (e.g. TraceService) to dispatch their own message types.
  /// Returns `true` if written to the socket, `false` if dropped (no channel).
  bool send(Map<String, dynamic> frame) {
    return _send({...frame, 'session_id': _sessionId});
  }

  // ---------------------------------------------------------------------------
  // Connection internals
  // ---------------------------------------------------------------------------

  Future<void> _doConnect() async {
    if (_disposed) return;

    // Always fetch a fresh token before connecting.
    // Retry bootstrap up to 3 times with 1s delay if server is still starting.
    String? token;
    for (var attempt = 0; attempt < 3; attempt++) {
      token = await apiClient.bootstrap();
      if (token != null) break;
      await Future<void>.delayed(const Duration(seconds: 1));
    }
    if (token == null) {
      _scheduleReconnect();
      return;
    }

    final uri = Uri.parse('$wsBaseUrl/ws/$_sessionId');
    _log('[WS] Connecting to $uri ...');
    try {
      _channel = WebSocketChannel.connect(uri);
      await _channel!.ready;
      _log('[WS] Connected successfully');
    } catch (e) {
      _log('[WS] Connection failed: $e');
      _channel = null;
      _scheduleReconnect();
      return;
    }

    // Send auth as first message.
    _send({'type': WsType.auth, 'token': token});

    _retries = 0;
    _startHeartbeat();

    _subscription = _channel!.stream.listen(
      _onMessage,
      onDone: _onDone,
      onError: (_) => _onDone(),
    );
  }

  void _onMessage(dynamic raw) {
    if (raw is! String) return;
    final Map<String, dynamic> msg;
    try {
      msg = jsonDecode(raw) as Map<String, dynamic>;
    } catch (e) {
      _log('[WS] JSON decode error: $e');
      return;
    }

    final type = msg['type'] as String?;
    if (type == null) return;

    // Handle pong silently.
    if (type == WsType.pong) return;

    _log('[WS] ← $type');

    // On auth error, invalidate token so reconnect fetches a fresh one.
    if (type == WsType.error &&
        (msg['error'] as String? ?? '').contains('Unauthorized')) {
      apiClient.invalidateToken();
    }

    // Dispatch to registered listeners.
    final cbs = _listeners[type];
    if (cbs != null) {
      for (final cb in List<WsMessageCallback>.of(cbs)) {
        try {
          cb(msg);
        } catch (e, st) {
          _log('[WS] Listener error for $type: $e\n$st');
        }
      }
    } else {
      _log('[WS] No listeners for type: $type');
    }
  }

  void _onDone() {
    _log('[WS] Connection closed (disposed=$_disposed)');
    _heartbeat?.cancel();
    _subscription?.cancel();
    _channel = null;
    if (!_disposed) _scheduleReconnect();
  }

  void _scheduleReconnect() {
    if (_disposed || _retries >= _maxRetries) return;
    _retries++;
    final delay = _baseReconnectDelay * _retries.clamp(1, 5);
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(delay, _doConnect);
  }

  void _startHeartbeat() {
    _heartbeat?.cancel();
    _heartbeat = Timer.periodic(_heartbeatInterval, (_) {
      _send({'type': WsType.ping});
    });
  }

  /// Write a message to the socket.
  ///
  /// Returns `true` on success, `false` if the channel is not connected
  /// or the write threw. Callers that care about delivery (e.g. the
  /// approval_response path) must check the return value and retry.
  bool _send(Map<String, dynamic> msg) {
    if (_channel == null) {
      _log(
        '[WS] WARN: _send called but _channel is null (type=${msg['type']})',
      );
      return false;
    }
    final type = msg['type'] as String? ?? '?';
    if (type != WsType.ping) {
      _log('[WS] → $type');
    }
    try {
      _channel!.sink.add(jsonEncode(msg));
      return true;
    } catch (e) {
      _log('[WS] _send failed for type=$type: $e');
      return false;
    }
  }
}
