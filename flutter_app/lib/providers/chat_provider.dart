/// Chat state management.
///
/// Manages messages, streaming tokens, tool indicators, approval
/// requests, pipeline state, and canvas content.
library;

import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart'
    show ChangeNotifier, debugPrint, kDebugMode, kIsWeb;
import 'package:http/http.dart' as http;
import 'package:cognithor_ui/services/websocket_service.dart';

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

enum MessageRole { user, assistant, system }

class ChatMessage {
  ChatMessage({
    String? id,
    required this.role,
    required this.text,
    DateTime? timestamp,
    this.metadata = const {},
    this.agentName,
  }) : id =
           id ??
           'msg_${DateTime.now().millisecondsSinceEpoch}_${_msgCounter++}',
       timestamp = timestamp ?? DateTime.now();

  static int _msgCounter = 0;

  final String id;
  final MessageRole role;
  String text;
  final DateTime timestamp;
  Map<String, dynamic> metadata;

  /// Name of the agent that produced this message (for delegation visibility).
  final String? agentName;

  /// Version history for edit support (Claude-style).
  /// Each entry is a (userText, assistantText) pair.
  /// [versions.last] is the current version.
  final List<MessageVersion> versions = [];

  /// Current version index (0-based). -1 = no versioning.
  int activeVersion = -1;

  /// Tree node ID when conversation tree is active.
  String? treeNodeId;

  bool get hasVersions => versions.length > 1;
  int get versionCount => versions.length;
}

/// A single edit version: the user's text and the assistant's response.
class MessageVersion {
  MessageVersion({required this.userText, this.assistantText = ''});

  final String userText;
  String assistantText;
}

class ApprovalRequest {
  const ApprovalRequest({
    required this.requestId,
    required this.tool,
    required this.params,
    required this.reason,
  });

  final String requestId;
  final String tool;
  final Map<String, dynamic> params;
  final String reason;
}

class PipelinePhase {
  const PipelinePhase({
    required this.phase,
    required this.status,
    this.elapsedMs = 0,
  });

  final String phase;
  final String status;
  final int elapsedMs;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

void _log(String msg) {
  if (kDebugMode) debugPrint(msg);
}

class ChatProvider extends ChangeNotifier {
  ChatProvider({
    this.apiBaseUrl = 'http://localhost:8741',
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client();

  /// REST base URL used for media upload (e.g. `http://localhost:8741`).
  final String apiBaseUrl;

  /// HTTP client — injected for testability.
  final http.Client _http;

  WebSocketService? _ws;
  bool _listenersRegistered = false;

  // ---------------------------------------------------------------------------
  // Pending video attachment
  // ---------------------------------------------------------------------------

  Map<String, dynamic>? _pendingVideoAttachment;
  Map<String, dynamic>? get pendingVideoAttachment => _pendingVideoAttachment;

  /// Bind to a WebSocket service and register listeners.
  /// Safe to call multiple times — only registers once per WS instance.
  void attach(WebSocketService ws) {
    if (_ws == ws && _listenersRegistered) return;
    _ws = ws;
    _listenersRegistered = false;
    _registerListeners();
  }

  WebSocketService get ws => _ws!;

  final List<ChatMessage> messages = [];
  final StringBuffer _streamBuffer = StringBuffer();
  bool isStreaming = false;
  String? activeTool;
  String statusText = '';
  ApprovalRequest? pendingApproval;
  List<PipelinePhase> pipeline = [];
  String? canvasHtml;
  String? canvasTitle;
  Map<String, dynamic>? planDetail;
  final List<Map<String, dynamic>> agentLog = [];

  /// Pending feedback follow-up from the server (shown as dialog).
  Map<String, String>? pendingFeedbackFollowup;

  /// Pre-flight plan preview data from WebSocket.
  Map<String, dynamic>? preFlightData;

  /// Latest tree update from backend (conversation_id, node IDs).
  Map<String, dynamic>? lastTreeUpdate;

  void dismissPreFlight() {
    preFlightData = null;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  /// True between sendMessage() and receiving the first response/status.
  /// Used by the UI to show an immediate typing indicator.
  bool isWaitingForResponse = false;

  void sendMessage(String text) {
    _log('[Chat] sendMessage: "$text" (messages.length=${messages.length})');

    // Merge any pending video attachment into the WS metadata.
    final videoMeta = _pendingVideoAttachment;
    _pendingVideoAttachment = null;

    messages.add(
      ChatMessage(
        role: MessageRole.user,
        text: text,
        metadata: videoMeta != null
            ? {'video_attachment': videoMeta}
            : const {},
      ),
    );

    if (_ws != null) {
      _ws!.sendMessage(
        text,
        metadata: videoMeta != null ? {'video_attachment': videoMeta} : null,
      );
    } else {
      _log('[Chat] WARN: no WebSocket attached — message not sent');
    }
    statusText = '';
    isWaitingForResponse = true;
    _log('[Chat] notifyListeners (messages.length=${messages.length})');
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Video upload + URL-paste detection
  // ---------------------------------------------------------------------------

  /// Multipart-uploads [localPath] to `/api/media/upload` and stores the
  /// returned upload descriptor as [pendingVideoAttachment].
  /// On Flutter Web [localPath] is unused; bytes must come from FilePicker.
  Future<void> sendVideo(String localPath, String filename) async {
    final uri = Uri.parse('$apiBaseUrl/api/media/upload');

    late http.Response resp;
    if (kIsWeb) {
      // On web there is no dart:io File access; caller is expected to use
      // sendVideoBytes() instead. Throw early so callers notice.
      throw UnsupportedError(
        'sendVideo(localPath) is not supported on Flutter Web. '
        'Use sendVideoBytes() instead.',
      );
    } else {
      final bytes = await File(localPath).readAsBytes();
      final request = http.MultipartRequest('POST', uri)
        ..files.add(
          http.MultipartFile.fromBytes('file', bytes, filename: filename),
        );
      final streamed = await _http.send(request);
      resp = await http.Response.fromStream(streamed);
    }

    if (resp.statusCode != 200) {
      throw Exception('Upload failed: HTTP ${resp.statusCode} — ${resp.body}');
    }
    final body = jsonDecode(resp.body) as Map<String, dynamic>;

    _pendingVideoAttachment = {
      'kind': 'video',
      'uuid': body['uuid'],
      'url': body['url'],
      'filename': filename,
      'duration_sec': body['duration_sec'],
      'sampling': body['sampling'],
      'thumb_url': body['thumb_url'],
    };
    notifyListeners();
  }

  /// URL-paste detection — call from TextField onChanged.
  /// Returns true if the pasted text was consumed as a video URL attachment.
  bool handlePastedTextForVideoUrl(String text) {
    final pattern = RegExp(
      r'^\s*(https?://\S+?\.(?:mp4|webm|mov|mkv|avi)(?:[?#]\S*)?)\s*$',
      caseSensitive: false,
    );
    final m = pattern.firstMatch(text);
    if (m == null) return false;
    final rawUrl = m.group(1)!;
    // Filename is derived from the URL path only — strip query + fragment.
    final pathOnly = rawUrl.split('?').first.split('#').first;
    final derivedFilename = pathOnly.split('/').last;
    _pendingVideoAttachment = {
      'kind': 'video',
      'url': rawUrl, // keep full URL with query/fragment for fetching
      'filename': derivedFilename,
      'thumb_url': null,
    };
    notifyListeners();
    return true;
  }

  /// Clear any pending video attachment (e.g. user dismisses the preview).
  void clearPendingVideo() {
    _pendingVideoAttachment = null;
    notifyListeners();
  }

  /// Index of the user message currently being edited (for version tracking).
  int? _editingUserIndex;

  /// Edit a user message at [index]: save old version, update text,
  /// remove assistant responses after it, and resend.
  void editAndResend(int index, String newText) {
    if (index < 0 || index >= messages.length) return;
    _log('[Chat] editAndResend: index=$index newText="$newText"');

    final userMsg = messages[index];

    // Save old version (user text + assistant response) if not already versioned
    if (userMsg.versions.isEmpty) {
      // First edit: save the original as version 0
      String oldAssistantText = '';
      if (index + 1 < messages.length &&
          messages[index + 1].role == MessageRole.assistant) {
        oldAssistantText = messages[index + 1].text;
      }
      userMsg.versions.add(
        MessageVersion(userText: userMsg.text, assistantText: oldAssistantText),
      );
    }

    // Cancel any in-progress streaming
    isStreaming = false;
    _streamBuffer.clear();
    activeTool = null;
    statusText = '';
    pipeline = [];

    // Remove all messages after the user message (assistant responses etc.)
    if (index + 1 < messages.length) {
      messages.removeRange(index + 1, messages.length);
    }

    // Update user message text
    userMsg.text = newText;

    // Add new version (assistant text will be filled when response arrives)
    userMsg.versions.add(MessageVersion(userText: newText));
    userMsg.activeVersion = userMsg.versions.length - 1;

    // Track which message we're editing so we can attach the response
    _editingUserIndex = index;

    notifyListeners();

    // Send the new text via WebSocket
    if (_ws != null) {
      _ws!.sendMessage(newText);
    }
  }

  /// Switch to a different version of an edited message.
  /// Removes ALL messages after the user message, then inserts
  /// the version's assistant response (if any).
  void switchVersion(int messageIndex, int versionIndex) {
    if (messageIndex < 0 || messageIndex >= messages.length) return;
    final msg = messages[messageIndex];
    if (versionIndex < 0 || versionIndex >= msg.versions.length) return;

    final version = msg.versions[versionIndex];
    msg.text = version.userText;
    msg.activeVersion = versionIndex;

    // Remove everything after the user message
    if (messageIndex + 1 < messages.length) {
      messages.removeRange(messageIndex + 1, messages.length);
    }

    // Insert the version's assistant response (if it has one)
    if (version.assistantText.isNotEmpty) {
      messages.add(
        ChatMessage(role: MessageRole.assistant, text: version.assistantText),
      );
    }

    notifyListeners();
  }

  /// Retry the last assistant response: remove it and resend the
  /// last user message.
  void retryLastResponse() {
    // Cancel any in-progress streaming
    isStreaming = false;
    _streamBuffer.clear();
    activeTool = null;
    statusText = '';
    pipeline = [];
    // Find last assistant message and remove it
    for (var i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role == MessageRole.assistant) {
        messages.removeRange(i, messages.length);
        break;
      }
    }
    // Find last user message and resend
    for (var i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role == MessageRole.user) {
        final text = messages[i].text;
        _log('[Chat] retryLastResponse: resending "$text"');
        if (_ws != null) {
          _ws!.sendMessage(text);
        }
        notifyListeners();
        return;
      }
    }
  }

  void sendFile(String name, String type, String base64) {
    // Keep a local preview for images so the chat bubble can render a
    // thumbnail without re-downloading. Non-image files skip the payload.
    const imageExts = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'};
    final isImage = imageExts.contains(type.toLowerCase());
    final msg = ChatMessage(
      role: MessageRole.user,
      text: isImage ? '[Bild: $name]' : '[File: $name]',
      metadata: isImage
          ? {'image_base64': base64, 'image_name': name, 'image_type': type}
          : {},
    );
    messages.add(msg);
    _ws?.sendFile(name, type, base64);
    notifyListeners();
  }

  void sendAudio(String base64, {String mime = 'audio/webm'}) {
    messages.add(ChatMessage(role: MessageRole.user, text: '[Voice message]'));
    _ws?.sendAudio(base64, mimeType: mime);
    notifyListeners();
  }

  /// Last error shown to the user (e.g. approval send failed). UI can
  /// observe this and surface it. Cleared automatically on the next
  /// successful action.
  String? lastError;

  /// Called by the ApprovalDialog after a successful REST call to dismiss
  /// the dialog from the UI.
  void clearPendingApproval() {
    pendingApproval = null;
    lastError = null;
    notifyListeners();
  }

  Future<void> respondApproval(bool approved) async {
    if (pendingApproval == null) return;
    final requestId = pendingApproval!.requestId;
    _log(
      '[Chat] respondApproval CALLED: id=$requestId, approved=$approved, ws=${_ws != null}',
    );

    // ALWAYS use REST for approval — the WebSocket path has proven unreliable.
    // The REST endpoint directly resolves the pending future on the backend.
    if (_ws != null) {
      try {
        final resp = await _ws!.apiClient.post('approval_response', {
          'request_id': requestId,
          'approved': approved,
        });
        _log('[Chat] respondApproval REST response: $resp');
        if (resp['ok'] == true) {
          lastError = null;
          pendingApproval = null;
          notifyListeners();
          return;
        }
        _log('[Chat] respondApproval REST failed: ${resp['error']}');
      } catch (e) {
        _log('[Chat] respondApproval REST exception: $e');
      }
    }

    // REST failed — try WebSocket as last resort
    if (_ws == null) {
      _log(
        '[Chat] ERROR: respondApproval called but _ws is null! id=$requestId',
      );
      lastError = 'No connection to backend. Please check your connection.';
      notifyListeners();
      return;
    }
    _log('[Chat] respondApproval falling back to WS send: id=$requestId');
    var ok = _ws!.respondApproval(requestId, approved);

    // If the socket was not connected, try to reconnect once and retry.
    if (!ok) {
      _log('[Chat] respondApproval: initial send failed, attempting reconnect');
      try {
        final sid = _ws!.sessionId;
        if (sid != null) {
          await _ws!.connect(sid);
          // Give the connection a brief moment to establish + auth.
          await Future<void>.delayed(const Duration(milliseconds: 500));
          ok = _ws!.respondApproval(requestId, approved);
        }
      } catch (e) {
        _log('[Chat] respondApproval reconnect failed: $e');
      }
    }

    if (!ok) {
      _log('[Chat] respondApproval FINAL FAILURE: id=$requestId');
      lastError =
          'Approval could not be sent (connection lost). Please try again.';
      // Keep pendingApproval so the user can retry.
      notifyListeners();
      return;
    }

    _log('[Chat] respondApproval delivered: id=$requestId');
    lastError = null;
    pendingApproval = null;
    notifyListeners();
  }

  void cancelOperation() {
    _ws?.cancelOperation();
  }

  /// Send thumbs up/down feedback for a specific message.
  void sendFeedback(int rating, String messageId, String assistantResponse) {
    _ws?.sendFeedback(rating, messageId, assistantResponse: assistantResponse);
  }

  /// Send a follow-up comment for negative feedback.
  void sendFeedbackComment(String feedbackId, String comment) {
    _ws?.sendFeedbackComment(feedbackId, comment);
  }

  /// Dismiss the pending feedback follow-up dialog.
  void dismissFeedbackFollowup() {
    pendingFeedbackFollowup = null;
    notifyListeners();
  }

  void clearChat() {
    messages.clear();
    _streamBuffer.clear();
    isStreaming = false;
    activeTool = null;
    statusText = '';
    pendingApproval = null;
    pipeline = [];
    canvasHtml = null;
    canvasTitle = null;
    planDetail = null;
    preFlightData = null;
    agentLog.clear();
    _pendingVideoAttachment = null;
    notifyListeners();
  }

  /// Replace all messages with loaded history from API.
  void loadFromHistory(List<Map<String, dynamic>> historyMessages) {
    messages.clear();
    for (final msg in historyMessages) {
      final role = switch (msg['role']?.toString()) {
        'user' => MessageRole.user,
        'assistant' => MessageRole.assistant,
        _ => MessageRole.system,
      };
      messages.add(
        ChatMessage(role: role, text: msg['content']?.toString() ?? ''),
      );
    }
    notifyListeners();
  }

  /// Clear all state for a fresh session.
  void clearForNewSession() {
    messages.clear();
    _streamBuffer.clear();
    isStreaming = false;
    activeTool = null;
    statusText = '';
    pendingApproval = null;
    pipeline = [];
    canvasHtml = null;
    canvasTitle = null;
    planDetail = null;
    preFlightData = null;
    agentLog.clear();
    _pendingVideoAttachment = null;
    notifyListeners();
  }

  void dismissCanvas() {
    canvasHtml = null;
    canvasTitle = null;
    notifyListeners();
  }

  void dismissPlan() {
    planDetail = null;
    notifyListeners();
  }

  /// Get a tree-compatible node ID for a message at index.
  /// Returns null if tree is not active.
  String? getTreeNodeId(int messageIndex) {
    if (messageIndex < 0 || messageIndex >= messages.length) return null;
    return messages[messageIndex].treeNodeId;
  }

  // ---------------------------------------------------------------------------
  // Agent log helper
  // ---------------------------------------------------------------------------

  void _logAgent(
    String phase,
    String? tool,
    String message, {
    String status = 'active',
  }) {
    agentLog.add({
      'phase': phase,
      if (tool != null) 'tool': tool,
      'status': status,
      'message': message,
      'timestamp': DateTime.now().toIso8601String(),
    });
  }

  // ---------------------------------------------------------------------------
  // WebSocket listeners
  // ---------------------------------------------------------------------------

  void _registerListeners() {
    if (_ws == null || _listenersRegistered) return;
    _log('[Chat] Registering WS listeners');
    _ws!.on(WsType.assistantMessage, _onAssistantMessage);
    _ws!.on(WsType.streamToken, _onStreamToken);
    _ws!.on(WsType.streamEnd, _onStreamEnd);
    _ws!.on(WsType.toolStart, _onToolStart);
    _ws!.on(WsType.toolResult, _onToolResult);
    _ws!.on(WsType.approvalRequest, _onApprovalRequest);
    _ws!.on(WsType.statusUpdate, _onStatusUpdate);
    _ws!.on(WsType.pipelineEvent, _onPipelineEvent);
    _ws!.on(WsType.canvasPush, _onCanvasPush);
    _ws!.on(WsType.canvasReset, _onCanvasReset);
    _ws!.on(WsType.planDetail, _onPlanDetail);
    _ws!.on(WsType.transcription, _onTranscription);
    _ws!.on(WsType.error, _onError);
    _ws!.on(WsType.feedbackFollowup, _onFeedbackFollowup);
    _listenersRegistered = true;
  }

  void _onAssistantMessage(Map<String, dynamic> msg) {
    final text = msg['text'] as String? ?? '';
    _log(
      '[Chat] _onAssistantMessage: "${text.length > 100 ? '${text.substring(0, 100)}...' : text}"',
    );
    // If we were streaming, finalize the buffer instead.
    if (isStreaming) {
      _finalizeStream();
    }
    if (text.isNotEmpty) {
      final meta = msg['metadata'] as Map<String, dynamic>? ?? {};
      final agent =
          msg['agent_name'] as String? ?? meta['agent_name'] as String?;
      messages.add(
        ChatMessage(
          role: MessageRole.assistant,
          text: text,
          metadata: meta,
          agentName: agent,
        ),
      );

      // If this is a response to an edited message, store in version history
      if (_editingUserIndex != null &&
          _editingUserIndex! < messages.length - 1) {
        final userMsg = messages[_editingUserIndex!];
        if (userMsg.versions.isNotEmpty) {
          userMsg.versions.last.assistantText = text;
        }
        _editingUserIndex = null;
      }
    }
    _logAgent('complete', null, 'Response complete', status: 'done');
    activeTool = null;
    statusText = '';
    isWaitingForResponse = false;
    pipeline = [];
    _log('[Chat] notifyListeners (messages.length=${messages.length})');
    notifyListeners();
  }

  void _onStreamToken(Map<String, dynamic> msg) {
    final token = msg['token'] as String? ?? '';
    if (!isStreaming) {
      isStreaming = true;
      isWaitingForResponse = false;
      _streamBuffer.clear();
    }
    _streamBuffer.write(token);
    notifyListeners();
  }

  void _onStreamEnd(Map<String, dynamic> msg) {
    _finalizeStream();
    notifyListeners();
  }

  void _finalizeStream() {
    if (_streamBuffer.isNotEmpty) {
      messages.add(
        ChatMessage(
          role: MessageRole.assistant,
          text: _streamBuffer.toString(),
        ),
      );
      _streamBuffer.clear();
    }
    isStreaming = false;
  }

  /// The current partial streaming text (for display while streaming).
  String get streamingText => _streamBuffer.toString();

  void _onToolStart(Map<String, dynamic> msg) {
    activeTool = msg['tool'] as String?;
    agentLog.add({
      'phase': 'execute',
      'tool': activeTool ?? '',
      'message': 'Tool started: $activeTool',
      'timestamp': DateTime.now().toIso8601String(),
    });
    notifyListeners();
  }

  void _onToolResult(Map<String, dynamic> msg) {
    final result = msg['result']?.toString() ?? '';
    final summary = result.length > 80
        ? '${result.substring(0, 80)}...'
        : result;
    _logAgent('execute', activeTool, 'Tool result: $summary', status: 'done');
    activeTool = null;
    notifyListeners();
  }

  void _onApprovalRequest(Map<String, dynamic> msg) {
    final tool = msg['tool'] as String? ?? 'unknown';
    final reason = msg['reason'] as String? ?? '';
    pendingApproval = ApprovalRequest(
      requestId: msg['request_id'] as String? ?? '',
      tool: tool,
      params: msg['params'] as Map<String, dynamic>? ?? {},
      reason: reason,
    );
    _logAgent('gate', tool, 'Approval required: $reason', status: 'pending');
    notifyListeners();
  }

  void _onStatusUpdate(Map<String, dynamic> msg) {
    final type = msg['type'] as String? ?? msg['status_type'] as String? ?? '';
    final text = msg['text'] as String? ?? msg['status'] as String? ?? '';

    if (type == 'tree_update') {
      try {
        final data = json.decode(text) as Map<String, dynamic>;
        lastTreeUpdate = data;
      } catch (_) {
        // ignore parse errors
      }
      notifyListeners();
      return;
    }

    if (type == 'pre_flight') {
      try {
        preFlightData = json.decode(text) as Map<String, dynamic>;
      } catch (_) {
        preFlightData = {
          'goal': text,
          'steps': <Map<String, dynamic>>[],
          'timeout': 3,
        };
      }
      notifyListeners();
      return;
    }

    // Detect delegation status and add a system message for visibility
    if (text.startsWith('Delegation:')) {
      messages.add(
        ChatMessage(
          role: MessageRole.system,
          text: text,
          agentName: 'delegation',
        ),
      );
    }

    statusText = text;
    if (statusText.isNotEmpty) {
      isWaitingForResponse = false; // Backend responded, show status instead
      _logAgent('info', null, statusText);
    }
    notifyListeners();
  }

  void _onPipelineEvent(Map<String, dynamic> msg) {
    final phase = msg['phase'] as String? ?? '';
    final status = msg['status'] as String? ?? '';
    final elapsed = msg['elapsed_ms'] as int? ?? 0;
    pipeline = [
      ...pipeline.where((p) => p.phase != phase),
      PipelinePhase(phase: phase, status: status, elapsedMs: elapsed),
    ];
    agentLog.add({
      'phase': phase,
      'status': status,
      'message': '$phase: $status',
      'timestamp': DateTime.now().toIso8601String(),
    });
    notifyListeners();
  }

  void _onCanvasPush(Map<String, dynamic> msg) {
    canvasHtml = msg['html'] as String?;
    canvasTitle = msg['title'] as String?;
    notifyListeners();
  }

  void _onCanvasReset(Map<String, dynamic> msg) {
    canvasHtml = null;
    canvasTitle = null;
    notifyListeners();
  }

  void _onPlanDetail(Map<String, dynamic> msg) {
    planDetail = msg;
    notifyListeners();
  }

  void _onTranscription(Map<String, dynamic> msg) {
    final text = msg['text'] as String? ?? '';
    // Update the last user "[Voice message]" placeholder.
    for (var i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role == MessageRole.user &&
          messages[i].text == '[Voice message]') {
        messages[i].text = text;
        break;
      }
    }
    notifyListeners();
  }

  void _onError(Map<String, dynamic> msg) {
    final err = msg['error'] as String? ?? 'Unknown error';
    _log('[Chat] _onError: $err');
    messages.add(ChatMessage(role: MessageRole.system, text: err));
    _logAgent('error', null, err, status: 'error');
    isStreaming = false;
    _streamBuffer.clear();
    notifyListeners();
  }

  void _onFeedbackFollowup(Map<String, dynamic> msg) {
    final feedbackId = msg['feedback_id'] as String? ?? '';
    final question = msg['question'] as String? ?? '';
    _log('[Chat] _onFeedbackFollowup: feedbackId=$feedbackId');
    pendingFeedbackFollowup = {'feedback_id': feedbackId, 'question': question};
    notifyListeners();
  }
}
