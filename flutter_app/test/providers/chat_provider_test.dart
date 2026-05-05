import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockWsService extends Mock implements WebSocketService {}

/// Captures the listener map that ChatProvider registers via [WebSocketService.on],
/// so tests can drive inbound WS messages without a real socket.
class _ListenerCapture {
  final Map<String, List<WsMessageCallback>> listeners = {};

  void wireOn(_MockWsService ws) {
    when(() => ws.on(any(), any())).thenAnswer((inv) {
      final type = inv.positionalArguments[0] as String;
      final cb = inv.positionalArguments[1] as WsMessageCallback;
      listeners.putIfAbsent(type, () => []).add(cb);
    });
  }

  void emit(String type, Map<String, dynamic> msg) {
    for (final cb in listeners[type] ?? const <WsMessageCallback>[]) {
      cb(msg);
    }
  }
}

void main() {
  setUpAll(() {
    registerFallbackValue(<String, dynamic>{});
    registerFallbackValue((Map<String, dynamic> _) {});
  });

  group('ChatProvider', () {
    test('default state is empty / not streaming', () {
      final p = ChatProvider();
      expect(p.messages, isEmpty);
      expect(p.isStreaming, isFalse);
      expect(p.isWaitingForResponse, isFalse);
      expect(p.activeTool, isNull);
      expect(p.statusText, isEmpty);
      expect(p.pendingApproval, isNull);
      expect(p.pipeline, isEmpty);
      expect(p.canvasHtml, isNull);
      expect(p.canvasTitle, isNull);
      expect(p.planDetail, isNull);
      expect(p.preFlightData, isNull);
      expect(p.pendingVideoAttachment, isNull);
      expect(p.pendingFeedbackFollowup, isNull);
      expect(p.streamingText, isEmpty);
      expect(p.lastError, isNull);
      expect(p.agentLog, isEmpty);
    });

    test('sendMessage without WS appends a user message and notifies', () {
      final p = ChatProvider();
      var notified = 0;
      p.addListener(() => notified++);

      p.sendMessage('hello');

      expect(p.messages.length, 1);
      expect(p.messages.single.role, MessageRole.user);
      expect(p.messages.single.text, 'hello');
      expect(p.isWaitingForResponse, isTrue);
      expect(notified, 1);
    });

    test(
      'sendMessage forwards the pending video attachment to WS metadata',
      () {
        final ws = _MockWsService();
        final cap = _ListenerCapture()..wireOn(ws);
        // Silence other on-registrations.
        cap;
        when(
          () => ws.sendMessage(any(), metadata: any(named: 'metadata')),
        ).thenReturn(null);
        final p = ChatProvider()..attach(ws);

        // Simulate URL-paste of an mp4 → pendingVideoAttachment populated.
        p.handlePastedTextForVideoUrl('https://x.com/clip.mp4');
        p.sendMessage('look at this');

        // Captured by mock:
        final captured = verify(
          () => ws.sendMessage(
            captureAny(),
            metadata: captureAny(named: 'metadata'),
          ),
        ).captured;
        expect(captured.first, 'look at this');
        final meta = captured.last as Map<String, dynamic>?;
        expect(meta, isNotNull);
        expect(meta!['video_attachment'], isNotNull);
        expect(p.pendingVideoAttachment, isNull); // consumed
      },
    );

    test('clearChat resets every observable field', () {
      final p = ChatProvider();
      p.sendMessage('first');
      p.handlePastedTextForVideoUrl('https://x.com/a.mp4');

      p.clearChat();

      expect(p.messages, isEmpty);
      expect(p.pendingVideoAttachment, isNull);
      expect(p.statusText, isEmpty);
      expect(p.pipeline, isEmpty);
      expect(p.canvasHtml, isNull);
      expect(p.planDetail, isNull);
      expect(p.preFlightData, isNull);
      expect(p.agentLog, isEmpty);
    });

    test('loadFromHistory replaces messages and parses roles', () {
      final p = ChatProvider();
      p.loadFromHistory([
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
        {'role': 'system', 'content': 'note'},
        {'role': 'unknown', 'content': 'fallback'}, // → system
      ]);

      expect(p.messages.length, 4);
      expect(p.messages[0].role, MessageRole.user);
      expect(p.messages[1].role, MessageRole.assistant);
      expect(p.messages[2].role, MessageRole.system);
      expect(p.messages[3].role, MessageRole.system);
    });

    test('clearPendingVideo nulls the pending attachment', () {
      final p = ChatProvider();
      p.handlePastedTextForVideoUrl('https://x.com/a.mp4');
      expect(p.pendingVideoAttachment, isNotNull);
      p.clearPendingVideo();
      expect(p.pendingVideoAttachment, isNull);
    });

    test(
      'dismissCanvas / dismissPlan / dismissPreFlight clear their fields',
      () {
        final ws = _MockWsService();
        final cap = _ListenerCapture()..wireOn(ws);
        final p = ChatProvider()..attach(ws);

        // Drive plan/canvas/preflight via inbound WS messages.
        cap.emit(WsType.canvasPush, {'html': '<b>x</b>', 'title': 'demo'});
        cap.emit(WsType.planDetail, {'goal': 'g'});
        cap.emit(WsType.statusUpdate, {
          'type': 'pre_flight',
          'text': '{"goal":"x","steps":[],"timeout":3}',
        });

        expect(p.canvasHtml, '<b>x</b>');
        expect(p.canvasTitle, 'demo');
        expect(p.planDetail, isNotNull);
        expect(p.preFlightData, isNotNull);

        p.dismissCanvas();
        p.dismissPlan();
        p.dismissPreFlight();

        expect(p.canvasHtml, isNull);
        expect(p.canvasTitle, isNull);
        expect(p.planDetail, isNull);
        expect(p.preFlightData, isNull);
      },
    );

    test('streamToken → streamEnd produces a single assistant message', () {
      final ws = _MockWsService();
      final cap = _ListenerCapture()..wireOn(ws);
      final p = ChatProvider()..attach(ws);

      cap.emit(WsType.streamToken, {'token': 'Hel'});
      cap.emit(WsType.streamToken, {'token': 'lo!'});
      expect(p.isStreaming, isTrue);
      expect(p.streamingText, 'Hello!');

      cap.emit(WsType.streamEnd, {});
      expect(p.isStreaming, isFalse);
      expect(p.messages.last.role, MessageRole.assistant);
      expect(p.messages.last.text, 'Hello!');
      expect(p.streamingText, isEmpty);
    });

    test('assistantMessage handler appends + clears waiting state', () {
      final ws = _MockWsService();
      final cap = _ListenerCapture()..wireOn(ws);
      final p = ChatProvider()..attach(ws);
      p.isWaitingForResponse = true;

      cap.emit(WsType.assistantMessage, {
        'text': 'response text',
        'metadata': {'k': 'v'},
        'agent_name': 'planner',
      });

      expect(p.messages.last.role, MessageRole.assistant);
      expect(p.messages.last.text, 'response text');
      expect(p.messages.last.agentName, 'planner');
      expect(p.isWaitingForResponse, isFalse);
    });

    test('error message is appended as system message + streaming reset', () {
      final ws = _MockWsService();
      final cap = _ListenerCapture()..wireOn(ws);
      final p = ChatProvider()..attach(ws);

      cap.emit(WsType.streamToken, {'token': 'partial'});
      cap.emit(WsType.error, {'error': 'boom'});

      expect(p.isStreaming, isFalse);
      expect(p.messages.last.role, MessageRole.system);
      expect(p.messages.last.text, 'boom');
    });

    test('toolStart + toolResult tracks activeTool', () {
      final ws = _MockWsService();
      final cap = _ListenerCapture()..wireOn(ws);
      final p = ChatProvider()..attach(ws);

      cap.emit(WsType.toolStart, {'tool': 'web.search'});
      expect(p.activeTool, 'web.search');

      cap.emit(WsType.toolResult, {'result': 'ok'});
      expect(p.activeTool, isNull);
    });

    test(
      'respondApproval uses REST and clears pendingApproval on ok',
      () async {
        final ws = _MockWsService();
        final api = _MockApiClient();
        when(() => ws.apiClient).thenReturn(api);
        final cap = _ListenerCapture()..wireOn(ws);
        final p = ChatProvider()..attach(ws);

        cap.emit(WsType.approvalRequest, {
          'request_id': 'r1',
          'tool': 'fs.write',
          'reason': 'risky',
          'params': {'path': '/tmp/x'},
        });
        expect(p.pendingApproval, isNotNull);

        when(
          () => api.post(any(), any()),
        ).thenAnswer((_) async => {'ok': true});

        await p.respondApproval(true);

        expect(p.pendingApproval, isNull);
        expect(p.lastError, isNull);
      },
    );

    test('retryLastResponse pops last assistant + resends last user text', () {
      final ws = _MockWsService();
      final cap = _ListenerCapture()..wireOn(ws);
      when(
        () => ws.sendMessage(any(), metadata: any(named: 'metadata')),
      ).thenReturn(null);
      final p = ChatProvider()..attach(ws);

      p.sendMessage('first user');
      cap.emit(WsType.assistantMessage, {'text': 'first answer'});
      expect(p.messages.length, 2);

      p.retryLastResponse();
      // Assistant removed, user kept.
      expect(p.messages.length, 1);
      expect(p.messages.last.role, MessageRole.user);
      // ws.sendMessage was called twice in total: once via sendMessage(), once via retry.
      final all = verify(
        () => ws.sendMessage(captureAny(), metadata: any(named: 'metadata')),
      ).captured;
      expect(all.length, 2);
      expect(all, ['first user', 'first user']);
    });
  });
}
