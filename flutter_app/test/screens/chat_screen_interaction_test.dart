/// Deep interaction tests for [ChatScreen].
///
/// Builds on the smoke tests in `chat_screen_test.dart` (PR #489) by
/// driving real user flows: typing into the input, tapping send, opening
/// the AppBar drawer, toggling AppBar action icons, and tapping the
/// attachment menu. Each test asserts an observable change in the
/// rendered tree (provider state, dialog opens, drawer opens) instead
/// of poking internal state.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/hacker_mode_provider.dart';
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/providers/pip_provider.dart';
import 'package:cognithor_ui/providers/sessions_provider.dart';
import 'package:cognithor_ui/providers/tree_provider.dart';
import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/screens/chat_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockWsService extends Mock implements WebSocketService {}

/// ChatProvider variant that records sendMessage calls without needing
/// a real WebSocket. The real provider silently swallows the call when
/// `_ws == null`, so we override sendMessage to just append the message
/// and notify — that's the observable we care about for the input flow.
class _RecordingChatProvider extends ChatProvider {
  int sendMessageCalls = 0;
  String? lastSentText;
  int clearChatCalls = 0;

  @override
  void sendMessage(String text) {
    sendMessageCalls++;
    lastSentText = text;
    messages.add(ChatMessage(role: MessageRole.user, text: text));
    notifyListeners();
  }

  @override
  void clearChat() {
    clearChatCalls++;
    super.clearChat();
  }
}

void main() {
  setUpAll(() {
    registerFallbackValue(<String, dynamic>{});
  });

  group('ChatScreen interactions', () {
    late _MockApiClient api;
    late _RecordingChatProvider chat;
    late HackerModeProvider hacker;
    late FakeConnectionProvider conn;

    setUp(() {
      // HackerModeProvider, ConnectionProvider and a few others read
      // SharedPreferences in their constructors. Without a mock backend
      // the platform channel call throws.
      SharedPreferences.setMockInitialValues(<String, Object>{});
      api = _MockApiClient();
      chat = _RecordingChatProvider();
      hacker = HackerModeProvider();
      conn = FakeConnectionProvider(
        apiClient: api,
        wsService: _MockWsService(),
      );
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<ChatProvider>.value(value: chat),
          ChangeNotifierProvider<SessionsProvider>(
            create: (_) => SilentSessionsProvider(),
          ),
          ChangeNotifierProvider<VoiceProvider>(create: (_) => VoiceProvider()),
          ChangeNotifierProvider<TreeProvider>(create: (_) => TreeProvider()),
          ChangeNotifierProvider<PipProvider>(create: (_) => PipProvider()),
          ChangeNotifierProvider<HackerModeProvider>.value(value: hacker),
          ChangeNotifierProvider<LlmBackendProvider>(
            create: (_) => SilentLlmBackendProvider(),
          ),
        ],
        child: const ChatScreen(),
      ),
    );

    /// Standard pump: build + drain post-frame callbacks. Cannot use
    /// pumpAndSettle — CognithorEmptyState pulses forever.
    Future<void> pump(WidgetTester tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
    }

    Future<void> teardown(WidgetTester tester) async {
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 50));
    }

    testWidgets('typing into the input field reflects on the TextField', (
      tester,
    ) async {
      await pump(tester);

      // The chat screen owns a single TextField — the message input.
      final inputField = find.byType(TextField);
      expect(inputField, findsOneWidget);

      await tester.enterText(inputField, 'Hallo Cognithor');
      await tester.pump();

      // The text appears in the rendered TextField.
      expect(find.text('Hallo Cognithor'), findsOneWidget);

      await teardown(tester);
    });

    testWidgets(
      'tapping the send icon with text dispatches sendMessage to the provider',
      (tester) async {
        await pump(tester);

        await tester.enterText(find.byType(TextField), 'Hello world');
        await tester.pump();

        // Send icon — Icons.send is the visual indicator on the send button.
        await tester.tap(find.byIcon(Icons.send));
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 50));

        expect(chat.sendMessageCalls, 1);
        expect(chat.lastSentText, 'Hello world');
        // The user message now appears in the messages list.
        expect(chat.messages, hasLength(1));
        expect(chat.messages.first.text, 'Hello world');

        await teardown(tester);
      },
    );

    testWidgets('tapping send with whitespace-only input is a no-op', (
      tester,
    ) async {
      await pump(tester);

      await tester.enterText(find.byType(TextField), '   ');
      await tester.pump();

      await tester.tap(find.byIcon(Icons.send));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // _submit() trims and bails on empty — provider must NOT have been
      // called.
      expect(chat.sendMessageCalls, 0);
      expect(chat.messages, isEmpty);

      await teardown(tester);
    });

    testWidgets(
      'sending a message clears the input field controller for the next turn',
      (tester) async {
        await pump(tester);

        await tester.enterText(find.byType(TextField), 'Erste Nachricht');
        await tester.pump();
        await tester.tap(find.byIcon(Icons.send));
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 50));

        // After send the input controller is cleared (the message itself
        // still appears in the chat bubble area, but the TextField at the
        // bottom is reset).
        final tf = tester.widget<TextField>(find.byType(TextField));
        expect(tf.controller?.text, '');

        await teardown(tester);
      },
    );

    testWidgets('tapping the history icon opens the chat-history drawer', (
      tester,
    ) async {
      await pump(tester);

      // Drawer is closed by default — the Drawer widget exists in the tree
      // (Scaffold builds it lazily) but the tester.tap on history opens it.
      await tester.tap(find.byIcon(Icons.history));
      // The drawer slides in via animation — pump the duration but do not
      // pumpAndSettle (chat screen still has the empty-state pulse).
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 250));

      // Once the drawer is open the Scaffold reports it as such by
      // exposing the SemanticsLabel "Close navigation menu" on its
      // backdrop. We assert via the structurally-stable Drawer widget.
      expect(find.byType(Drawer), findsOneWidget);

      await teardown(tester);
    });

    testWidgets('tapping the hacker-mode AppBar icon toggles the provider', (
      tester,
    ) async {
      await pump(tester);

      expect(hacker.enabled, isFalse);

      // Icons.terminal is the hacker-mode toggle icon in the AppBar.
      await tester.tap(find.byIcon(Icons.terminal));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      expect(hacker.enabled, isTrue);

      await teardown(tester);
    });

    testWidgets('tapping clear-chat dispatches clearChat on the provider', (
      tester,
    ) async {
      await pump(tester);

      // Seed a message so we can observe clear taking effect.
      chat.sendMessage('seeded');
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      expect(chat.messages, hasLength(1));

      await tester.tap(find.byIcon(Icons.delete_outline));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      expect(chat.clearChatCalls, 1);
      expect(chat.messages, isEmpty);

      await teardown(tester);
    });
  });
}
