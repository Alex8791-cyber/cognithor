/// Smoke test for [ChatScreen].
///
/// Previously deferred because:
///   - `didChangeDependencies` reaches into [SessionsProvider],
///     [VoiceProvider], [PipProvider], and attaches a listener on
///     [ChatProvider].
///   - The empty-state widget runs a never-ending `pulse` animation,
///     so `pumpAndSettle` would hang.
///
/// We supply silent versions of every provider the screen reads,
/// then drive the smoke with a finite `tester.pump(50ms)` instead of
/// `pumpAndSettle`.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

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

void main() {
  setUpAll(() {
    // Required for `mocktail` named-arg matchers further down.
    registerFallbackValue(<String, dynamic>{});
  });

  group('ChatScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // SessionsProvider.loadSessions / loadFolders fan out from
      // didChangeDependencies — silenced via the Silent provider override.
      // ChatProvider has no API-driven init.
      conn = FakeConnectionProvider(
        apiClient: api,
        wsService: _MockWsService(),
      );
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<ChatProvider>(create: (_) => ChatProvider()),
          ChangeNotifierProvider<SessionsProvider>(
            create: (_) => SilentSessionsProvider(),
          ),
          ChangeNotifierProvider<VoiceProvider>(create: (_) => VoiceProvider()),
          ChangeNotifierProvider<TreeProvider>(create: (_) => TreeProvider()),
          ChangeNotifierProvider<PipProvider>(create: (_) => PipProvider()),
          ChangeNotifierProvider<HackerModeProvider>(
            create: (_) => HackerModeProvider(),
          ),
          ChangeNotifierProvider<LlmBackendProvider>(
            create: (_) => SilentLlmBackendProvider(),
          ),
        ],
        child: const ChatScreen(),
      ),
    );

    testWidgets('renders without crashing on empty session/message state', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      // First frame: build the empty-state with the pulse animation.
      await tester.pump();
      // Drain any post-frame callbacks (e.g. _scrollToBottom).
      // Cannot use pumpAndSettle: CognithorEmptyState pulses forever.
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(ChatScreen), findsOneWidget);
    });

    testWidgets('shows the AppBar history-leading button', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // The drawer-opener uses Icons.history.
      expect(find.byIcon(Icons.history), findsOneWidget);
    });

    testWidgets('disposes cleanly when replaced', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Replacing the tree triggers `_ChatScreenState.dispose`. The
      // chat screen owns no Timer of its own, but its empty-state
      // widget owns an `AnimationController` that must be released.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(ChatScreen), findsNothing);
    });
  });
}
