import 'dart:io';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/widgets/chat_input.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

LlmBackendProvider _mkBackendProvider(String active) {
  final p = LlmBackendProvider(apiBaseUrl: 'http://test');
  p.active = active;
  return p;
}

ChatProvider _mkChatProvider() {
  return ChatProvider();
}

Widget _wrap(Widget child, LlmBackendProvider bp, ChatProvider cp) {
  return MaterialApp(
    localizationsDelegates: const [
      AppLocalizations.delegate,
      GlobalMaterialLocalizations.delegate,
      GlobalWidgetsLocalizations.delegate,
    ],
    supportedLocales: AppLocalizations.supportedLocales,
    home: Scaffold(
      body: MultiProvider(
        providers: [
          ChangeNotifierProvider<LlmBackendProvider>.value(value: bp),
          ChangeNotifierProvider<ChatProvider>.value(value: cp),
          ChangeNotifierProvider<VoiceProvider>(create: (_) => VoiceProvider()),
        ],
        child: child,
      ),
    ),
  );
}

void main() {
  testWidgets('paperclip opens popup menu with 4 entries', (tester) async {
    final bp = _mkBackendProvider('vllm');
    final cp = _mkChatProvider();
    await tester.pumpWidget(
      _wrap(ChatInput(onSend: (_) {}, onCancel: () {}), bp, cp),
    );

    await tester.tap(find.byKey(const ValueKey('chat-input-paperclip')));
    await tester.pumpAndSettle();

    expect(find.text('Bild hochladen'), findsOneWidget);
    expect(find.text('Video hochladen'), findsOneWidget);
    expect(find.text('Datei hochladen'), findsOneWidget);
    expect(find.text('URL einfügen'), findsOneWidget);
  });

  testWidgets('Video entry disabled when active backend != vllm', (
    tester,
  ) async {
    final bp = _mkBackendProvider('ollama');
    final cp = _mkChatProvider();
    await tester.pumpWidget(
      _wrap(ChatInput(onSend: (_) {}, onCancel: () {}), bp, cp),
    );

    await tester.tap(find.byKey(const ValueKey('chat-input-paperclip')));
    await tester.pumpAndSettle();

    final videoItem = tester.widget<PopupMenuItem<String>>(
      find.ancestor(
        of: find.text('Video hochladen'),
        matching: find.byType(PopupMenuItem<String>),
      ),
    );
    expect(videoItem.enabled, isFalse);
  });

  testWidgets(
    'URL dialog accepts valid video URL and sets pending attachment',
    (tester) async {
      final bp = _mkBackendProvider('vllm');
      final cp = _mkChatProvider();
      await tester.pumpWidget(
        _wrap(ChatInput(onSend: (_) {}, onCancel: () {}), bp, cp),
      );

      await tester.tap(find.byKey(const ValueKey('chat-input-paperclip')));
      await tester.pumpAndSettle();
      await tester.tap(find.text('URL einfügen'));
      await tester.pumpAndSettle();

      // Dialog should be open (title + menu entry both contain "URL einfügen")
      expect(find.text('URL einfügen'), findsAtLeastNWidgets(1));
      // Scope the TextField lookup to the dialog — the ChatInput's own
      // TextField is also in the tree.
      final dialogField = find.descendant(
        of: find.byType(AlertDialog),
        matching: find.byType(TextField),
      );
      expect(dialogField, findsOneWidget);

      await tester.enterText(dialogField, 'https://x.com/clip.mp4');
      await tester.tap(find.text('Hinzufügen'));
      await tester.pumpAndSettle();

      expect(cp.pendingVideoAttachment, isNotNull);
      expect(cp.pendingVideoAttachment!['url'], 'https://x.com/clip.mp4');
    },
  );

  testWidgets('URL dialog rejects non-video URL with snackbar', (tester) async {
    final bp = _mkBackendProvider('vllm');
    final cp = _mkChatProvider();
    await tester.pumpWidget(
      _wrap(ChatInput(onSend: (_) {}, onCancel: () {}), bp, cp),
    );

    await tester.tap(find.byKey(const ValueKey('chat-input-paperclip')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('URL einfügen'));
    await tester.pumpAndSettle();

    final dialogField = find.descendant(
      of: find.byType(AlertDialog),
      matching: find.byType(TextField),
    );
    await tester.enterText(dialogField, 'https://example.com/page.html');
    await tester.tap(find.text('Hinzufügen'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Ungültige URL'), findsOneWidget);
    expect(cp.pendingVideoAttachment, isNull);
  });

  test('URL dialog disposes TextEditingController to prevent leak', () {
    // Source-level regression: this is the simplest way to prove the fix
    // stays in place. A missing dispose() would leak one controller per
    // dialog open. Bug I2-r3 was fixed by moving the controller into a
    // StatefulWidget (`_UrlInputDialog`) whose State.dispose() releases it.
    final source = File('lib/widgets/chat_input.dart').readAsStringSync();

    // Source must still allocate a TextEditingController for the URL input.
    expect(
      source.contains('TextEditingController()'),
      isTrue,
      reason: 'URL dialog must allocate a TextEditingController',
    );

    // Scope the dispose check to the stateful URL-input dialog widget.
    final urlDialogStart = source.indexOf('_UrlInputDialogState');
    expect(
      urlDialogStart,
      greaterThan(-1),
      reason: '_UrlInputDialogState must own the controller lifecycle',
    );

    final tail = source.substring(urlDialogStart);
    expect(
      tail.contains('controller.dispose()'),
      isTrue,
      reason: 'Dialog must dispose its TextEditingController (Bug I2-r3)',
    );
  });

  test('chat_input does not allocate FocusNode inline in build()', () {
    final source = File('lib/widgets/chat_input.dart').readAsStringSync();

    // Find the build() method body. A robust way: find the first occurrence
    // of "Widget build(BuildContext context)" and scan forward.
    final buildStart = source.indexOf('Widget build(BuildContext context)');
    expect(
      buildStart,
      greaterThan(-1),
      reason: 'ChatInput must have a build() method',
    );

    // Take everything from build() to the end of the method — rough, but
    // enough: the next top-level `Widget ` or end-of-class is the boundary.
    final rest = source.substring(buildStart);
    final buildEnd = rest.indexOf(
      '\n  }',
    ); // state-class indent + closing brace
    final buildBody = buildEnd == -1 ? rest : rest.substring(0, buildEnd);

    expect(
      buildBody.contains('FocusNode()'),
      isFalse,
      reason:
          'FocusNode() must not be allocated inline in build() — it leaks '
          'on every rebuild. Promote to a state field disposed in dispose() '
          '(Bug-2 round 4).',
    );
  });
}
