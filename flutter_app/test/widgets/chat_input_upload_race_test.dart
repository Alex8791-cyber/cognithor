/// Regression test for Critical Bug C2: send-during-upload race.
///
/// When the user picks a video, [ChatProvider.sendVideo] starts a
/// multipart POST that can take several seconds. During that window
/// the user could hit Enter/Send and ship a naked text message that
/// orphans the pending video onto the NEXT message.
///
/// The fix wires [_pickVideo] through [_isUploading] with try/finally
/// (matching [_pickFile]) AND guards [_submit] against running while
/// an upload is active. The Send [IconButton] is visually disabled
/// (progress indicator, null onPressed) while the upload runs.
library;

import 'dart:async';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/widgets/chat_input.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:provider/provider.dart';

/// Mock [FilePicker] platform that returns a single fake file. Uses
/// [MockPlatformInterfaceMixin] so [FilePicker.platform = this] works
/// without triggering the token verification exception.
class _MockFilePicker extends FilePicker with MockPlatformInterfaceMixin {
  @override
  Future<FilePickerResult?> pickFiles({
    String? dialogTitle,
    String? initialDirectory,
    FileType type = FileType.any,
    List<String>? allowedExtensions,
    dynamic onFileLoading,
    bool allowCompression = true,
    int compressionQuality = 30,
    bool allowMultiple = false,
    bool withData = false,
    bool withReadStream = false,
    bool lockParentWindow = false,
    bool readSequential = false,
  }) async {
    return FilePickerResult([
      PlatformFile(name: 'dummy.mp4', size: 1024, path: '/tmp/dummy.mp4'),
    ]);
  }
}

/// Stub provider that hangs forever on [sendVideo] so the widget stays
/// in the "uploading" state. We drive the race via Enter on the text
/// field and assert that [sendMessage] is NOT called.
class _StubChatProvider extends ChatProvider {
  final Completer<void> uploadCompleter = Completer<void>();
  int sendMessageCalls = 0;
  int sendVideoCalls = 0;

  @override
  Future<void> sendVideo(String localPath, String filename) async {
    sendVideoCalls++;
    await uploadCompleter.future; // block like a slow multipart POST
  }

  @override
  void sendMessage(String text) {
    sendMessageCalls++;
  }
}

Widget _wrap(Widget child, _StubChatProvider cp) {
  final bp = LlmBackendProvider(apiBaseUrl: 'http://test');
  bp.active = 'vllm';
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
          ChangeNotifierProvider<ChatProvider>.value(value: cp),
          ChangeNotifierProvider<LlmBackendProvider>.value(value: bp),
          ChangeNotifierProvider<VoiceProvider>(create: (_) => VoiceProvider()),
        ],
        child: child,
      ),
    ),
  );
}

void main() {
  setUp(() {
    FilePicker.platform = _MockFilePicker();
  });

  testWidgets(
    'Send button is disabled with spinner while video upload is in progress',
    (tester) async {
      final cp = _StubChatProvider();
      final sent = <String>[];
      await tester.pumpWidget(
        _wrap(ChatInput(onSend: sent.add, onCancel: () {}), cp),
      );

      // Precondition: Send icon visible, no progress indicator in the send slot.
      expect(find.byIcon(Icons.send), findsOneWidget);

      // Trigger _pickVideo via the paperclip menu.
      await tester.tap(find.byKey(const ValueKey('chat-input-paperclip')));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Video hochladen'));
      // Let the mock picker resolve + the widget rebuild, but keep
      // sendVideo blocked on the Completer.
      await tester.pump();
      await tester.pump();

      expect(
        cp.sendVideoCalls,
        1,
        reason: 'sendVideo should have been invoked from _pickVideo',
      );

      // The Send icon must now be replaced by a CircularProgressIndicator.
      expect(
        find.byIcon(Icons.send),
        findsNothing,
        reason: 'Send icon should be hidden while uploading',
      );
      // Two progress indicators total: one in the paperclip slot, one
      // replacing the Send icon.
      expect(find.byType(CircularProgressIndicator), findsNWidgets(2));

      // Locate the IconButton that previously held the Send icon. It is
      // the last IconButton in the row; its onPressed must now be null.
      final sendButton = tester
          .widgetList<IconButton>(find.byType(IconButton))
          .last;
      expect(
        sendButton.onPressed,
        isNull,
        reason: 'Send button onPressed must be null while uploading',
      );

      // Simulate the user pressing Enter in the TextField (the bug path).
      final textField = find.byType(TextField);
      await tester.enterText(textField, 'hello');
      await tester.testTextInput.receiveAction(TextInputAction.send);
      await tester.pump();

      expect(
        sent,
        isEmpty,
        reason: 'onSend must not fire while upload is in progress (race guard)',
      );

      // Release the upload so the widget tears down cleanly.
      cp.uploadCompleter.complete();
      await tester.pumpAndSettle();
    },
  );

  testWidgets('after upload completes, user can send text normally again', (
    tester,
  ) async {
    final cp = _StubChatProvider();
    final sent = <String>[];
    await tester.pumpWidget(
      _wrap(ChatInput(onSend: sent.add, onCancel: () {}), cp),
    );

    // Kick off the upload.
    await tester.tap(find.byKey(const ValueKey('chat-input-paperclip')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Video hochladen'));
    await tester.pump();
    await tester.pump();

    // While uploading, pressing the Send IconButton must be a no-op
    // because its onPressed is null.
    final sendButton = tester
        .widgetList<IconButton>(find.byType(IconButton))
        .last;
    expect(sendButton.onPressed, isNull);

    // Release the upload — _isUploading should flip back to false.
    cp.uploadCompleter.complete();
    await tester.pumpAndSettle();

    // Send icon is back.
    expect(find.byIcon(Icons.send), findsOneWidget);

    // Now typing + Enter should reach onSend exactly once.
    await tester.enterText(find.byType(TextField), 'hello');
    await tester.testTextInput.receiveAction(TextInputAction.send);
    await tester.pump();

    expect(sent, [
      'hello',
    ], reason: 'onSend should fire once the upload finishes and flag resets');
  });
}
