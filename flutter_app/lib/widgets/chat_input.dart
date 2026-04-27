import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:provider/provider.dart';

class ChatInput extends StatefulWidget {
  const ChatInput({
    super.key,
    required this.onSend,
    required this.onCancel,
    this.onFile,
    this.isProcessing = false,
    this.controller,
    this.focusNode,
  });

  final void Function(String text) onSend;
  final VoidCallback onCancel;
  final void Function(String name, String type, String base64)? onFile;
  final bool isProcessing;

  /// Optional external [TextEditingController] for programmatic text insertion
  /// (e.g. edit-message). When null an internal controller is used.
  final TextEditingController? controller;

  /// Optional external [FocusNode]. When null an internal node is used.
  final FocusNode? focusNode;

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> {
  TextEditingController? _ownController;
  FocusNode? _ownFocusNode;
  // Dedicated FocusNode for the KeyboardListener wrapping the TextField.
  // Must be owned by State (not rebuilt inline in build()) or every rebuild
  // leaks a ChangeNotifier holding native resources. (Bug-2 round 4)
  late final FocusNode _keyboardListenerFocusNode;
  bool _isUploading = false;

  @override
  void initState() {
    super.initState();
    _keyboardListenerFocusNode = FocusNode();
  }

  TextEditingController get _controller =>
      widget.controller ?? (_ownController ??= TextEditingController());
  FocusNode get _focusNode =>
      widget.focusNode ?? (_ownFocusNode ??= FocusNode());

  void _submit() {
    // Guard against the send-during-upload race: if a video upload is
    // still in flight, _pendingVideoAttachment is not yet populated,
    // so letting the text message go now would orphan the video onto
    // the NEXT message. See C2 regression test.
    if (_isUploading) return;
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    widget.onSend(text);
    _controller.clear();
    _focusNode.requestFocus();
  }

  Future<void> _pickImage() async {
    if (widget.onFile == null) return;
    try {
      final result = await FilePicker.platform.pickFiles(
        withData: true,
        type: FileType.custom,
        allowedExtensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'],
      );
      if (result == null) return;
      final file = result.files.single;
      if (file.bytes == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(AppLocalizations.of(context).fileReadError)),
          );
        }
        return;
      }
      setState(() => _isUploading = true);
      final b64 = base64Encode(file.bytes!);
      widget.onFile!(file.name, file.extension ?? 'bin', b64);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              AppLocalizations.of(context).uploadError(e.toString()),
            ),
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  Future<void> _pickVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp4', 'webm', 'mov', 'mkv', 'avi'],
      withData: false,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.first;
    if (file.path == null) return;
    if (!mounted) return;
    // Track the upload in _isUploading so the Send button and the
    // _submit() guard can both block user input while the multipart
    // POST is in flight — otherwise a quick Enter ships a naked text
    // message and orphans the pending video onto the NEXT message.
    setState(() => _isUploading = true);
    Object? uploadError;
    try {
      await context.read<ChatProvider>().sendVideo(file.path!, file.name);
    } catch (e) {
      uploadError = e;
    }

    if (!mounted) return;

    // See Bug I1-r2: defer the _isUploading=false reset to the next frame
    // so ChatProvider.notifyListeners() (fired inside sendVideo after it
    // set _pendingVideoAttachment) has already rebuilt the tree before
    // the send button's guard is lifted. Without this deferral there is
    // a one-frame window where _isUploading is false but the build has
    // not yet consumed the provider update, leaving the UI briefly
    // inconsistent. The error snackbar is shown in the same post-frame
    // callback so its timing matches the UI state transition.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      setState(() => _isUploading = false);
      if (uploadError != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Video upload fehlgeschlagen: $uploadError')),
        );
      }
    });
  }

  Future<void> _pickFile() async {
    if (widget.onFile == null) return;
    try {
      final result = await FilePicker.platform.pickFiles(
        withData: true,
        type: FileType.custom,
        allowedExtensions: [
          'pdf',
          'txt',
          'md',
          'csv',
          'json',
          'xml',
          'yaml',
          'yml',
          'doc',
          'docx',
          'xls',
          'xlsx',
          'pptx',
          'py',
          'js',
          'ts',
          'dart',
          'html',
          'css',
          'zip',
          'tar',
          'gz',
        ],
      );
      if (result == null) return;
      final file = result.files.single;
      if (file.bytes == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(AppLocalizations.of(context).fileReadError)),
          );
        }
        return;
      }
      setState(() => _isUploading = true);
      final b64 = base64Encode(file.bytes!);
      widget.onFile!(file.name, file.extension ?? 'bin', b64);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              AppLocalizations.of(context).uploadError(e.toString()),
            ),
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  Future<void> _showUrlDialog() async {
    // Use a stateful dialog widget so the TextEditingController is owned
    // by the dialog's State and disposed in its own dispose() lifecycle.
    // Previously the controller was allocated here and leaked — every
    // dialog open created an undisposed controller. (Bug I2-r3)
    final url = await showDialog<String>(
      context: context,
      builder: (ctx) => const _UrlInputDialog(),
    );

    if (url == null || url.isEmpty) return;
    if (!mounted) return;

    final provider = context.read<ChatProvider>();
    final accepted = provider.handlePastedTextForVideoUrl(url);
    if (!accepted && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Ungültige URL — direkt auf .mp4/.webm/.mov/.mkv/.avi endend erforderlich.',
          ),
        ),
      );
    }
  }

  @override
  void dispose() {
    _keyboardListenerFocusNode.dispose();
    _ownController?.dispose();
    _ownFocusNode?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Container(
      padding: EdgeInsets.fromLTRB(
        MediaQuery.of(context).size.width > 400 ? 16 : 8,
        8,
        MediaQuery.of(context).size.width > 400 ? 16 : 8,
        16,
      ),
      decoration: BoxDecoration(
        color: Theme.of(context).scaffoldBackgroundColor,
        border: Border(top: BorderSide(color: Theme.of(context).dividerColor)),
      ),
      child: Row(
        children: [
          // Attach file / media button
          if (_isUploading)
            const Padding(
              padding: EdgeInsets.all(12),
              child: SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            PopupMenuButton<String>(
              key: const ValueKey('chat-input-paperclip'),
              icon: Icon(
                Icons.attach_file,
                color: CognithorTheme.textSecondary,
              ),
              iconSize: 22,
              tooltip: l.attachFile,
              enabled: !widget.isProcessing,
              onSelected: (value) async {
                switch (value) {
                  case 'image':
                    await _pickImage();
                    break;
                  case 'video':
                    await _pickVideo();
                    break;
                  case 'file':
                    await _pickFile();
                    break;
                  case 'url':
                    _showUrlDialog();
                    break;
                }
              },
              itemBuilder: (context) {
                final activeBackend = context.read<LlmBackendProvider>().active;
                final vllmActive = activeBackend == 'vllm';
                return [
                  const PopupMenuItem<String>(
                    value: 'image',
                    child: Text('Bild hochladen'),
                  ),
                  PopupMenuItem<String>(
                    value: 'video',
                    enabled: vllmActive,
                    child: Tooltip(
                      message: vllmActive
                          ? 'Video hochladen (nur mit vLLM-Backend)'
                          : 'Video-Analyse erfordert vLLM — unter Settings → LLM Backends wechseln.',
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Text('Video hochladen'),
                          if (!vllmActive)
                            const Padding(
                              padding: EdgeInsets.only(left: 6),
                              child: Icon(
                                Icons.lock,
                                size: 14,
                                color: Colors.grey,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                  const PopupMenuItem<String>(
                    value: 'file',
                    child: Text('Datei hochladen'),
                  ),
                  const PopupMenuItem<String>(
                    value: 'url',
                    child: Text('URL einfügen'),
                  ),
                ];
              },
            ),

          // Text field
          Expanded(
            child: KeyboardListener(
              focusNode: _keyboardListenerFocusNode,
              onKeyEvent: (event) {
                if (event is KeyDownEvent &&
                    event.logicalKey == LogicalKeyboardKey.enter &&
                    !HardwareKeyboard.instance.isShiftPressed) {
                  _submit();
                }
              },
              child: TextField(
                controller: _controller,
                focusNode: _focusNode,
                autofocus: true,
                maxLines: 4,
                minLines: 1,
                textInputAction: TextInputAction.send,
                decoration: InputDecoration(
                  hintText: l.typeMessage,
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                ),
                onSubmitted: (_) => _submit(),
              ),
            ),
          ),

          const SizedBox(width: 4),

          // Voice button
          Consumer<VoiceProvider>(
            builder: (context, voice, _) {
              return IconButton(
                onPressed: () => voice.toggle(),
                icon: Icon(
                  voice.isActive ? Icons.mic : Icons.mic_none,
                  color: voice.isActive
                      ? CognithorTheme.sectionChat
                      : CognithorTheme.textSecondary,
                ),
                tooltip: l.voiceMode,
                iconSize: 22,
              );
            },
          ),

          // Send / Cancel
          if (widget.isProcessing)
            IconButton(
              onPressed: widget.onCancel,
              icon: Icon(Icons.stop_circle, color: CognithorTheme.red),
              tooltip: l.cancel,
            )
          else
            IconButton(
              onPressed: _isUploading ? null : _submit,
              icon: _isUploading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Icon(Icons.send, color: CognithorTheme.accent),
              tooltip: _isUploading ? 'Upload läuft…' : l.send,
            ),
        ],
      ),
    );
  }
}

/// Stateful dialog body for `_showUrlDialog`.
///
/// Owns its `TextEditingController` so disposal happens via the normal
/// `State.dispose()` lifecycle. Fixes Bug I2-r3: controller leak when
/// the parent widget allocated the controller itself and failed to
/// dispose it.
class _UrlInputDialog extends StatefulWidget {
  const _UrlInputDialog();

  @override
  State<_UrlInputDialog> createState() => _UrlInputDialogState();
}

class _UrlInputDialogState extends State<_UrlInputDialog> {
  final TextEditingController controller = TextEditingController();

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('URL einfügen'),
      content: TextField(
        controller: controller,
        autofocus: true,
        keyboardType: TextInputType.url,
        decoration: const InputDecoration(
          hintText: 'https://example.com/clip.mp4',
          helperText: 'Direkter Link zu .mp4 / .webm / .mov / .mkv / .avi',
        ),
        onSubmitted: (v) => Navigator.of(context).pop(v.trim()),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(null),
          child: const Text('Abbrechen'),
        ),
        FilledButton(
          onPressed: () => Navigator.of(context).pop(controller.text.trim()),
          child: const Text('Hinzufügen'),
        ),
      ],
    );
  }
}
