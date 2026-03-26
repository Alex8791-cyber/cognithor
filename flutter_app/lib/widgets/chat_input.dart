import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/voice_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
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
  bool _isUploading = false;

  TextEditingController get _controller =>
      widget.controller ?? (_ownController ??= TextEditingController());
  FocusNode get _focusNode =>
      widget.focusNode ?? (_ownFocusNode ??= FocusNode());

  void _submit() {
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    widget.onSend(text);
    _controller.clear();
    _focusNode.requestFocus();
  }

  Future<void> _pickFile() async {
    if (widget.onFile == null) return;
    try {
      final result = await FilePicker.platform.pickFiles(
        withData: true,
        type: FileType.custom,
        allowedExtensions: [
          'pdf', 'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml',
          'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp',
          'doc', 'docx', 'xls', 'xlsx', 'pptx',
          'py', 'js', 'ts', 'dart', 'html', 'css',
          'zip', 'tar', 'gz',
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
          SnackBar(content: Text(AppLocalizations.of(context).uploadError(e.toString()))),
        );
      }
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  @override
  void dispose() {
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
        border: Border(
          top: BorderSide(color: Theme.of(context).dividerColor),
        ),
      ),
      child: Row(
        children: [
          // Attach file button
          if (widget.onFile != null)
            _isUploading
                ? const Padding(
                    padding: EdgeInsets.all(12),
                    child: SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  )
                : IconButton(
                    onPressed: widget.isProcessing ? null : _pickFile,
                    icon: Icon(
                      Icons.attach_file,
                      color: JarvisTheme.textSecondary,
                    ),
                    tooltip: l.attachFile,
                    iconSize: 22,
                  ),

          // Text field
          Expanded(
            child: KeyboardListener(
              focusNode: FocusNode(),
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
                      ? JarvisTheme.sectionChat
                      : JarvisTheme.textSecondary,
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
              icon: Icon(Icons.stop_circle, color: JarvisTheme.red),
              tooltip: l.cancel,
            )
          else
            IconButton(
              onPressed: _submit,
              icon: Icon(Icons.send, color: JarvisTheme.accent),
              tooltip: l.send,
            ),
        ],
      ),
    );
  }
}
