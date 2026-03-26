import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:jarvis_ui/theme/jarvis_theme.dart';

/// Inline action icons (copy, edit) shown beneath a chat message bubble.
///
/// * User messages: edit + copy
/// * Assistant messages: copy only (feedback buttons are rendered separately)
class MessageActionButtons extends StatefulWidget {
  const MessageActionButtons({
    super.key,
    required this.text,
    required this.isUser,
    this.onEdit,
  });

  /// The plain-text content of the message (used for clipboard / edit).
  final String text;

  /// Whether this message belongs to the user (determines which icons show).
  final bool isUser;

  /// Called when the user taps the edit icon. Typically populates the input
  /// field with [text].
  final VoidCallback? onEdit;

  @override
  State<MessageActionButtons> createState() => _MessageActionButtonsState();
}

class _MessageActionButtonsState extends State<MessageActionButtons> {
  bool _copied = false;

  Future<void> _copyToClipboard() async {
    await Clipboard.setData(ClipboardData(text: widget.text));
    if (!mounted) return;
    setState(() => _copied = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _copied = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    final alignment =
        widget.isUser ? MainAxisAlignment.end : MainAxisAlignment.start;

    return Padding(
      padding: const EdgeInsets.only(top: 2, bottom: 4),
      child: Row(
        mainAxisAlignment: alignment,
        children: [
          if (widget.isUser && widget.onEdit != null)
            _ActionIcon(
              icon: Icons.edit_outlined,
              tooltip: 'Bearbeiten',
              onTap: widget.onEdit!,
            ),
          if (widget.isUser && widget.onEdit != null)
            const SizedBox(width: 2),
          _ActionIcon(
            icon: _copied ? Icons.check : Icons.copy_outlined,
            tooltip: _copied ? 'Kopiert!' : 'Kopieren',
            onTap: _copied ? null : _copyToClipboard,
            highlight: _copied,
          ),
        ],
      ),
    );
  }
}

class _ActionIcon extends StatelessWidget {
  const _ActionIcon({
    required this.icon,
    required this.tooltip,
    required this.onTap,
    this.highlight = false,
  });

  final IconData icon;
  final String tooltip;
  final VoidCallback? onTap;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(4),
          child: Icon(
            icon,
            size: 15,
            color: highlight ? JarvisTheme.green : JarvisTheme.textTertiary,
          ),
        ),
      ),
    );
  }
}
