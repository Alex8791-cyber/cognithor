import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:cognithor_ui/theme/cognithor_theme.dart';

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
    this.onRetry,
    this.showRetry = false,
  });

  /// The plain-text content of the message (used for clipboard / edit).
  final String text;

  /// Whether this message belongs to the user (determines which icons show).
  final bool isUser;

  /// Called when the user taps the edit icon (user messages only).
  final VoidCallback? onEdit;

  /// Called when the user taps the retry icon (assistant messages only).
  final VoidCallback? onRetry;

  /// Whether to show the retry icon (only on last assistant message).
  final bool showRetry;

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
    final alignment = widget.isUser
        ? MainAxisAlignment.end
        : MainAxisAlignment.start;

    return Padding(
      padding: const EdgeInsets.only(top: 2, bottom: 4),
      child: Row(
        mainAxisAlignment: alignment,
        children: [
          if (widget.isUser && widget.onEdit != null)
            _ActionIcon(
              icon: Icons.edit_outlined,
              tooltip: 'Edit',
              onTap: widget.onEdit!,
            ),
          if (widget.isUser && widget.onEdit != null) const SizedBox(width: 2),
          _ActionIcon(
            icon: _copied ? Icons.check : Icons.copy_outlined,
            tooltip: _copied ? 'Copied!' : 'Copy',
            onTap: _copied ? null : _copyToClipboard,
            highlight: _copied,
          ),
          if (!widget.isUser && widget.showRetry && widget.onRetry != null) ...[
            const SizedBox(width: 2),
            _ActionIcon(
              icon: Icons.refresh,
              tooltip: 'Retry',
              onTap: widget.onRetry!,
            ),
          ],
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
            color: highlight
                ? CognithorTheme.green
                : CognithorTheme.textTertiary,
          ),
        ),
      ),
    );
  }
}
