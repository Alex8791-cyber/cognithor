import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:jarvis_ui/providers/chat_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';

class ChatBubble extends StatelessWidget {
  const ChatBubble({
    super.key,
    required this.role,
    required this.text,
    this.isStreaming = false,
  });

  final MessageRole role;
  final String text;
  final bool isStreaming;

  @override
  Widget build(BuildContext context) {
    final isUser = role == MessageRole.user;
    final isSystem = role == MessageRole.system;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        margin: const EdgeInsets.symmetric(vertical: 4),
        child: isUser
            ? _buildUserBubble(context)
            : isSystem
                ? _buildSystemBubble(context)
                : _buildAssistantBubble(context),
      ),
    );
  }

  // ── User Bubble ──────────────────────────────────────────────────────
  Widget _buildUserBubble(BuildContext context) {
    const baseColor = JarvisTheme.sectionChat;
    // 20% darker shade
    final darkerColor = Color.lerp(baseColor, Colors.black, 0.2)!;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [baseColor.withValues(alpha: 0.18), darkerColor.withValues(alpha: 0.12)],
        ),
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(16),
          topRight: Radius.circular(16),
          bottomLeft: Radius.circular(16),
          bottomRight: Radius.circular(4), // chat tail
        ),
        border: Border.all(
          color: baseColor.withValues(alpha: 0.20),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Flexible(
            child: SelectableText(
              text,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                height: 1.5,
              ),
            ),
          ),
          if (isStreaming) ...[
            const SizedBox(width: 6),
            const SizedBox(
              width: 8,
              height: 8,
              child: CircularProgressIndicator(
                strokeWidth: 1.5,
                color: JarvisTheme.sectionChat,
              ),
            ),
          ],
        ],
      ),
    );
  }

  // ── Assistant Bubble ─────────────────────────────────────────────────
  Widget _buildAssistantBubble(BuildContext context) {
    return GlassPanel(
      tint: JarvisTheme.sectionChat,
      borderRadius: 16,
      padding: EdgeInsets.zero,
      child: ClipRRect(
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(16),
          topRight: Radius.circular(16),
          bottomLeft: Radius.circular(4), // chat tail
          bottomRight: Radius.circular(16),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Left accent bar
            Container(
              width: 3,
              decoration: const BoxDecoration(
                color: JarvisTheme.sectionChat,
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(16),
                  bottomLeft: Radius.circular(4),
                ),
              ),
            ),
            // Content
            Flexible(
              child: Padding(
                padding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 12),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Flexible(child: _buildMarkdownContent(context)),
                    if (isStreaming) ...[
                      const SizedBox(width: 6),
                      const SizedBox(
                        width: 8,
                        height: 8,
                        child: CircularProgressIndicator(
                          strokeWidth: 1.5,
                          color: JarvisTheme.sectionChat,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── System Bubble ────────────────────────────────────────────────────
  Widget _buildSystemBubble(BuildContext context) {
    return GlassPanel(
      tint: JarvisTheme.red,
      borderRadius: 16,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Flexible(
            child: SelectableText(
              text,
              style: TextStyle(
                color: JarvisTheme.red,
                fontSize: 14,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Markdown Content ─────────────────────────────────────────────────
  Widget _buildMarkdownContent(BuildContext context) {
    return MarkdownBody(
      data: text,
      selectable: true,
      shrinkWrap: true,
      onTapLink: (text, href, title) {
        if (href != null) {
          launchUrl(Uri.parse(href), mode: LaunchMode.externalApplication);
        }
      },
      styleSheet: MarkdownStyleSheet(
        p: TextStyle(
          color: Theme.of(context).colorScheme.onSurface,
          fontSize: 14,
          height: 1.5,
        ),
        code: const TextStyle(
          fontFamily: 'monospace',
          fontSize: 13,
          color: JarvisTheme.sectionChat,
          backgroundColor: JarvisTheme.codeBlockBg,
        ),
        codeblockDecoration: BoxDecoration(
          color: JarvisTheme.codeBlockBg,
          borderRadius: BorderRadius.circular(JarvisTheme.spacingSm),
          border: Border.all(
            color: JarvisTheme.sectionChat.withValues(alpha: 0.15),
          ),
          boxShadow: [
            BoxShadow(
              color: JarvisTheme.sectionChat.withValues(alpha: 0.08),
              blurRadius: 8,
              spreadRadius: -1,
            ),
          ],
        ),
        codeblockPadding: const EdgeInsets.all(14),
        h1: TextStyle(
          color: JarvisTheme.textPrimary,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        h2: TextStyle(
          color: JarvisTheme.textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
        h3: TextStyle(
          color: JarvisTheme.textPrimary,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
        blockquoteDecoration: const BoxDecoration(
          border: Border(
            left: BorderSide(color: JarvisTheme.sectionChat, width: 3),
          ),
        ),
        blockquotePadding: const EdgeInsets.only(left: 12),
        listBullet: const TextStyle(color: JarvisTheme.sectionChat),
        a: const TextStyle(color: JarvisTheme.sectionChat),
        strong: TextStyle(
          color: Theme.of(context).colorScheme.onSurface,
          fontWeight: FontWeight.bold,
        ),
        em: TextStyle(
          color: Theme.of(context).colorScheme.onSurface,
          fontStyle: FontStyle.italic,
        ),
        tableHead: TextStyle(
          color: JarvisTheme.textPrimary,
          fontWeight: FontWeight.bold,
        ),
        tableBorder: TableBorder.all(color: Theme.of(context).dividerColor),
        tableHeadAlign: TextAlign.left,
        tableCellsPadding: const EdgeInsets.symmetric(
          horizontal: 8,
          vertical: 4,
        ),
      ),
    );
  }
}

/// Standalone code block widget with a copy button.
/// Can be used outside of Markdown for displaying code snippets.
class CodeBlockWithCopy extends StatefulWidget {
  const CodeBlockWithCopy({super.key, required this.code, this.language});

  final String code;
  final String? language;

  @override
  State<CodeBlockWithCopy> createState() => _CodeBlockWithCopyState();
}

class _CodeBlockWithCopyState extends State<CodeBlockWithCopy> {
  bool _copied = false;

  void _copyToClipboard() {
    Clipboard.setData(ClipboardData(text: widget.code));
    setState(() => _copied = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _copied = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      decoration: BoxDecoration(
        color: JarvisTheme.codeBlockBg,
        borderRadius: BorderRadius.circular(JarvisTheme.spacingSm),
        border: Border.all(
          color: JarvisTheme.sectionChat.withValues(alpha: 0.15),
        ),
        boxShadow: [
          BoxShadow(
            color: JarvisTheme.sectionChat.withValues(alpha: 0.08),
            blurRadius: 8,
            spreadRadius: -1,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header with optional language label and copy button
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Theme.of(context).dividerColor.withAlpha(80),
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(JarvisTheme.spacingSm),
              ),
            ),
            child: Row(
              children: [
                if (widget.language != null)
                  Text(
                    widget.language!,
                    style: TextStyle(
                      fontSize: 11,
                      color: JarvisTheme.textTertiary,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                const Spacer(),
                InkWell(
                  onTap: _copyToClipboard,
                  borderRadius: BorderRadius.circular(4),
                  child: Padding(
                    padding: const EdgeInsets.all(4),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _copied ? Icons.check : Icons.copy,
                          size: 14,
                          color: _copied
                              ? JarvisTheme.green
                              : JarvisTheme.textSecondary,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _copied ? 'Copied' : 'Copy',
                          style: TextStyle(
                            fontSize: 11,
                            color: _copied
                                ? JarvisTheme.green
                                : JarvisTheme.textSecondary,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          // Code content
          Padding(
            padding: const EdgeInsets.all(14),
            child: SelectableText(
              widget.code,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 13,
                color: JarvisTheme.textPrimary,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
