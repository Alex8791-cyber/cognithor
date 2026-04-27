import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/glass_panel.dart';

class ChatBubble extends StatelessWidget {
  const ChatBubble({
    super.key,
    required this.role,
    required this.text,
    this.isStreaming = false,
    this.metadata = const {},
    this.agentName,
  });

  final MessageRole role;
  final String text;
  final bool isStreaming;
  final Map<String, dynamic> metadata;
  final String? agentName;

  @override
  Widget build(BuildContext context) {
    final isUser = role == MessageRole.user;
    final isSystem = role == MessageRole.system;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth:
              MediaQuery.of(context).size.width *
              (MediaQuery.of(context).size.width > 400 ? 0.85 : 0.78),
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
    const baseColor = CognithorTheme.sectionChat;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final isVideo = metadata['kind'] == 'video';

    final textBubble = Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: isDark
            ? baseColor.withValues(alpha: 0.20)
            : baseColor.withValues(alpha: 0.12),
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(16),
          topRight: Radius.circular(16),
          bottomLeft: Radius.circular(16),
          bottomRight: Radius.circular(4),
        ),
        border: Border.all(
          color: isDark
              ? baseColor.withValues(alpha: 0.40)
              : baseColor.withValues(alpha: 0.50),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_imagePreviewBytes() != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.memory(
                  _imagePreviewBytes()!,
                  width: 240,
                  fit: BoxFit.cover,
                  gaplessPlayback: true,
                ),
              ),
            ),
          Row(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Flexible(
                child: SelectableText(
                  text,
                  style: TextStyle(
                    color: isDark ? Colors.white : const Color(0xFF1A1A2E),
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
                    color: CognithorTheme.sectionChat,
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );

    if (isVideo) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          _buildVideoPreview(context, metadata),
          const SizedBox(height: 6),
          textBubble,
        ],
      );
    }

    return textBubble;
  }

  // ── Video Preview ────────────────────────────────────────────────────
  Widget _buildVideoPreview(BuildContext context, Map<String, dynamic> meta) {
    final filename = meta['filename'] as String? ?? 'video';
    final durationSec = (meta['duration_sec'] as num?)?.toDouble() ?? 0.0;
    final sampling = meta['sampling'] as Map<String, dynamic>? ?? const {};
    final thumbUrl = meta['thumb_url'] as String?;

    final samplingLabel = sampling.containsKey('fps')
        ? 'fps=${sampling['fps']}'
        : 'num_frames=${sampling['num_frames'] ?? '?'}';

    final mins = (durationSec / 60).floor();
    final secs = (durationSec % 60).floor();
    final durationLabel = '$mins:${secs.toString().padLeft(2, '0')}';

    String? fullThumbUrl;
    if (thumbUrl != null) {
      fullThumbUrl = thumbUrl;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Theme.of(
              context,
            ).colorScheme.primary.withValues(alpha: 0.15),
            border: Border.all(
              color: Theme.of(
                context,
              ).colorScheme.primary.withValues(alpha: 0.4),
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 96,
                height: 54,
                child: Container(
                  color: Colors.grey.shade800,
                  child: fullThumbUrl != null
                      ? Image.network(
                          fullThumbUrl,
                          fit: BoxFit.cover,
                          errorBuilder: (context, err, stack) => const Center(
                            child: Text(
                              '\u{1F3AC}',
                              style: TextStyle(fontSize: 22),
                            ),
                          ),
                        )
                      : const Center(
                          child: Text(
                            '\u{1F3AC}',
                            style: TextStyle(fontSize: 22),
                          ),
                        ),
                ),
              ),
              const SizedBox(width: 10),
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      filename,
                      style: const TextStyle(fontWeight: FontWeight.w500),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '$durationLabel \u00B7 $samplingLabel',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.grey.shade400,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        if (durationSec > 15 * 60)
          Container(
            key: const ValueKey('video-long-banner'),
            margin: const EdgeInsets.only(top: 6),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.orange.withValues(alpha: 0.15),
              border: const Border(
                left: BorderSide(color: Colors.orange, width: 3),
              ),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              'Video ${(durationSec / 60).round()} min \u2014 nur 32 Frames werden gesampled. '
              'Zerlege in 5-Min-Clips f\u00FCr mehr Detail.',
              style: const TextStyle(fontSize: 11, color: Colors.orange),
            ),
          ),
      ],
    );
  }

  /// Decode the image_base64 metadata if present. Returns null for non-image
  /// messages or if the base64 is malformed — never throws.
  Uint8List? _imagePreviewBytes() {
    final b64 = metadata['image_base64'];
    if (b64 is! String || b64.isEmpty) return null;
    try {
      return base64Decode(b64);
    } catch (_) {
      return null;
    }
  }

  /// Whether the agent name is a non-default agent (i.e. a delegated agent).
  bool get _isDelegatedAgent {
    if (agentName == null || agentName!.isEmpty) return false;
    final lower = agentName!.toLowerCase();
    return lower != 'jarvis' && lower != 'cognithor';
  }

  // ── Assistant Bubble ─────────────────────────────────────────────────
  Widget _buildAssistantBubble(BuildContext context) {
    const tint = CognithorTheme.sectionChat;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (_isDelegatedAgent)
          Padding(
            padding: const EdgeInsets.only(bottom: 4, left: 4),
            child: Chip(
              label: Text(
                agentName!,
                style: const TextStyle(fontSize: 11, color: Colors.white),
              ),
              backgroundColor: CognithorTheme.purple.withValues(alpha: 0.7),
              visualDensity: VisualDensity.compact,
              padding: EdgeInsets.zero,
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
              side: BorderSide.none,
            ),
          ),
        Container(
          decoration: BoxDecoration(
            color: isDark
                ? tint.withValues(alpha: 0.10)
                : tint.withValues(alpha: 0.08),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(16),
              topRight: Radius.circular(16),
              bottomLeft: Radius.circular(4),
              bottomRight: Radius.circular(16),
            ),
            border: Border.all(
              color: isDark
                  ? tint.withValues(alpha: 0.25)
                  : tint.withValues(alpha: 0.35),
            ),
          ),
          child: IntrinsicHeight(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Left accent bar
                Container(
                  width: 3,
                  decoration: const BoxDecoration(
                    color: tint,
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
                      horizontal: 16,
                      vertical: 12,
                    ),
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
                              color: tint,
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
        ),
        // Token/model info row (only if metadata present)
        if (metadata.isNotEmpty && !isStreaming)
          Padding(
            padding: const EdgeInsets.only(top: 4, left: 6),
            child: DefaultTextStyle(
              style:
                  theme.textTheme.labelSmall?.copyWith(
                    color: CognithorTheme.textSecondary.withValues(alpha: 0.6),
                    fontSize: 10,
                  ) ??
                  const TextStyle(fontSize: 10),
              child: Wrap(
                spacing: 10,
                children: [
                  if (metadata['model'] != null)
                    Text(metadata['model'].toString()),
                  if (metadata['backend'] != null)
                    Text(metadata['backend'].toString()),
                  if ((metadata['input_tokens'] as int? ?? 0) > 0 ||
                      (metadata['output_tokens'] as int? ?? 0) > 0)
                    Text(
                      '${metadata['input_tokens'] ?? 0} in / '
                      '${metadata['output_tokens'] ?? 0} out tokens',
                    ),
                  if ((metadata['duration_ms'] as int? ?? 0) > 0)
                    Text(
                      '${((metadata['duration_ms'] as int) / 1000).toStringAsFixed(1)}s',
                    ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  // ── System Bubble ────────────────────────────────────────────────────
  Widget _buildSystemBubble(BuildContext context) {
    return GlassPanel(
      tint: CognithorTheme.red,
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
                color: CognithorTheme.red,
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
        code: TextStyle(
          fontFamily: 'monospace',
          fontSize: 13,
          color: CognithorTheme.sectionChat,
          backgroundColor: CognithorTheme.codeBlockBg,
        ),
        codeblockDecoration: BoxDecoration(
          color: CognithorTheme.codeBlockBg,
          borderRadius: BorderRadius.circular(CognithorTheme.spacingSm),
          border: Border.all(
            color: CognithorTheme.sectionChat.withValues(alpha: 0.15),
          ),
          boxShadow: [
            BoxShadow(
              color: CognithorTheme.sectionChat.withValues(alpha: 0.08),
              blurRadius: 8,
              spreadRadius: -1,
            ),
          ],
        ),
        codeblockPadding: const EdgeInsets.all(14),
        h1: TextStyle(
          color: CognithorTheme.textPrimary,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        h2: TextStyle(
          color: CognithorTheme.textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
        h3: TextStyle(
          color: CognithorTheme.textPrimary,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
        blockquoteDecoration: const BoxDecoration(
          border: Border(
            left: BorderSide(color: CognithorTheme.sectionChat, width: 3),
          ),
        ),
        blockquotePadding: const EdgeInsets.only(left: 12),
        listBullet: const TextStyle(color: CognithorTheme.sectionChat),
        a: const TextStyle(color: CognithorTheme.sectionChat),
        strong: TextStyle(
          color: Theme.of(context).colorScheme.onSurface,
          fontWeight: FontWeight.bold,
        ),
        em: TextStyle(
          color: Theme.of(context).colorScheme.onSurface,
          fontStyle: FontStyle.italic,
        ),
        tableHead: TextStyle(
          color: CognithorTheme.textPrimary,
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
        color: CognithorTheme.codeBlockBg,
        borderRadius: BorderRadius.circular(CognithorTheme.spacingSm),
        border: Border.all(
          color: CognithorTheme.sectionChat.withValues(alpha: 0.15),
        ),
        boxShadow: [
          BoxShadow(
            color: CognithorTheme.sectionChat.withValues(alpha: 0.08),
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
                top: Radius.circular(CognithorTheme.spacingSm),
              ),
            ),
            child: Row(
              children: [
                if (widget.language != null)
                  Text(
                    widget.language!,
                    style: TextStyle(
                      fontSize: 11,
                      color: CognithorTheme.textTertiary,
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
                              ? CognithorTheme.green
                              : CognithorTheme.textSecondary,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _copied ? 'Copied' : 'Copy',
                          style: TextStyle(
                            fontSize: 11,
                            color: _copied
                                ? CognithorTheme.green
                                : CognithorTheme.textSecondary,
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
                color: CognithorTheme.textPrimary,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
