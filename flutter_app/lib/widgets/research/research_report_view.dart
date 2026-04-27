import 'package:flutter/material.dart';
import 'package:cognithor_ui/providers/research_provider.dart';

/// Renders a Markdown research report as selectable text.
///
/// Headings (`#`, `##`, `###`) are bolded and sized down accordingly.
/// Citation references like `[1]`, `[2]` are highlighted with the accent color.
/// Code blocks (``` fences) are rendered with a monospace font on a surface bg.
class ResearchReportView extends StatelessWidget {
  final ResearchResult result;

  const ResearchReportView({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    const accentColor = Color(0xFF00BCD4); // research accent

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Query header
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: accentColor.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: accentColor.withValues(alpha: 0.25)),
          ),
          child: Row(
            children: [
              const Icon(Icons.biotech, color: accentColor, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  result.query,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),

        // Metadata row
        Row(
          children: [
            _MetaBadge(
              icon: Icons.account_tree_outlined,
              label: '${result.hops} hops',
              color: accentColor,
            ),
            const SizedBox(width: 8),
            _MetaBadge(
              icon: Icons.verified_outlined,
              label:
                  '${(result.confidenceAvg * 100).toStringAsFixed(0)}% confidence',
              color: _confidenceColor(result.confidenceAvg),
            ),
            const SizedBox(width: 8),
            _MetaBadge(
              icon: Icons.link,
              label: '${result.sources.length} sources',
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ],
        ),
        const SizedBox(height: 16),

        // Report body
        _MarkdownBody(markdown: result.reportMd, accentColor: accentColor),

        // Sources list
        if (result.sources.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(
            'Sources',
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: accentColor,
            ),
          ),
          const SizedBox(height: 8),
          ...result.sources.asMap().entries.map((entry) {
            final idx = entry.key + 1;
            final src = entry.value;
            final title =
                src['title'] as String? ?? src['url'] as String? ?? '';
            final url = src['url'] as String? ?? '';
            return Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '[$idx]',
                    style: const TextStyle(
                      color: accentColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                      fontFamily: 'monospace',
                    ),
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (title.isNotEmpty)
                          Text(
                            title,
                            style: theme.textTheme.bodySmall?.copyWith(
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        if (url.isNotEmpty)
                          SelectableText(
                            url,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                              fontSize: 11,
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ],
    );
  }

  Color _confidenceColor(double confidence) {
    if (confidence >= 0.8) return const Color(0xFF00E676);
    if (confidence >= 0.5) return const Color(0xFFFFAB40);
    return const Color(0xFFFF5252);
  }
}

class _MetaBadge extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;

  const _MetaBadge({
    required this.icon,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 4),
          Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

/// Simple line-by-line Markdown renderer.
///
/// Handles: headings (`#`/`##`/`###`), bold (`**text**`), code fences,
/// horizontal rules (`---`), and citation references (`[N]`).
class _MarkdownBody extends StatelessWidget {
  final String markdown;
  final Color accentColor;

  const _MarkdownBody({required this.markdown, required this.accentColor});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final lines = markdown.split('\n');
    final widgets = <Widget>[];
    bool inCodeBlock = false;
    final codeLines = <String>[];

    void flushCode() {
      if (codeLines.isNotEmpty) {
        widgets.add(
          Container(
            width: double.infinity,
            margin: const EdgeInsets.symmetric(vertical: 4),
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: theme.colorScheme.outlineVariant),
            ),
            child: SelectableText(
              codeLines.join('\n'),
              style: theme.textTheme.bodySmall?.copyWith(
                fontFamily: 'monospace',
                fontSize: 12,
                color: theme.colorScheme.onSurface,
              ),
            ),
          ),
        );
        codeLines.clear();
      }
    }

    for (final rawLine in lines) {
      // Code fence toggle
      if (rawLine.trimLeft().startsWith('```')) {
        if (inCodeBlock) {
          flushCode();
          inCodeBlock = false;
        } else {
          inCodeBlock = true;
        }
        continue;
      }
      if (inCodeBlock) {
        codeLines.add(rawLine);
        continue;
      }

      // Horizontal rule
      if (rawLine.trim() == '---' || rawLine.trim() == '***') {
        widgets.add(const Divider(height: 20));
        continue;
      }

      // Empty line → spacing
      if (rawLine.trim().isEmpty) {
        widgets.add(const SizedBox(height: 6));
        continue;
      }

      // Headings
      if (rawLine.startsWith('### ')) {
        widgets.add(
          Padding(
            padding: const EdgeInsets.only(top: 12, bottom: 4),
            child: Text(
              rawLine.substring(4),
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        );
        continue;
      }
      if (rawLine.startsWith('## ')) {
        widgets.add(
          Padding(
            padding: const EdgeInsets.only(top: 14, bottom: 4),
            child: Text(
              rawLine.substring(3),
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        );
        continue;
      }
      if (rawLine.startsWith('# ')) {
        widgets.add(
          Padding(
            padding: const EdgeInsets.only(top: 16, bottom: 6),
            child: Text(
              rawLine.substring(2),
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        );
        continue;
      }

      // Regular paragraph line — render with citation highlighting
      widgets.add(
        Padding(
          padding: const EdgeInsets.only(bottom: 2),
          child: _InlineText(
            line: rawLine,
            accentColor: accentColor,
            baseStyle: theme.textTheme.bodyMedium,
          ),
        ),
      );
    }

    // Flush any unclosed code block
    if (inCodeBlock) flushCode();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: widgets,
    );
  }
}

/// Renders a single line with citation references `[N]` highlighted.
class _InlineText extends StatelessWidget {
  final String line;
  final Color accentColor;
  final TextStyle? baseStyle;

  const _InlineText({
    required this.line,
    required this.accentColor,
    this.baseStyle,
  });

  @override
  Widget build(BuildContext context) {
    // Build a RichText with citation spans highlighted
    final spans = <TextSpan>[];
    final citationRegex = RegExp(r'\[(\d+)\]');
    int lastEnd = 0;

    // Strip leading list marker for display
    var text = line;
    if (text.startsWith('- ') || text.startsWith('* ')) {
      text = '\u2022 ${text.substring(2)}';
    }

    // Strip simple bold markers **text**
    text = text.replaceAllMapped(
      RegExp(r'\*\*(.+?)\*\*'),
      (m) => m.group(1) ?? '',
    );

    for (final match in citationRegex.allMatches(text)) {
      if (match.start > lastEnd) {
        spans.add(TextSpan(text: text.substring(lastEnd, match.start)));
      }
      spans.add(
        TextSpan(
          text: match.group(0),
          style: TextStyle(
            color: accentColor,
            fontWeight: FontWeight.bold,
            fontFamily: 'monospace',
            fontSize: 12,
          ),
        ),
      );
      lastEnd = match.end;
    }
    if (lastEnd < text.length) {
      spans.add(TextSpan(text: text.substring(lastEnd)));
    }

    return SelectableText.rich(TextSpan(style: baseStyle, children: spans));
  }
}
