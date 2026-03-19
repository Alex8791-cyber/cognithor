import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';

/// A side panel that shows context about the currently active tool.
///
/// Appears on the right side of the chat when the assistant is using a tool.
class ContextPanel extends StatelessWidget {
  const ContextPanel({
    super.key,
    required this.activeTool,
    required this.statusText,
    this.onClose,
  });

  final String? activeTool;
  final String statusText;
  final VoidCallback? onClose;

  @override
  Widget build(BuildContext context) {
    if (activeTool == null) return const SizedBox.shrink();

    return SizedBox(
      width: 350,
      child: GlassPanel(
        tint: JarvisTheme.sectionChat,
        borderRadius: 16,
        padding: EdgeInsets.zero,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 8, 12),
              child: Row(
                children: [
                  Icon(
                    _iconForTool(activeTool!),
                    size: 18,
                    color: JarvisTheme.sectionChat,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _titleForTool(activeTool!),
                      style: const TextStyle(
                        color: JarvisTheme.sectionChat,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  if (onClose != null)
                    IconButton(
                      icon: Icon(
                        Icons.close,
                        size: 16,
                        color: JarvisTheme.textSecondary,
                      ),
                      onPressed: onClose,
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(
                        minWidth: 32,
                        minHeight: 32,
                      ),
                    ),
                ],
              ),
            ),

            // Divider
            Container(
              height: 1,
              color: JarvisTheme.sectionChat.withValues(alpha: 0.12),
            ),

            // Content area
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: _buildContent(context),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context) {
    if (_isSearchTool(activeTool!)) {
      return _SearchContent(statusText: statusText);
    }
    if (_isCodeTool(activeTool!)) {
      return _CodeContent(statusText: statusText);
    }
    return _GenericContent(
      toolName: activeTool!,
      statusText: statusText,
    );
  }

  static bool _isSearchTool(String tool) =>
      tool == 'web_search' || tool == 'search_and_read';

  static bool _isCodeTool(String tool) =>
      tool == 'run_python' || tool == 'analyze_code';

  static IconData _iconForTool(String tool) {
    if (_isSearchTool(tool)) return Icons.search;
    if (_isCodeTool(tool)) return Icons.terminal;
    return Icons.build_circle_outlined;
  }

  static String _titleForTool(String tool) {
    if (_isSearchTool(tool)) return 'Web Search';
    if (_isCodeTool(tool)) return 'Code Execution';
    // Format tool name nicely: snake_case -> Title Case
    return tool
        .split('_')
        .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
  }
}

// ── Search Content ─────────────────────────────────────────────────────

class _SearchContent extends StatefulWidget {
  const _SearchContent({required this.statusText});

  final String statusText;

  @override
  State<_SearchContent> createState() => _SearchContentState();
}

class _SearchContentState extends State<_SearchContent>
    with SingleTickerProviderStateMixin {
  late final AnimationController _dotsController;

  @override
  void initState() {
    super.initState();
    _dotsController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
  }

  @override
  void dispose() {
    _dotsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Animated "Searching..." text
        AnimatedBuilder(
          animation: _dotsController,
          builder: (context, _) {
            final dotCount = (_dotsController.value * 4).floor() % 4;
            final dots = '.' * dotCount;
            return Text(
              'Searching$dots',
              style: const TextStyle(
                color: JarvisTheme.sectionChat,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            );
          },
        ),
        const SizedBox(height: 16),

        // Progress bar
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: AnimatedBuilder(
            animation: _dotsController,
            builder: (context, _) {
              return LinearProgressIndicator(
                value: null,
                backgroundColor:
                    JarvisTheme.sectionChat.withValues(alpha: 0.08),
                valueColor: AlwaysStoppedAnimation<Color>(
                  JarvisTheme.sectionChat.withValues(alpha: 0.6),
                ),
                minHeight: 3,
              );
            },
          ),
        ),
        const SizedBox(height: 16),

        // Status text
        if (widget.statusText.isNotEmpty)
          Text(
            widget.statusText,
            style: TextStyle(
              color: JarvisTheme.textSecondary,
              fontSize: 12,
              height: 1.5,
            ),
          ),
      ],
    );
  }
}

// ── Code Content ───────────────────────────────────────────────────────

class _CodeContent extends StatelessWidget {
  const _CodeContent({required this.statusText});

  final String statusText;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Executing code...',
          style: TextStyle(
            color: JarvisTheme.sectionChat,
            fontSize: 13,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 12),

        // Terminal-style output area
        Expanded(
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: JarvisTheme.codeBlockBg,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: JarvisTheme.sectionChat.withValues(alpha: 0.15),
              ),
            ),
            child: SingleChildScrollView(
              child: Text(
                statusText.isNotEmpty ? statusText : '> Running...',
                style: TextStyle(
                  fontFamily: 'JetBrains Mono',
                  fontSize: 11,
                  color: JarvisTheme.matrix,
                  height: 1.6,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ── Generic Tool Content ───────────────────────────────────────────────

class _GenericContent extends StatelessWidget {
  const _GenericContent({
    required this.toolName,
    required this.statusText,
  });

  final String toolName;
  final String statusText;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Tool name badge
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: JarvisTheme.sectionChat.withValues(alpha: 0.10),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: JarvisTheme.sectionChat.withValues(alpha: 0.20),
            ),
          ),
          child: Text(
            toolName,
            style: const TextStyle(
              fontFamily: 'JetBrains Mono',
              fontSize: 11,
              color: JarvisTheme.sectionChat,
            ),
          ),
        ),
        const SizedBox(height: 16),

        // Spinner
        Center(
          child: SizedBox(
            width: 28,
            height: 28,
            child: CircularProgressIndicator(
              strokeWidth: 2.5,
              valueColor: AlwaysStoppedAnimation<Color>(
                JarvisTheme.sectionChat.withValues(alpha: 0.6),
              ),
            ),
          ),
        ),
        const SizedBox(height: 16),

        // Status
        if (statusText.isNotEmpty)
          Text(
            statusText,
            style: TextStyle(
              color: JarvisTheme.textSecondary,
              fontSize: 12,
              height: 1.5,
            ),
          ),
      ],
    );
  }
}
