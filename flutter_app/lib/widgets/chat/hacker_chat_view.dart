import 'dart:math';
import 'package:flutter/material.dart';
import 'package:jarvis_ui/providers/chat_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

/// Matrix green color used throughout hacker mode.
const _matrixGreen = Color(0xFF00FF41);

/// Terminal-style chat view for hacker mode.
///
/// Displays messages in a monospaced terminal format with timestamps,
/// role prefixes, and a subtle Matrix rain background effect.
class HackerChatView extends StatefulWidget {
  const HackerChatView({
    super.key,
    required this.messages,
    required this.streamingText,
    required this.isStreaming,
    required this.activeTool,
    required this.scrollController,
  });

  final List<ChatMessage> messages;
  final String streamingText;
  final bool isStreaming;
  final String? activeTool;
  final ScrollController scrollController;

  @override
  State<HackerChatView> createState() => _HackerChatViewState();
}

class _HackerChatViewState extends State<HackerChatView>
    with SingleTickerProviderStateMixin {
  late final AnimationController _rainController;

  @override
  void initState() {
    super.initState();
    _rainController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 8),
    )..repeat();
  }

  @override
  void dispose() {
    _rainController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black,
      child: Stack(
        children: [
          // Matrix rain background
          Positioned.fill(
            child: AnimatedBuilder(
              animation: _rainController,
              builder: (context, _) {
                return CustomPaint(
                  painter: _MatrixRainPainter(
                    time: _rainController.value,
                  ),
                );
              },
            ),
          ),

          // Terminal content
          ListView.builder(
            controller: widget.scrollController,
            padding: const EdgeInsets.all(16),
            itemCount: widget.messages.length +
                (widget.isStreaming ? 1 : 0) +
                (widget.activeTool != null ? 1 : 0),
            itemBuilder: (context, index) {
              // Active tool line
              if (widget.activeTool != null && index == widget.messages.length) {
                return _buildToolLine(widget.activeTool!);
              }

              // Streaming line
              final streamIndex = widget.messages.length +
                  (widget.activeTool != null ? 1 : 0);
              if (widget.isStreaming && index == streamIndex) {
                return _buildTerminalLine(
                  timestamp: DateTime.now(),
                  prefix: 'ASST',
                  text: widget.streamingText,
                  color: _matrixGreen,
                  showCursor: true,
                );
              }

              // Regular messages
              final msg = widget.messages[index];
              return _buildMessageLine(msg);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildMessageLine(ChatMessage msg) {
    final prefix = switch (msg.role) {
      MessageRole.user => 'USER',
      MessageRole.assistant => 'ASST',
      MessageRole.system => 'SYS!',
    };
    final color = switch (msg.role) {
      MessageRole.user => Colors.white,
      MessageRole.assistant => _matrixGreen,
      MessageRole.system => JarvisTheme.red,
    };

    return _buildTerminalLine(
      timestamp: msg.timestamp,
      prefix: prefix,
      text: msg.text,
      color: color,
    );
  }

  Widget _buildToolLine(String tool) {
    return _buildTerminalLine(
      timestamp: DateTime.now(),
      prefix: 'TOOL',
      text: tool,
      color: JarvisTheme.sectionChat,
    );
  }

  Widget _buildTerminalLine({
    required DateTime timestamp,
    required String prefix,
    required String text,
    required Color color,
    bool showCursor = false,
  }) {
    final ts = '${_pad(timestamp.hour)}:${_pad(timestamp.minute)}:${_pad(timestamp.second)}';
    final mono = JarvisTheme.monoTextTheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: SelectableText.rich(
        TextSpan(
          style: mono.bodyMedium?.copyWith(
            fontSize: 13,
            height: 1.6,
          ),
          children: [
            TextSpan(
              text: '[$ts] ',
              style: TextStyle(color: _matrixGreen.withValues(alpha: 0.5)),
            ),
            TextSpan(
              text: '$prefix > ',
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: text,
              style: TextStyle(color: color.withValues(alpha: 0.9)),
            ),
            if (showCursor)
              const TextSpan(
                text: '\u2588', // block cursor
                style: TextStyle(color: _matrixGreen),
              ),
          ],
        ),
      ),
    );
  }

  static String _pad(int n) => n.toString().padLeft(2, '0');
}

// ── Matrix Rain Background Painter ─────────────────────────────────────

class _MatrixRainPainter extends CustomPainter {
  _MatrixRainPainter({required this.time});

  final double time;

  // Characters used in the Matrix rain
  static const _chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#\$%^&*';
  static final _random = Random(42); // fixed seed for consistent columns
  // Pre-generate column data once
  static List<double>? _columnSpeeds;
  static List<double>? _columnOffsets;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    const charWidth = 14.0;
    const charHeight = 18.0;
    final cols = (size.width / charWidth).ceil();
    final rows = (size.height / charHeight).ceil();

    // Initialize column data once
    if (_columnSpeeds == null || _columnSpeeds!.length != cols) {
      _columnSpeeds = List.generate(cols, (_) => 0.3 + _random.nextDouble() * 0.7);
      _columnOffsets = List.generate(cols, (_) => _random.nextDouble() * rows);
    }

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    for (int col = 0; col < cols; col++) {
      final speed = _columnSpeeds![col];
      final offset = _columnOffsets![col];
      final currentRow = ((time * speed * rows * 2 + offset) % (rows + 10)).floor();

      for (int row = 0; row < rows; row++) {
        final distFromHead = currentRow - row;
        if (distFromHead < 0 || distFromHead > 12) continue;

        // Fade out further from the head
        final alpha = (1.0 - distFromHead / 12.0) * 0.06; // Very low opacity
        if (alpha <= 0) continue;

        final charIndex = (col * 17 + row * 31 + (time * 10).floor()) % _chars.length;
        final char = _chars[charIndex];

        textPainter.text = TextSpan(
          text: char,
          style: TextStyle(
            fontFamily: 'JetBrains Mono',
            fontSize: 12,
            color: _matrixGreen.withValues(alpha: alpha),
          ),
        );
        textPainter.layout();
        textPainter.paint(
          canvas,
          Offset(col * charWidth, row * charHeight),
        );
      }
    }
  }

  @override
  bool shouldRepaint(_MatrixRainPainter oldDelegate) =>
      oldDelegate.time != time;
}
