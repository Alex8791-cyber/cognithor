import 'package:flutter/material.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/chat/matrix_rain_painter.dart';

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
      decoration: const BoxDecoration(
        gradient: RadialGradient(
          center: Alignment.center,
          radius: 1.2,
          colors: [
            Color(0xFF001A00), // very dark green center
            Colors.black, // pure black edges
          ],
        ),
      ),
      child: Stack(
        children: [
          // Matrix rain background
          Positioned.fill(
            child: ListenableBuilder(
              listenable: _rainController,
              builder: (context, _) {
                return CustomPaint(
                  painter: MatrixRainPainter(time: _rainController.value * 8),
                );
              },
            ),
          ),

          // Terminal content
          ListView.builder(
            controller: widget.scrollController,
            padding: const EdgeInsets.all(16),
            itemCount:
                widget.messages.length +
                (widget.isStreaming ? 1 : 0) +
                (widget.activeTool != null ? 1 : 0),
            itemBuilder: (context, index) {
              // Active tool line
              if (widget.activeTool != null &&
                  index == widget.messages.length) {
                return _buildToolLine(widget.activeTool!);
              }

              // Streaming line
              final streamIndex =
                  widget.messages.length + (widget.activeTool != null ? 1 : 0);
              if (widget.isStreaming && index == streamIndex) {
                return _buildTerminalLine(
                  timestamp: DateTime.now(),
                  prefix: 'ASST',
                  text: widget.streamingText,
                  color: matrixGreen,
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
      MessageRole.assistant => matrixGreen,
      MessageRole.system => CognithorTheme.red,
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
      color: CognithorTheme.sectionChat,
    );
  }

  Widget _buildTerminalLine({
    required DateTime timestamp,
    required String prefix,
    required String text,
    required Color color,
    bool showCursor = false,
  }) {
    final ts =
        '${_pad(timestamp.hour)}:${_pad(timestamp.minute)}:${_pad(timestamp.second)}';
    final mono = CognithorTheme.monoTextTheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: SelectableText.rich(
        TextSpan(
          style: mono.bodyMedium?.copyWith(fontSize: 13, height: 1.6),
          children: [
            TextSpan(
              text: '[$ts] ',
              style: TextStyle(color: matrixGreen.withValues(alpha: 0.5)),
            ),
            TextSpan(
              text: '$prefix > ',
              style: TextStyle(color: color, fontWeight: FontWeight.bold),
            ),
            TextSpan(
              text: text,
              style: TextStyle(color: color.withValues(alpha: 0.9)),
            ),
            if (showCursor)
              const TextSpan(
                text: '\u2588', // block cursor
                style: TextStyle(color: matrixGreen),
              ),
          ],
        ),
      ),
    );
  }

  static String _pad(int n) => n.toString().padLeft(2, '0');
}
