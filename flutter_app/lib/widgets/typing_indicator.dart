import 'dart:math';
import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/glass_panel.dart';

/// Holographic waveform typing indicator shown when the assistant is thinking.
class TypingIndicator extends StatefulWidget {
  const TypingIndicator({super.key});

  @override
  State<TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Align(
      alignment: Alignment.centerLeft,
      child: GlassPanel(
        tint: CognithorTheme.sectionChat,
        borderRadius: 16,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 120,
              height: 30,
              child: AnimatedBuilder(
                animation: _controller,
                builder: (context, _) {
                  return CustomPaint(
                    painter: WaveformPainter(
                      time: _controller.value * 2 * pi,
                      color: CognithorTheme.sectionChat,
                    ),
                  );
                },
              ),
            ),
            const SizedBox(width: 10),
            Text(
              l.thinking,
              style: const TextStyle(
                color: CognithorTheme.sectionChat,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Draws 3 sine waves at different frequencies for a holographic waveform effect.
class WaveformPainter extends CustomPainter {
  const WaveformPainter({required this.time, required this.color});

  final double time;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    for (int wave = 0; wave < 3; wave++) {
      final path = Path();
      final freq = 2.0 + wave * 0.8;
      final amp = size.height * (0.15 + wave * 0.08);
      final phase = wave * 1.2;

      path.moveTo(0, size.height / 2);
      for (double x = 0; x <= size.width; x += 2) {
        final y =
            size.height / 2 +
            sin((x / size.width) * freq * pi + time * 3 + phase) * amp;
        path.lineTo(x, y);
      }

      canvas.drawPath(
        path,
        Paint()
          ..color = color.withValues(alpha: 0.3 - wave * 0.08)
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round,
      );
    }
  }

  @override
  bool shouldRepaint(WaveformPainter oldDelegate) =>
      oldDelegate.time != time || oldDelegate.color != color;
}
