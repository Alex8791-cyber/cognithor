import 'dart:math';

import 'package:flutter/material.dart';

/// Animated radial gauge with neon glow for metrics display.
///
/// Renders a 270-degree arc with a proportional value arc, glow effect,
/// and centered label/value text. Value changes animate smoothly.
class RadialGauge extends StatefulWidget {
  const RadialGauge({
    super.key,
    required this.value,
    required this.label,
    required this.color,
    this.size = 120,
    this.strokeWidth = 8,
    this.valueText,
  });

  /// Current value from 0.0 to 1.0.
  final double value;

  /// Label shown below the value text.
  final String label;

  /// Neon color for the value arc and glow.
  final Color color;

  /// Overall widget size (width and height).
  final double size;

  /// Stroke width of the arc.
  final double strokeWidth;

  /// Optional text override for the center (e.g. "45ms" instead of "45%").
  final String? valueText;

  @override
  State<RadialGauge> createState() => _RadialGaugeState();
}

class _RadialGaugeState extends State<RadialGauge>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  double _previousValue = 0;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _animation = Tween<double>(begin: 0, end: widget.value).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutQuart),
    );
    _previousValue = widget.value;
    _controller.forward();
  }

  @override
  void didUpdateWidget(covariant RadialGauge oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.value != widget.value) {
      _previousValue = _animation.value;
      _animation = Tween<double>(
        begin: _previousValue,
        end: widget.value,
      ).animate(
        CurvedAnimation(parent: _controller, curve: Curves.easeOutQuart),
      );
      _controller
        ..reset()
        ..forward();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ListenableBuilder(
      listenable: _animation,
      builder: (context, child) {
        final animatedValue = _animation.value.clamp(0.0, 1.0);
        final displayText = widget.valueText ??
            '${(animatedValue * 100).round()}%';

        return SizedBox(
          width: widget.size,
          height: widget.size,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CustomPaint(
                size: Size(widget.size, widget.size),
                painter: _RadialGaugePainter(
                  value: animatedValue,
                  color: widget.color,
                  strokeWidth: widget.strokeWidth,
                ),
              ),
              Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    displayText,
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontSize: widget.size * 0.18,
                      fontWeight: FontWeight.bold,
                      color: widget.color,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    widget.label,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontSize: widget.size * 0.09,
                    ),
                    textAlign: TextAlign.center,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}

class _RadialGaugePainter extends CustomPainter {
  _RadialGaugePainter({
    required this.value,
    required this.color,
    required this.strokeWidth,
  });

  final double value;
  final Color color;
  final double strokeWidth;

  static const double _startAngle = 135 * pi / 180; // bottom-left
  static const double _sweepAngle = 270 * pi / 180; // 270-degree arc

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (min(size.width, size.height) - strokeWidth * 2) / 2;
    final rect = Rect.fromCircle(center: center, radius: radius);

    // Background arc (dark)
    final bgPaint = Paint()
      ..color = color.withValues(alpha: 0.12)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(rect, _startAngle, _sweepAngle, false, bgPaint);

    if (value <= 0) return;

    final valueSweep = _sweepAngle * value;

    // Glow layers (multiple draws at decreasing alpha for bloom effect)
    for (int i = 3; i >= 1; i--) {
      final glowPaint = Paint()
        ..color = color.withValues(alpha: 0.06 * i)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth + i * 4.0
        ..strokeCap = StrokeCap.round
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, i * 3.0);
      canvas.drawArc(rect, _startAngle, valueSweep, false, glowPaint);
    }

    // Value arc (neon colored)
    final valuePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(rect, _startAngle, valueSweep, false, valuePaint);
  }

  @override
  bool shouldRepaint(covariant _RadialGaugePainter old) =>
      old.value != value || old.color != color || old.strokeWidth != strokeWidth;
}
