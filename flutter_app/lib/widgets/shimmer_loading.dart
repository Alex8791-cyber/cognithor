import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// A premium shimmer/skeleton loading effect.
///
/// Renders [count] rounded rectangles with a smooth gradient sweep
/// animation that moves left-to-right on a 1.5 s loop.
class ShimmerLoading extends StatefulWidget {
  const ShimmerLoading({
    super.key,
    this.count = 3,
    this.height = 80,
    this.borderRadius = 12,
  });

  final int count;
  final double height;
  final double borderRadius;

  @override
  State<ShimmerLoading> createState() => _ShimmerLoadingState();
}

class _ShimmerLoadingState extends State<ShimmerLoading>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    final baseColor = theme.cardColor;
    final highlightColor = isDark
        ? HSLColor.fromColor(baseColor)
              .withLightness(
                (HSLColor.fromColor(baseColor).lightness + 0.06).clamp(
                  0.0,
                  1.0,
                ),
              )
              .toColor()
        : HSLColor.fromColor(baseColor)
              .withLightness(
                (HSLColor.fromColor(baseColor).lightness - 0.04).clamp(
                  0.0,
                  1.0,
                ),
              )
              .toColor();

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          children: List.generate(widget.count, (i) {
            return Padding(
              padding: EdgeInsets.only(
                top: i > 0 ? CognithorTheme.spacingSm : 0,
              ),
              child: _ShimmerRect(
                height: widget.height,
                borderRadius: widget.borderRadius,
                baseColor: baseColor,
                highlightColor: highlightColor,
                progress: _controller.value,
              ),
            );
          }),
        );
      },
    );
  }
}

class _ShimmerRect extends StatelessWidget {
  const _ShimmerRect({
    required this.height,
    required this.borderRadius,
    required this.baseColor,
    required this.highlightColor,
    required this.progress,
  });

  final double height;
  final double borderRadius;
  final Color baseColor;
  final Color highlightColor;
  final double progress;

  @override
  Widget build(BuildContext context) {
    // The gradient slides from left (-1) to right (+2) over the animation.
    final dx = -1.0 + 3.0 * progress;

    return Container(
      height: height,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(borderRadius),
        gradient: LinearGradient(
          begin: Alignment(dx - 0.6, 0),
          end: Alignment(dx + 0.6, 0),
          colors: [baseColor, highlightColor, baseColor],
          stops: const [0.0, 0.5, 1.0],
        ),
      ),
    );
  }
}
