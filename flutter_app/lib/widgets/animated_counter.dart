import 'package:flutter/material.dart';

/// A widget that smoothly animates between numeric values.
///
/// When [value] changes, the displayed number counts up (or down) from the
/// old value to the new one using an [easeOutQuart] curve.
class AnimatedCounter extends StatefulWidget {
  const AnimatedCounter({
    super.key,
    required this.value,
    this.duration = const Duration(milliseconds: 800),
    this.style,
    this.prefix = '',
    this.suffix = '',
  });

  final num value;
  final Duration duration;
  final TextStyle? style;
  final String prefix;
  final String suffix;

  @override
  State<AnimatedCounter> createState() => _AnimatedCounterState();
}

class _AnimatedCounterState extends State<AnimatedCounter>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  late double _oldValue;
  late double _currentTarget;

  @override
  void initState() {
    super.initState();
    _oldValue = widget.value.toDouble();
    _currentTarget = _oldValue;
    _controller = AnimationController(vsync: this, duration: widget.duration);
    _animation = _buildTween(_oldValue, _currentTarget);
  }

  Animation<double> _buildTween(double begin, double end) {
    return Tween<double>(
      begin: begin,
      end: end,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutQuart));
  }

  @override
  void didUpdateWidget(AnimatedCounter oldWidget) {
    super.didUpdateWidget(oldWidget);
    final newTarget = widget.value.toDouble();
    if (newTarget != _currentTarget) {
      // Start from wherever we currently are (handles mid-animation changes).
      _oldValue = _animation.value;
      _currentTarget = newTarget;
      _animation = _buildTween(_oldValue, _currentTarget);
      _controller
        ..reset()
        ..forward();
    }
    if (widget.duration != oldWidget.duration) {
      _controller.duration = widget.duration;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  String _formatValue(double v) {
    // Show decimals only when the target value is fractional.
    if (_currentTarget == _currentTarget.roundToDouble()) {
      return v.round().toString();
    }
    return v.toStringAsFixed(1);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, _) {
        return Text(
          '${widget.prefix}${_formatValue(_animation.value)}${widget.suffix}',
          style: widget.style,
        );
      },
    );
  }
}
