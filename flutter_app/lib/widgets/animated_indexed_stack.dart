import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// A drop-in replacement for [IndexedStack] that cross-fades and subtly slides
/// between children when the [index] changes.
class AnimatedIndexedStack extends StatefulWidget {
  const AnimatedIndexedStack({
    super.key,
    required this.index,
    required this.children,
    this.duration = CognithorTheme.animDuration,
    this.curve = CognithorTheme.animCurve,
    this.slideOffset = 50.0,
  });

  final int index;
  final List<Widget> children;
  final Duration duration;
  final Curve curve;

  /// Horizontal pixel offset used for the slide component.  Positive values
  /// slide incoming content from the right; the outgoing content slides left.
  final double slideOffset;

  @override
  State<AnimatedIndexedStack> createState() => _AnimatedIndexedStackState();
}

class _AnimatedIndexedStackState extends State<AnimatedIndexedStack>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeIn;
  late Animation<Offset> _slideIn;

  int _displayIndex = 0;

  @override
  void initState() {
    super.initState();
    _displayIndex = widget.index;
    _controller = AnimationController(vsync: this, duration: widget.duration);

    _buildAnimations();
    // Start fully visible.
    _controller.value = 1.0;
  }

  void _buildAnimations() {
    final curved = CurvedAnimation(parent: _controller, curve: widget.curve);
    _fadeIn = Tween<double>(begin: 0.0, end: 1.0).animate(curved);
    _slideIn = Tween<Offset>(
      begin: Offset(widget.slideOffset / 300, 0),
      end: Offset.zero,
    ).animate(curved);
  }

  @override
  void didUpdateWidget(AnimatedIndexedStack old) {
    super.didUpdateWidget(old);

    if (old.duration != widget.duration || old.curve != widget.curve) {
      _controller.duration = widget.duration;
      _buildAnimations();
    }

    if (old.index != widget.index) {
      // Fade out current, then swap index and fade in.
      _controller.reverse().then((_) {
        if (!mounted) return;
        setState(() => _displayIndex = widget.index);
        _controller.forward();
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeIn,
      child: SlideTransition(
        position: _slideIn,
        child: IndexedStack(index: _displayIndex, children: widget.children),
      ),
    );
  }
}
