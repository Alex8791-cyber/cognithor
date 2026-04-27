import 'package:flutter/material.dart';

/// An animated list wrapper that staggers children on first load.
///
/// Each child fades in and slides up with increasing delay, creating
/// a cascading entrance effect.
class StaggeredList extends StatefulWidget {
  const StaggeredList({
    super.key,
    required this.children,
    this.staggerDelay = const Duration(milliseconds: 50),
    this.animationDuration = const Duration(milliseconds: 400),
    this.curve = Curves.easeOutQuart,
  });

  final List<Widget> children;
  final Duration staggerDelay;
  final Duration animationDuration;
  final Curve curve;

  @override
  State<StaggeredList> createState() => _StaggeredListState();
}

class _StaggeredListState extends State<StaggeredList>
    with TickerProviderStateMixin {
  late List<AnimationController> _controllers;
  late List<Animation<double>> _fadeAnimations;
  late List<Animation<Offset>> _slideAnimations;

  @override
  void initState() {
    super.initState();
    _initAnimations();
    _startAnimations();
  }

  void _initAnimations() {
    _controllers = List.generate(
      widget.children.length,
      (index) =>
          AnimationController(vsync: this, duration: widget.animationDuration),
    );

    _fadeAnimations = _controllers.map((controller) {
      return CurvedAnimation(
        parent: controller,
        curve: widget.curve,
      ).drive(Tween<double>(begin: 0.0, end: 1.0));
    }).toList();

    _slideAnimations = _controllers.map((controller) {
      return CurvedAnimation(
        parent: controller,
        curve: widget.curve,
      ).drive(Tween<Offset>(begin: const Offset(0, 20), end: Offset.zero));
    }).toList();
  }

  Future<void> _startAnimations() async {
    for (var i = 0; i < _controllers.length; i++) {
      if (!mounted) return;
      if (i > 0) {
        await Future<void>.delayed(widget.staggerDelay);
      }
      if (!mounted) return;
      _controllers[i].forward();
    }
  }

  @override
  void didUpdateWidget(StaggeredList oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.children.length != widget.children.length) {
      _disposeControllers();
      _initAnimations();
      _startAnimations();
    }
  }

  void _disposeControllers() {
    for (final controller in _controllers) {
      controller.dispose();
    }
  }

  @override
  void dispose() {
    _disposeControllers();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: List.generate(widget.children.length, (index) {
        // Guard against rebuild during didUpdateWidget transition.
        if (index >= _controllers.length) {
          return widget.children[index];
        }
        return _StaggeredItem(
          controller: _controllers[index],
          fadeAnimation: _fadeAnimations[index],
          slideAnimation: _slideAnimations[index],
          child: widget.children[index],
        );
      }),
    );
  }
}

class _StaggeredItem extends AnimatedWidget {
  const _StaggeredItem({
    required AnimationController controller,
    required this.fadeAnimation,
    required this.slideAnimation,
    required this.child,
  }) : super(listenable: controller);

  final Animation<double> fadeAnimation;
  final Animation<Offset> slideAnimation;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Transform.translate(
      offset: slideAnimation.value,
      child: Opacity(opacity: fadeAnimation.value, child: child),
    );
  }
}
