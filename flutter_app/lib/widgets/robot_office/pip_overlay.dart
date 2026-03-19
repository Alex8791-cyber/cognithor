import 'dart:math';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/robot_office/robot_office_widget.dart';

// ---------------------------------------------------------------------------
// Robot Office Picture-in-Picture overlay
// ---------------------------------------------------------------------------

/// A draggable, resizable floating window that shows the Robot Office.
///
/// Wrap this around the main app content so the PiP floats above everything,
/// including bottom navigation and app bars.
class RobotOfficePip extends StatefulWidget {
  const RobotOfficePip({
    super.key,
    required this.child,
  });

  /// The main app content underneath the overlay.
  final Widget child;

  @override
  State<RobotOfficePip> createState() => _RobotOfficePipState();
}

class _RobotOfficePipState extends State<RobotOfficePip>
    with SingleTickerProviderStateMixin {
  // Position of the PiP window (top-left corner).
  double _pipX = 20;
  double _pipY = 80;

  // Size modes.
  bool _expanded = false; // false = small PiP, true = large view
  bool _minimized = false; // just shows a small robot icon

  // Small PiP dimensions.
  static const double _smallWidth = 280;
  static const double _smallHeight = 180;

  // Large PiP dimensions.
  static const double _largeWidth = 500;
  static const double _largeHeight = 320;

  // Minimized bubble size.
  static const double _bubbleSize = 48;

  // Snap threshold in logical pixels.
  static const double _snapThreshold = 40;

  // Pulse animation for minimized bubble.
  late final AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  // Current dimensions based on state.
  double get _width => _expanded ? _largeWidth : _smallWidth;
  double get _height => _expanded ? _largeHeight : _smallHeight;

  /// Clamps PiP position so it stays within the screen.
  void _clampPosition(Size screenSize) {
    if (_minimized) {
      _pipX = _pipX.clamp(0, screenSize.width - _bubbleSize);
      _pipY = _pipY.clamp(0, screenSize.height - _bubbleSize);
    } else {
      _pipX = _pipX.clamp(0, screenSize.width - _width);
      _pipY = _pipY.clamp(0, screenSize.height - _height);
    }
  }

  /// Snaps the PiP to the nearest screen edge if close enough.
  void _snapToEdges(Size screenSize) {
    final double elementWidth = _minimized ? _bubbleSize : _width;
    final double elementHeight = _minimized ? _bubbleSize : _height;

    // Snap left.
    if (_pipX < _snapThreshold) {
      _pipX = 8;
    }
    // Snap right.
    if (_pipX > screenSize.width - elementWidth - _snapThreshold) {
      _pipX = screenSize.width - elementWidth - 8;
    }
    // Snap top.
    if (_pipY < _snapThreshold) {
      _pipY = 8;
    }
    // Snap bottom.
    if (_pipY > screenSize.height - elementHeight - _snapThreshold) {
      _pipY = screenSize.height - elementHeight - 8;
    }
  }

  void _onPanUpdate(DragUpdateDetails d) {
    setState(() {
      _pipX += d.delta.dx;
      _pipY += d.delta.dy;
    });
  }

  void _onPanEnd(DragEndDetails _) {
    final screenSize = MediaQuery.of(context).size;
    setState(() {
      _snapToEdges(screenSize);
      _clampPosition(screenSize);
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    // Ensure position is valid after layout changes (e.g. rotation).
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        final clamped = _clampedPosition(screenSize);
        if (clamped.dx != _pipX || clamped.dy != _pipY) {
          setState(() {
            _pipX = clamped.dx;
            _pipY = clamped.dy;
          });
        }
      }
    });

    return Stack(
      children: [
        widget.child,
        if (_minimized)
          _buildMinimizedBubble()
        else
          _buildPipWindow(),
      ],
    );
  }

  Offset _clampedPosition(Size screenSize) {
    final double elementWidth = _minimized ? _bubbleSize : _width;
    final double elementHeight = _minimized ? _bubbleSize : _height;
    return Offset(
      _pipX.clamp(0, max(0, screenSize.width - elementWidth)),
      _pipY.clamp(0, max(0, screenSize.height - elementHeight)),
    );
  }

  // ── Minimized bubble ────────────────────────────────────────────────────

  Widget _buildMinimizedBubble() {
    return Positioned(
      left: _pipX,
      top: _pipY,
      child: GestureDetector(
        onPanUpdate: _onPanUpdate,
        onPanEnd: _onPanEnd,
        onTap: () => setState(() => _minimized = false),
        child: ListenableBuilder(
          listenable: _pulseController,
          builder: (context, animChild) {
            final scale = 1.0 + _pulseController.value * 0.08;
            return Transform.scale(
              scale: scale,
              child: animChild,
            );
          },
          child: Container(
            width: _bubbleSize,
            height: _bubbleSize,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: const Color(0xFF0c1220),
              border: Border.all(
                color: JarvisTheme.accent.withValues(alpha: 0.7),
                width: 2,
              ),
              boxShadow: [
                BoxShadow(
                  color: JarvisTheme.accent.withValues(alpha: 0.3),
                  blurRadius: 12,
                  spreadRadius: 2,
                ),
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.4),
                  blurRadius: 8,
                ),
              ],
            ),
            child: const Icon(
              Icons.smart_toy,
              color: JarvisTheme.info,
              size: 24,
            ),
          ),
        ),
      ),
    );
  }

  // ── PiP window ──────────────────────────────────────────────────────────

  Widget _buildPipWindow() {
    return Positioned(
      left: _pipX,
      top: _pipY,
      child: GestureDetector(
        onPanUpdate: _onPanUpdate,
        onPanEnd: _onPanEnd,
        child: AnimatedContainer(
          duration: JarvisTheme.animDuration,
          curve: JarvisTheme.animCurve,
          width: _width,
          height: _height,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.5),
                blurRadius: 24,
                spreadRadius: 2,
                offset: const Offset(0, 8),
              ),
              BoxShadow(
                color: JarvisTheme.accent.withValues(alpha: 0.08),
                blurRadius: 40,
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Stack(
                children: [
                  // Robot Office animation fills the entire window.
                  Positioned.fill(
                    child: Consumer<PipProvider>(
                      builder: (ctx, pip, _) =>
                          RobotOfficeWidget(isRunning: pip.busy),
                    ),
                  ),

                  // Semi-transparent glassmorphism layer on border.
                  Positioned.fill(
                    child: IgnorePointer(
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.06),
                            width: 1,
                          ),
                        ),
                      ),
                    ),
                  ),

                  // Control bar at top-right.
                  Positioned(
                    top: 6,
                    right: 6,
                    child: _PipControlBar(
                      onMinimize: () => setState(() => _minimized = true),
                      onExpand: () => setState(() => _expanded = !_expanded),
                      onClose: () {
                        // Hide the entire PiP by removing it from the stack.
                        // The parent (MainShell) controls actual visibility
                        // via PipProvider, so we minimize to bubble instead.
                        setState(() => _minimized = true);
                      },
                      isExpanded: _expanded,
                    ),
                  ),

                  // Drag indicator at top-left.
                  Positioned(
                    top: 8,
                    left: 10,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.drag_indicator,
                          size: 16,
                          color: Colors.white.withValues(alpha: 0.3),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          'Robot Office',
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: Colors.white.withValues(alpha: 0.5),
                            letterSpacing: 0.5,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// PiP Control Bar — three small buttons for minimize / expand / close
// ---------------------------------------------------------------------------

class _PipControlBar extends StatelessWidget {
  const _PipControlBar({
    required this.onMinimize,
    required this.onExpand,
    required this.onClose,
    required this.isExpanded,
  });

  final VoidCallback onMinimize;
  final VoidCallback onExpand;
  final VoidCallback onClose;
  final bool isExpanded;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _ControlButton(
            icon: Icons.minimize,
            tooltip: 'Minimize',
            onTap: onMinimize,
          ),
          _ControlButton(
            icon: isExpanded
                ? Icons.close_fullscreen
                : Icons.open_in_full,
            tooltip: isExpanded ? 'Shrink' : 'Expand',
            onTap: onExpand,
          ),
          _ControlButton(
            icon: Icons.close,
            tooltip: 'Close',
            onTap: onClose,
            color: JarvisTheme.red,
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Individual control button
// ---------------------------------------------------------------------------

class _ControlButton extends StatefulWidget {
  const _ControlButton({
    required this.icon,
    required this.tooltip,
    required this.onTap,
    this.color,
  });

  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;
  final Color? color;

  @override
  State<_ControlButton> createState() => _ControlButtonState();
}

class _ControlButtonState extends State<_ControlButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final color = widget.color ?? Colors.white;
    final effectiveAlpha = _hovered ? 0.9 : 0.5;

    return Tooltip(
      message: widget.tooltip,
      child: MouseRegion(
        onEnter: (_) => setState(() => _hovered = true),
        onExit: (_) => setState(() => _hovered = false),
        child: GestureDetector(
          onTap: widget.onTap,
          child: Padding(
            padding: const EdgeInsets.all(4),
            child: Icon(
              widget.icon,
              size: 14,
              color: color.withValues(alpha: effectiveAlpha),
            ),
          ),
        ),
      ),
    );
  }
}
