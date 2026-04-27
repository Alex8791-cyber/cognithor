import 'package:flutter/material.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class PipelineIndicator extends StatelessWidget {
  const PipelineIndicator({super.key, required this.phases});

  final List<PipelinePhase> phases;

  static const _phaseOrder = ['plan', 'gate', 'execute', 'replan', 'complete'];

  Color _phaseColor(String status) => switch (status) {
    'start' => CognithorTheme.accent,
    'done' => CognithorTheme.green,
    'error' => CognithorTheme.red,
    _ => CognithorTheme.textTertiary,
  };

  IconData _phaseIcon(String phase) => switch (phase) {
    'plan' => Icons.psychology,
    'gate' => Icons.shield,
    'execute' => Icons.build,
    'replan' => Icons.replay,
    'complete' => Icons.check_circle,
    _ => Icons.circle,
  };

  @override
  Widget build(BuildContext context) {
    final phaseMap = {for (final p in phases) p.phase: p};

    // Build the ordered list of visible phases
    final visiblePhases = _phaseOrder
        .where((name) => phaseMap.containsKey(name))
        .toList();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        border: Border(
          bottom: BorderSide(color: Theme.of(context).dividerColor),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          for (int i = 0; i < visiblePhases.length; i++) ...[
            _PhaseChip(
              phase: phaseMap[visiblePhases[i]]!,
              icon: _phaseIcon(visiblePhases[i]),
              color: _phaseColor(phaseMap[visiblePhases[i]]!.status),
            ),
            if (i < visiblePhases.length - 1)
              _PhaseConnector(
                completed: phaseMap[visiblePhases[i]]!.status == 'done',
              ),
          ],
        ],
      ),
    );
  }
}

/// Connector line between phase chips.
class _PhaseConnector extends StatelessWidget {
  const _PhaseConnector({required this.completed});

  final bool completed;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 6),
      child: SizedBox(
        width: 24,
        height: 2,
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: completed
                ? CognithorTheme.green.withAlpha(150)
                : Theme.of(context).dividerColor,
            borderRadius: BorderRadius.circular(1),
          ),
        ),
      ),
    );
  }
}

class _PhaseChip extends StatefulWidget {
  const _PhaseChip({
    required this.phase,
    required this.icon,
    required this.color,
  });

  final PipelinePhase phase;
  final IconData icon;
  final Color color;

  @override
  State<_PhaseChip> createState() => _PhaseChipState();
}

class _PhaseChipState extends State<_PhaseChip>
    with SingleTickerProviderStateMixin {
  late final AnimationController _glowController;
  late final Animation<double> _glowAnimation;

  @override
  void initState() {
    super.initState();
    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    _glowAnimation = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _glowController, curve: Curves.easeInOut),
    );
    if (widget.phase.status == 'start') {
      _glowController.repeat(reverse: true);
    }
  }

  @override
  void didUpdateWidget(covariant _PhaseChip oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.phase.status == 'start' && !_glowController.isAnimating) {
      _glowController.repeat(reverse: true);
    } else if (widget.phase.status != 'start' && _glowController.isAnimating) {
      _glowController.stop();
      _glowController.value = 1.0;
    }
  }

  @override
  void dispose() {
    _glowController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isActive = widget.phase.status == 'start';
    final isDone = widget.phase.status == 'done';
    final color = widget.color;

    return AnimatedBuilder(
      animation: _glowAnimation,
      builder: (context, child) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: color.withAlpha(
              isActive ? (25 * _glowAnimation.value).round() : 20,
            ),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: color.withAlpha(
                isActive ? (80 * _glowAnimation.value).round() : 50,
              ),
            ),
            boxShadow: isActive
                ? [
                    BoxShadow(
                      color: color.withAlpha(
                        (40 * _glowAnimation.value).round(),
                      ),
                      blurRadius: 8,
                      spreadRadius: 1,
                    ),
                  ]
                : [],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (isActive)
                _GlowingDot(color: color, animation: _glowAnimation)
              else if (isDone)
                Icon(Icons.check, size: 14, color: color)
              else
                Icon(widget.icon, size: 14, color: color),
              const SizedBox(width: 6),
              Text(
                widget.phase.phase,
                style: TextStyle(
                  color: color,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.3,
                ),
              ),
              if (widget.phase.elapsedMs > 0) ...[
                const SizedBox(width: 4),
                Text(
                  '${widget.phase.elapsedMs}ms',
                  style: TextStyle(color: color.withAlpha(150), fontSize: 10),
                ),
              ],
            ],
          ),
        );
      },
    );
  }
}

/// Animated glowing dot for active phases.
class _GlowingDot extends StatelessWidget {
  const _GlowingDot({required this.color, required this.animation});

  final Color color;
  final Animation<double> animation;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 14,
      height: 14,
      child: Center(
        child: Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color,
            boxShadow: [
              BoxShadow(
                color: color.withAlpha((120 * animation.value).round()),
                blurRadius: 6 * animation.value,
                spreadRadius: 2 * animation.value,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
