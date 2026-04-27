import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// Inline branch navigator shown at fork points: < 1/3 >
class BranchNavigator extends StatelessWidget {
  const BranchNavigator({
    super.key,
    required this.currentIndex,
    required this.totalBranches,
    required this.onPrevious,
    required this.onNext,
  });

  final int currentIndex;
  final int totalBranches;
  final VoidCallback onPrevious;
  final VoidCallback onNext;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: CognithorTheme.accent.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: CognithorTheme.accent.withValues(alpha: 0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _NavBtn(
            icon: Icons.chevron_left,
            enabled: currentIndex > 0,
            onTap: onPrevious,
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6),
            child: Text(
              '${currentIndex + 1} / $totalBranches',
              style: TextStyle(
                fontSize: 11,
                color: CognithorTheme.accent,
                fontFamily: 'monospace',
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          _NavBtn(
            icon: Icons.chevron_right,
            enabled: currentIndex < totalBranches - 1,
            onTap: onNext,
          ),
        ],
      ),
    );
  }
}

class _NavBtn extends StatelessWidget {
  const _NavBtn({
    required this.icon,
    required this.enabled,
    required this.onTap,
  });
  final IconData icon;
  final bool enabled;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(10),
      onTap: enabled ? onTap : null,
      child: Icon(
        icon,
        size: 18,
        color: enabled ? CognithorTheme.accent : CognithorTheme.textTertiary,
      ),
    );
  }
}
