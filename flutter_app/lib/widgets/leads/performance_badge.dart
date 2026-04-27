import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class PerformanceBadge extends StatelessWidget {
  const PerformanceBadge({
    super.key,
    required this.score,
    this.compact = false,
  });

  final int score;
  final bool compact;

  Color get _color {
    if (score >= 70) return CognithorTheme.green;
    if (score >= 40) return Colors.orange;
    if (score > 0) return CognithorTheme.red;
    return CognithorTheme.textSecondary;
  }

  @override
  Widget build(BuildContext context) {
    if (score <= 0) return const SizedBox.shrink();

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 4 : 8,
        vertical: compact ? 2 : 4,
      ),
      decoration: BoxDecoration(
        color: _color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(compact ? 4 : 8),
        border: Border.all(color: _color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.trending_up, size: compact ? 10 : 14, color: _color),
          SizedBox(width: compact ? 2 : 4),
          Text(
            '$score',
            style: TextStyle(
              color: _color,
              fontWeight: FontWeight.w700,
              fontSize: compact ? 9 : 12,
            ),
          ),
        ],
      ),
    );
  }
}
