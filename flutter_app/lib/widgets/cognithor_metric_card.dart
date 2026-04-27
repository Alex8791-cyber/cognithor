import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';

class CognithorMetricCard extends StatelessWidget {
  const CognithorMetricCard({
    super.key,
    required this.title,
    required this.value,
    this.subtitle,
    this.trend,
    this.icon,
    this.color,
  });

  final String title;
  final String value;
  final String? subtitle;
  final double? trend;
  final IconData? icon;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final effectiveColor = color ?? CognithorTheme.accent;

    return NeonCard(
      tint: effectiveColor,
      glowOnHover: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              if (icon != null) ...[
                Icon(
                  icon,
                  size: CognithorTheme.iconSizeMd,
                  color: effectiveColor,
                ),
                const SizedBox(width: CognithorTheme.spacingSm),
              ],
              Expanded(child: Text(title, style: theme.textTheme.bodySmall)),
              if (trend != null) _buildTrend(),
            ],
          ),
          const SizedBox(height: CognithorTheme.spacingSm),
          Text(
            value,
            style: theme.textTheme.titleLarge?.copyWith(
              color: effectiveColor,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (subtitle != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(subtitle!, style: theme.textTheme.bodySmall),
            ),
        ],
      ),
    );
  }

  Widget _buildTrend() {
    final isPositive = trend! >= 0;
    final trendColor = isPositive ? CognithorTheme.green : CognithorTheme.red;
    final arrow = isPositive ? Icons.arrow_upward : Icons.arrow_downward;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(arrow, size: CognithorTheme.iconSizeSm, color: trendColor),
        const SizedBox(width: 2),
        Text(
          '${trend!.abs().toStringAsFixed(1)}%',
          style: TextStyle(
            color: trendColor,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}
