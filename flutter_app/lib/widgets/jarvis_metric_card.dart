import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';

class JarvisMetricCard extends StatelessWidget {
  const JarvisMetricCard({
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
    final effectiveColor = color ?? JarvisTheme.accent;

    return GlassPanel(
      tint: effectiveColor,
      glowOnHover: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              if (icon != null) ...[
                Icon(icon, size: JarvisTheme.iconSizeMd, color: effectiveColor),
                const SizedBox(width: JarvisTheme.spacingSm),
              ],
              Expanded(
                child: Text(
                  title,
                  style: theme.textTheme.bodySmall,
                ),
              ),
              if (trend != null) _buildTrend(),
            ],
          ),
          const SizedBox(height: JarvisTheme.spacingSm),
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
    final trendColor = isPositive ? JarvisTheme.green : JarvisTheme.red;
    final arrow = isPositive ? Icons.arrow_upward : Icons.arrow_downward;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(arrow, size: JarvisTheme.iconSizeSm, color: trendColor),
        const SizedBox(width: 2),
        Text(
          '${trend!.abs().toStringAsFixed(1)}%',
          style: TextStyle(color: trendColor, fontSize: 12, fontWeight: FontWeight.w600),
        ),
      ],
    );
  }
}
