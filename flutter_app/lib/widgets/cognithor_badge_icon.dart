import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorBadgeIcon extends StatelessWidget {
  const CognithorBadgeIcon({
    super.key,
    required this.icon,
    this.count = 0,
    this.color,
  });

  final IconData icon;
  final int count;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        Icon(
          icon,
          color: color ?? CognithorTheme.textSecondary,
          size: CognithorTheme.iconSizeMd,
        ),
        if (count > 0)
          Positioned(
            right: -6,
            top: -4,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
              constraints: const BoxConstraints(minWidth: 16, minHeight: 16),
              decoration: BoxDecoration(
                color: CognithorTheme.red,
                borderRadius: BorderRadius.circular(CognithorTheme.chipRadius),
              ),
              child: Center(
                child: Text(
                  count > 99 ? '99+' : count.toString(),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}
