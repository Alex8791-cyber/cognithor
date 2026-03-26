import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

/// Claude-style version navigator: "< 1/2 >" arrows for edited messages.
class VersionNavigator extends StatelessWidget {
  const VersionNavigator({
    super.key,
    required this.currentVersion,
    required this.totalVersions,
    required this.onPrevious,
    required this.onNext,
  });

  final int currentVersion;
  final int totalVersions;
  final VoidCallback onPrevious;
  final VoidCallback onNext;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _NavButton(
          icon: Icons.chevron_left,
          onTap: currentVersion > 0 ? onPrevious : null,
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            '${currentVersion + 1} / $totalVersions',
            style: TextStyle(
              fontSize: 11,
              color: JarvisTheme.textSecondary,
              fontFamily: 'monospace',
            ),
          ),
        ),
        _NavButton(
          icon: Icons.chevron_right,
          onTap: currentVersion < totalVersions - 1 ? onNext : null,
        ),
      ],
    );
  }
}

class _NavButton extends StatelessWidget {
  const _NavButton({required this.icon, required this.onTap});

  final IconData icon;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return InkWell(
      borderRadius: BorderRadius.circular(10),
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.all(2),
        child: Icon(
          icon,
          size: 16,
          color: enabled ? JarvisTheme.textSecondary : JarvisTheme.textTertiary,
        ),
      ),
    );
  }
}
