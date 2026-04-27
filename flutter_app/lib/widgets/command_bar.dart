import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/navigation_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/glass_panel.dart';

class CommandBar extends StatelessWidget {
  const CommandBar({super.key, this.onSearchTap});

  final VoidCallback? onSearchTap;

  @override
  Widget build(BuildContext context) {
    final nav = context.watch<NavigationProvider>();

    return SizedBox(
      height: 40,
      child: GlassPanel(
        tint: nav.sectionColor,
        borderRadius: 0,
        blur: 10,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Row(
          children: [
            // Left: Section icon + name
            Icon(Icons.circle, size: 8, color: nav.sectionColor),
            const SizedBox(width: 8),
            Text(
              nav.sectionName,
              style: TextStyle(
                color: nav.sectionColor,
                fontSize: 12,
                fontWeight: FontWeight.w600,
                letterSpacing: 1,
              ),
            ),
            const Spacer(),
            // Center: Search hint — hidden on mobile (search is in drawer)
            if (MediaQuery.of(context).size.width > 500)
              GestureDetector(
                onTap: onSearchTap,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.07),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.14),
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.search,
                        size: 14,
                        color: CognithorTheme.textSecondary,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        AppLocalizations.of(context).globalSearch,
                        style: TextStyle(
                          color: CognithorTheme.textSecondary,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            if (MediaQuery.of(context).size.width > 500) const Spacer(),
            // Right: Status + model
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: CognithorTheme.green,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: CognithorTheme.green.withValues(alpha: 0.6),
                    blurRadius: 10,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Text(
              AppLocalizations.of(context).running,
              style: TextStyle(
                color: CognithorTheme.textSecondary,
                fontSize: 11,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
