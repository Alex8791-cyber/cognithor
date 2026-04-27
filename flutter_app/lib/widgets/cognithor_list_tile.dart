import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorListTile extends StatelessWidget {
  const CognithorListTile({
    super.key,
    required this.title,
    this.subtitle,
    this.leading,
    this.trailing,
    this.onTap,
    this.dense = false,
  });

  final String title;
  final String? subtitle;
  final Widget? leading;
  final Widget? trailing;
  final VoidCallback? onTap;
  final bool dense;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final verticalPad = dense ? CognithorTheme.spacingSm : 12.0;

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
      child: Padding(
        padding: EdgeInsets.symmetric(
          horizontal: CognithorTheme.spacing,
          vertical: verticalPad,
        ),
        child: Row(
          children: [
            if (leading != null) ...[
              leading!,
              const SizedBox(width: CognithorTheme.spacingSm),
            ],
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(title, style: theme.textTheme.bodyLarge),
                  if (subtitle != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(subtitle!, style: theme.textTheme.bodySmall),
                    ),
                ],
              ),
            ),
            if (trailing != null) ...[
              const SizedBox(width: CognithorTheme.spacingSm),
              trailing!,
            ],
          ],
        ),
      ),
    );
  }
}
