import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorCodeBlock extends StatelessWidget {
  const CognithorCodeBlock({
    super.key,
    required this.code,
    this.language,
    this.maxLines,
  });

  final String code;
  final String? language;
  final int? maxLines;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: CognithorTheme.codeBlockBg,
        borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Theme.of(context).cardColor,
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(CognithorTheme.cardRadius),
              ),
            ),
            child: Row(
              children: [
                if (language != null)
                  Text(
                    language!,
                    style: TextStyle(
                      color: CognithorTheme.textSecondary,
                      fontSize: 12,
                    ),
                  ),
                const Spacer(),
                IconButton(
                  icon: Icon(
                    Icons.copy,
                    size: CognithorTheme.iconSizeSm,
                    color: CognithorTheme.textSecondary,
                  ),
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: code));
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(AppLocalizations.of(context).copied),
                        duration: const Duration(seconds: 2),
                      ),
                    );
                  },
                  constraints: const BoxConstraints(
                    minWidth: 32,
                    minHeight: 32,
                  ),
                  padding: EdgeInsets.zero,
                  tooltip: AppLocalizations.of(context).copy,
                ),
              ],
            ),
          ),
          // Code body
          Padding(
            padding: const EdgeInsets.all(12),
            child: SelectableText(
              code,
              maxLines: maxLines,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 13,
                color: CognithorTheme.textPrimary,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
