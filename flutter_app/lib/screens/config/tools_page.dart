import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class ToolsPage extends StatelessWidget {
  const ToolsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final tools = cfg.cfg['tools'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Warning banner
            Container(
              padding: const EdgeInsets.all(12),
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: CognithorTheme.orange.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: CognithorTheme.orange.withValues(alpha: 0.3),
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.warning_amber,
                    color: CognithorTheme.orange,
                    size: 20,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      l.toolsWarning,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: CognithorTheme.orange,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            // Section header
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                l.toolsSectionDesktop,
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  color: CognithorTheme.accent,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            CognithorToggleField(
              label: l.toolsComputerUseLabel,
              description: l.toolsComputerUseDesc,
              value: tools['computer_use_enabled'] == true,
              onChanged: (v) => cfg.set('tools.computer_use_enabled', v),
            ),
            CognithorToggleField(
              label: l.toolsDesktopLabel,
              description: l.toolsDesktopDesc,
              value: tools['desktop_tools_enabled'] == true,
              onChanged: (v) => cfg.set('tools.desktop_tools_enabled', v),
            ),
          ],
        );
      },
    );
  }
}
