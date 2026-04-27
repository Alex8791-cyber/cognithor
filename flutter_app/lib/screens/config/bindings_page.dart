import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class BindingsConfigPage extends StatelessWidget {
  const BindingsConfigPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Row(
              children: [
                Text(
                  AppLocalizations.of(context).bindingsTitle,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontSize: 16),
                ),
                const Spacer(),
                IconButton(
                  icon: Icon(Icons.add, color: CognithorTheme.accent),
                  onPressed: () => cfg.addBinding({
                    'name': 'new-binding',
                    'channel': '',
                    'pattern': '',
                    'target': '',
                    'enabled': true,
                  }),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ...List.generate(cfg.bindings.length, (i) {
              final b = cfg.bindings[i];
              return CognithorCollapsibleCard(
                title: (b['name'] ?? 'Binding $i').toString(),
                icon: Icons.link,
                children: [
                  CognithorTextField(
                    label: 'Name',
                    value: (b['name'] ?? '').toString(),
                    onChanged: (v) => cfg.updateBinding(i, {...b, 'name': v}),
                  ),
                  CognithorTextField(
                    label: 'Channel Filter',
                    value: (b['channel'] ?? '').toString(),
                    onChanged: (v) =>
                        cfg.updateBinding(i, {...b, 'channel': v}),
                    placeholder: 'e.g. telegram, slack, *',
                  ),
                  CognithorTextField(
                    label: 'Pattern (regex)',
                    value: (b['pattern'] ?? '').toString(),
                    onChanged: (v) =>
                        cfg.updateBinding(i, {...b, 'pattern': v}),
                    mono: true,
                  ),
                  CognithorTextField(
                    label: 'Target Agent',
                    value: (b['target'] ?? '').toString(),
                    onChanged: (v) => cfg.updateBinding(i, {...b, 'target': v}),
                  ),
                  CognithorToggleField(
                    label: 'Enabled',
                    value: b['enabled'] == true,
                    onChanged: (v) =>
                        cfg.updateBinding(i, {...b, 'enabled': v}),
                  ),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed: () => cfg.removeBinding(i),
                      icon: Icon(
                        Icons.delete,
                        size: 16,
                        color: CognithorTheme.red,
                      ),
                      label: Text(
                        AppLocalizations.of(context).remove,
                        style: TextStyle(color: CognithorTheme.red),
                      ),
                    ),
                  ),
                ],
              );
            }),
          ],
        );
      },
    );
  }
}
