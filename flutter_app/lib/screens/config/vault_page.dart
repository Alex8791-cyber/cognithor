import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class VaultPage extends StatelessWidget {
  const VaultPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final l = AppLocalizations.of(context);
    final encryptOn =
        ((context.watch<ConfigProvider>().cfg['vault']
                as Map<String, dynamic>? ??
            {})['encrypt_files'] ==
        true);

    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final vault = cfg.cfg['vault'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorToggleField(
              label: l.vaultActive,
              value: vault['enabled'] == true,
              onChanged: (v) => cfg.set('vault.enabled', v),
              description: l.vaultActiveDesc,
            ),
            const SizedBox(height: 12),
            CognithorToggleField(
              label: l.vaultEncryption,
              value: vault['encrypt_files'] == true,
              onChanged: (v) => cfg.set('vault.encrypt_files', v),
              description: l.vaultEncryptionDesc,
            ),
            const SizedBox(height: 8),
            // Security info box — changes based on toggle state
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: encryptOn
                    ? Colors.green.withValues(alpha: 0.1)
                    : Colors.orange.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: encryptOn
                      ? Colors.green.withValues(alpha: 0.3)
                      : Colors.orange.withValues(alpha: 0.3),
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    encryptOn ? Icons.lock : Icons.lock_open,
                    size: 20,
                    color: encryptOn ? Colors.green : Colors.orange,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      encryptOn ? l.vaultEncryptOn : l.vaultEncryptOff,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: encryptOn
                            ? Colors.green[300]
                            : Colors.orange[300],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            // What's always encrypted info
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.withValues(alpha: 0.2)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.shield, size: 18, color: Colors.blue[300]),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          l.vaultAlwaysEncrypted,
                          style: theme.textTheme.bodySmall?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: Colors.blue[300],
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    l.vaultAlwaysEncryptedList,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: Colors.blue[200],
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            CognithorToggleField(
              label: l.vaultAutoSave,
              value: vault['auto_save_research'] == true,
              onChanged: (v) => cfg.set('vault.auto_save_research', v),
              description: l.vaultAutoSaveDesc,
            ),
          ],
        );
      },
    );
  }
}
