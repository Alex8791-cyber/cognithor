import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/providers/config_provider.dart';
import 'package:jarvis_ui/widgets/form/form_widgets.dart';

class VaultPage extends StatelessWidget {
  const VaultPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final encryptOn =
        ((context.watch<ConfigProvider>().cfg['vault'] as Map<String, dynamic>? ?? {})['encrypt_files'] == true);

    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final vault = cfg.cfg['vault'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            JarvisToggleField(
              label: 'Vault aktiv',
              value: vault['enabled'] == true,
              onChanged: (v) => cfg.set('vault.enabled', v),
              description: 'Knowledge Vault aktivieren/deaktivieren',
            ),
            const SizedBox(height: 12),
            JarvisToggleField(
              label: 'Dateiverschluesselung',
              value: vault['encrypt_files'] == true,
              onChanged: (v) => cfg.set('vault.encrypt_files', v),
              description: 'Vault .md Dateien mit AES-256 verschluesseln.',
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
                      encryptOn
                          ? 'Maximale Sicherheit: Vault-Dateien sind verschluesselt. '
                              'Datenbanken + Memory + Vault = alles geschuetzt. '
                              'Obsidian kann diese Dateien nicht lesen.'
                          : 'Obsidian-kompatibel: Vault-Dateien sind Klartext. '
                              'Datenbanken und Memory-Dateien sind trotzdem verschluesselt. '
                              'Fuer vollen Schutz der Vault-Dateien: '
                              'BitLocker (Windows) oder LUKS (Linux) aktivieren.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: encryptOn ? Colors.green[300] : Colors.orange[300],
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
                      Text(
                        'Immer verschluesselt (unabhaengig von diesem Toggle):',
                        style: theme.textTheme.bodySmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue[300],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    '  - 33 SQLite-Datenbanken (SQLCipher / AES-256)\n'
                    '  - CORE.md (Agent-Persoenlichkeit)\n'
                    '  - Episodische Erinnerungen (.md)\n'
                    '  - Gelernte Prozeduren (.md)\n'
                    '  - Lernplaene (.json)\n'
                    '  - Credentials (Fernet / PBKDF2)\n'
                    '  - Schluessel: OS Keyring (nicht auf der Festplatte)',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: Colors.blue[200],
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            JarvisToggleField(
              label: 'Auto-Save Recherchen',
              value: vault['auto_save_research'] == true,
              onChanged: (v) => cfg.set('vault.auto_save_research', v),
              description: 'Web-Recherchen automatisch im Vault speichern',
            ),
          ],
        );
      },
    );
  }
}
