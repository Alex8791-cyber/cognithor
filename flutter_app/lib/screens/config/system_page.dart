import 'dart:convert';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/cognithor_toast.dart';

class SystemConfigPage extends StatelessWidget {
  const SystemConfigPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final l = AppLocalizations.of(context);

    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Stop (manual restart required)
            _ActionCard(
              icon: Icons.stop_circle_outlined,
              title: l.restartBackend,
              description: l.stopBackendDescription,
              buttonLabel: l.stopLabel,
              onPressed: () async {
                final api = context.read<ConnectionProvider>().api;
                final confirmed = await showDialog<bool>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    title: Text(l.stopBackend),
                    content: Text(l.stopBackendConfirmBody),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.of(ctx).pop(false),
                        child: Text(l.cancel),
                      ),
                      ElevatedButton(
                        onPressed: () => Navigator.of(ctx).pop(true),
                        child: Text(l.stopLabel),
                      ),
                    ],
                  ),
                );
                if (confirmed == true) {
                  await api.shutdownServer();
                  if (context.mounted) {
                    CognithorToast.show(
                      context,
                      l.backendStopped,
                      type: ToastType.warning,
                    );
                  }
                }
              },
            ),
            const SizedBox(height: 12),
            // Export
            _ActionCard(
              icon: Icons.download,
              title: l.exportConfig,
              description: l.downloadConfigDesc,
              buttonLabel: l.exportLabel,
              onPressed: () async {
                final json = cfg.exportJson();
                final date = DateTime.now().toIso8601String().split('T').first;
                final dataUri =
                    'data:application/json;charset=utf-8,${Uri.encodeComponent(json)}';
                final launched = await launchUrl(
                  Uri.parse(dataUri),
                  webOnlyWindowName: 'cognithor-config-$date.json',
                );
                if (!launched) {
                  // Fallback: copy to clipboard
                  await Clipboard.setData(ClipboardData(text: json));
                }
                if (context.mounted) {
                  CognithorToast.show(
                    context,
                    launched ? l.exportConfig : l.copiedToClipboard,
                    type: ToastType.success,
                  );
                }
              },
            ),
            const SizedBox(height: 12),
            // Import
            _ActionCard(
              icon: Icons.upload,
              title: l.importConfig,
              description: l.loadConfigDesc,
              buttonLabel: l.importLabel,
              onPressed: () async {
                final result = await FilePicker.platform.pickFiles(
                  type: FileType.custom,
                  allowedExtensions: ['json'],
                  withData: true,
                );
                if (result != null && result.files.single.bytes != null) {
                  final content = utf8.decode(result.files.single.bytes!);
                  await cfg.importJson(content);
                  if (context.mounted) {
                    CognithorToast.show(
                      context,
                      l.configImported,
                      type: ToastType.success,
                    );
                  }
                }
              },
            ),
            const SizedBox(height: 12),
            // Factory Reset (not yet implemented)
            _ActionCard(
              icon: Icons.warning_amber,
              title: l.factoryReset,
              description: l.resetAllDesc,
              buttonLabel: l.resetLabel,
              isDanger: true,
              onPressed: () async {
                await showDialog<void>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    title: Text(l.factoryReset),
                    content: Text(l.factoryResetNotImpl),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.of(ctx).pop(),
                        child: Text(l.ok),
                      ),
                    ],
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            // Runtime info
            Text(
              l.runtimeInfo,
              style: theme.textTheme.titleLarge?.copyWith(fontSize: 16),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.cardColor,
                borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
                border: Border.all(color: theme.dividerColor),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _infoRow(
                    theme,
                    'Version',
                    (cfg.cfg['version'] ?? '-').toString(),
                  ),
                  _infoRow(
                    theme,
                    'Owner',
                    (cfg.cfg['owner_name'] ?? '-').toString(),
                  ),
                  _infoRow(
                    theme,
                    'Mode',
                    (cfg.cfg['operation_mode'] ?? '-').toString(),
                  ),
                  _infoRow(
                    theme,
                    'Backend',
                    (cfg.cfg['llm_backend_type'] ?? '-').toString(),
                  ),
                  _infoRow(
                    theme,
                    'Language',
                    (cfg.cfg['language'] ?? '-').toString(),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _infoRow(ThemeData theme, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 100,
            child: Text(label, style: theme.textTheme.bodySmall),
          ),
          Expanded(child: Text(value, style: theme.textTheme.bodyMedium)),
        ],
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.buttonLabel,
    required this.onPressed,
    this.isDanger = false,
  });

  final IconData icon;
  final String title;
  final String description;
  final String buttonLabel;
  final VoidCallback onPressed;
  final bool isDanger;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = isDanger ? CognithorTheme.red : CognithorTheme.accent;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
        border: Border.all(
          color: isDanger
              ? CognithorTheme.red.withValues(alpha: 0.3)
              : Theme.of(context).dividerColor,
        ),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(description, style: theme.textTheme.bodySmall),
              ],
            ),
          ),
          ElevatedButton(
            onPressed: onPressed,
            style: isDanger
                ? ElevatedButton.styleFrom(
                    backgroundColor: CognithorTheme.red,
                    foregroundColor: Colors.white,
                  )
                : null,
            child: Text(buttonLabel),
          ),
        ],
      ),
    );
  }
}
