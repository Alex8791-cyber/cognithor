import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late TextEditingController _urlController;

  @override
  void initState() {
    super.initState();
    final conn = context.read<ConnectionProvider>();
    _urlController = TextEditingController(text: conn.serverUrl);
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final conn = context.watch<ConnectionProvider>();
    final l = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(title: Text(l.settings)),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // Server URL
          Text(l.serverUrl, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 8),
          TextField(
            controller: _urlController,
            decoration: InputDecoration(hintText: l.serverUrlHint),
            onSubmitted: (v) => conn.setServerUrl(v),
          ),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () async {
              await conn.setServerUrl(_urlController.text);
              if (context.mounted &&
                  conn.state == CognithorConnectionState.connected) {
                Navigator.of(context).pop();
              }
            },
            child: Text(l.save),
          ),
          const SizedBox(height: 32),

          // Version info
          if (conn.backendVersion != null)
            Text(
              l.version(conn.backendVersion!),
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: CognithorTheme.accent),
            ),
        ],
      ),
    );
  }
}
