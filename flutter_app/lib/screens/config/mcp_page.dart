import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class McpPage extends StatelessWidget {
  const McpPage({super.key});

  static List<String> _toStringList(dynamic v) {
    if (v is List) return v.map((e) => e.toString()).toList();
    return [];
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        // Ensure 'servers' key exists as a List
        final rawServers = cfg.mcpServers['servers'];
        final servers = rawServers is List ? rawServers : <dynamic>[];
        if (rawServers == null) {
          cfg.mcpServers['servers'] = servers;
        }
        final a2a = cfg.a2a;

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Row(
              children: [
                Text(
                  AppLocalizations.of(context).mcpServers,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontSize: 16),
                ),
                const Spacer(),
                IconButton(
                  icon: Icon(Icons.add, color: CognithorTheme.accent),
                  onPressed: () {
                    final updated = List<dynamic>.from(servers)
                      ..add({
                        'name': 'new-server',
                        'command': '',
                        'args': <String>[],
                        'enabled': true,
                      });
                    cfg.mcpServers['servers'] = updated;
                    cfg.notify();
                  },
                ),
              ],
            ),
            const SizedBox(height: 8),
            ...List.generate(servers.length, (i) {
              final raw = servers[i];
              final s = raw is Map<String, dynamic> ? raw : <String, dynamic>{};
              return CognithorCollapsibleCard(
                title: (s['name'] ?? 'Server $i').toString(),
                icon: Icons.dns,
                badge: s['enabled'] == true ? 'ON' : 'OFF',
                children: [
                  CognithorTextField(
                    label: 'Name',
                    value: (s['name'] ?? '').toString(),
                    onChanged: (v) {
                      s['name'] = v;
                      cfg.notify();
                    },
                  ),
                  CognithorTextField(
                    label: 'Command',
                    value: (s['command'] ?? '').toString(),
                    onChanged: (v) {
                      s['command'] = v;
                      cfg.notify();
                    },
                  ),
                  CognithorListField(
                    label: 'Arguments',
                    value: _toStringList(s['args']),
                    onChanged: (v) {
                      s['args'] = v;
                      cfg.notify();
                    },
                    placeholder: '--flag value',
                  ),
                  CognithorToggleField(
                    label: 'Enabled',
                    value: s['enabled'] == true,
                    onChanged: (v) {
                      s['enabled'] = v;
                      cfg.notify();
                    },
                  ),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed: () {
                        servers.removeAt(i);
                        cfg.notify();
                      },
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
            const Divider(height: 32),
            CognithorCollapsibleCard(
              title: 'A2A Protocol',
              icon: Icons.swap_horiz,
              children: [
                CognithorToggleField(
                  label: 'Enabled',
                  value: a2a['enabled'] == true,
                  onChanged: (v) {
                    cfg.a2a['enabled'] = v;
                    cfg.notify();
                  },
                ),
                CognithorJsonEditor(
                  label: 'Remotes',
                  value: a2a['remotes'] ?? [],
                  onChanged: (v) {
                    cfg.a2a['remotes'] = v;
                    cfg.notify();
                  },
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}
