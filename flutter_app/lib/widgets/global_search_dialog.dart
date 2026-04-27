import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// Indexed terms for all config pages.
/// Each entry: (display label, page key, search terms).
const _fieldIndex = <(String, String, List<String>)>[
  ('General', 'general', ['owner', 'mode', 'version', 'cost', 'budget']),
  ('Language', 'language', ['language', 'locale', 'translation', 'i18n']),
  (
    'Providers',
    'providers',
    [
      'provider',
      'api key',
      'ollama',
      'openai',
      'anthropic',
      'gemini',
      'groq',
      'deepseek',
      'mistral',
      'openrouter',
      'backend',
    ],
  ),
  (
    'Planner',
    'planner',
    [
      'planner',
      'gatekeeper',
      'sandbox',
      'pge',
      'iterations',
      'escalation',
      'temperature',
    ],
  ),
  (
    'Executor',
    'executor',
    ['executor', 'timeout', 'retry', 'parallel', 'backoff'],
  ),
  (
    'Prompts',
    'prompts',
    ['prompt', 'system', 'replan', 'escalation', 'policy', 'personality'],
  ),
  (
    'Memory',
    'memory',
    [
      'memory',
      'chunk',
      'weight',
      'vector',
      'bm25',
      'graph',
      'recency',
      'compaction',
    ],
  ),
  (
    'Bindings',
    'bindings',
    ['binding', 'filter', 'pattern', 'target', 'routing'],
  ),
  (
    'Web',
    'web',
    [
      'web',
      'search',
      'domain',
      'fetch',
      'duckduckgo',
      'brave',
      'google',
      'jina',
    ],
  ),
  (
    'Vault',
    'vault',
    ['vault', 'encrypt', 'obsidian', 'knowledge', 'auto-save'],
  ),
  (
    'Channels',
    'channels',
    [
      'channel',
      'telegram',
      'slack',
      'discord',
      'whatsapp',
      'signal',
      'matrix',
      'teams',
      'voice',
      'irc',
      'twitch',
    ],
  ),
  (
    'Security',
    'security',
    ['security', 'path', 'blocked', 'command', 'credential', 'pattern'],
  ),
  ('Tools', 'tools', ['tool', 'computer use', 'desktop', 'allowed', 'blocked']),
  ('Audit', 'audit', ['audit', 'hash', 'chain', 'integrity', 'compliance']),
  (
    'Database',
    'database',
    ['database', 'sqlite', 'postgresql', 'postgres', 'encryption', 'pool'],
  ),
  ('Logging', 'logging', ['log', 'level', 'json', 'console', 'debug']),
  ('Cron', 'cron', ['cron', 'heartbeat', 'schedule', 'plugin', 'job']),
  ('MCP', 'mcp', ['mcp', 'server', 'a2a', 'protocol']),
  (
    'System Profile',
    'system_profile',
    ['system', 'profile', 'hardware', 'gpu', 'vram'],
  ),
  ('Budget', 'budget', ['budget', 'token', 'cost', 'limit']),
  (
    'Social Listening',
    'social',
    ['reddit', 'lead', 'social', 'scan', 'subreddit', 'intent', 'score'],
  ),
  (
    'System',
    'system',
    ['system', 'restart', 'export', 'import', 'reset', 'factory'],
  ),
];

class GlobalSearchDialog extends StatefulWidget {
  const GlobalSearchDialog({super.key, required this.onNavigate});

  /// Called with the page key (e.g. 'providers', 'planner') when a result is tapped.
  final void Function(String pageKey) onNavigate;

  @override
  State<GlobalSearchDialog> createState() => _GlobalSearchDialogState();
}

class _GlobalSearchDialogState extends State<GlobalSearchDialog> {
  final _ctrl = TextEditingController();
  List<(String, String, List<String>)> _results = [];

  void _search(String query) {
    if (query.isEmpty) {
      setState(() => _results = []);
      return;
    }
    final q = query.toLowerCase();
    setState(() {
      _results = _fieldIndex
          .where(
            (entry) =>
                entry.$1.toLowerCase().contains(q) ||
                entry.$3.any((term) => term.contains(q)),
          )
          .take(8)
          .toList();
    });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Dialog(
      backgroundColor: theme.cardColor,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 500, maxHeight: 400),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.all(16),
              child: TextField(
                controller: _ctrl,
                autofocus: true,
                decoration: const InputDecoration(
                  hintText: 'Search config pages...',
                  prefixIcon: Icon(Icons.search),
                  isDense: true,
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 10,
                  ),
                ),
                onChanged: _search,
              ),
            ),
            if (_results.isNotEmpty)
              Flexible(
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: _results.length,
                  itemBuilder: (context, i) {
                    final r = _results[i];
                    return ListTile(
                      dense: true,
                      title: Text(r.$1),
                      subtitle: Text(
                        r.$3.take(4).join(', '),
                        style: theme.textTheme.bodySmall,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      onTap: () {
                        Navigator.of(context).pop();
                        widget.onNavigate(r.$2);
                      },
                    );
                  },
                ),
              ),
            if (_results.isEmpty && _ctrl.text.isNotEmpty)
              Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  AppLocalizations.of(context).noMatchingPages,
                  style: theme.textTheme.bodySmall,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
