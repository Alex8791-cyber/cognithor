import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/config_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/form/form_widgets.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';

class ProvidersPage extends StatelessWidget {
  const ProvidersPage({super.key});

  static const _providers = [
    ('ollama', 'Ollama', Icons.computer),
    ('openai', 'OpenAI', Icons.auto_awesome),
    ('anthropic', 'Anthropic', Icons.psychology),
    ('claude-code', 'Claude Subscription', Icons.psychology),
    ('gemini', 'Google Gemini', Icons.diamond),
    ('groq', 'Groq', Icons.speed),
    ('deepseek', 'DeepSeek', Icons.search),
    ('mistral', 'Mistral', Icons.air),
    ('together', 'Together AI', Icons.group),
    ('openrouter', 'OpenRouter', Icons.router),
    ('xai', 'xAI', Icons.smart_toy),
    ('cerebras', 'Cerebras', Icons.memory),
    ('github', 'GitHub Models', Icons.code),
    ('bedrock', 'AWS Bedrock', Icons.cloud),
    ('huggingface', 'Hugging Face', Icons.face),
    ('moonshot', 'Moonshot', Icons.nightlight),
    ('lmstudio', 'LM Studio', Icons.laptop),
  ];

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final currentBackend =
            (cfg.cfg['llm_backend_type'] ?? 'ollama').toString();

        // Sort: active provider first, rest in original order.
        final sorted = List<(String, String, IconData)>.from(_providers)
          ..sort((a, b) {
            if (a.$1 == currentBackend && b.$1 != currentBackend) return -1;
            if (b.$1 == currentBackend && a.$1 != currentBackend) return 1;
            return 0;
          });

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // --- Current Backend status card ---
            _CurrentBackendCard(
              currentBackend: currentBackend,
              onChangeBackend: () => _showBackendDialog(context, cfg),
            ),
            const SizedBox(height: 16),
            // --- Prominent backend selector ---
            _BackendSelector(
              currentBackend: currentBackend,
              onChanged: (v) => cfg.set('llm_backend_type', v),
            ),
            const SizedBox(height: 16),
            // --- Provider cards (active first) ---
            ...sorted.map(
              (p) => _ProviderCard(
                cfg: cfg,
                provider: p,
                isActive: p.$1 == currentBackend,
              ),
            ),
          ],
        );
      },
    );
  }

  void _showBackendDialog(BuildContext context, ConfigProvider cfg) {
    final conn = context.read<ConnectionProvider>();
    showDialog<String>(
      context: context,
      builder: (ctx) => _BackendSwitchDialog(
        currentBackend: (cfg.cfg['llm_backend_type'] ?? 'ollama').toString(),
      ),
    ).then((selected) {
      if (selected != null && selected.isNotEmpty) {
        cfg.set('llm_backend_type', selected);
        // Also switch on backend
        try {
          conn.api.switchBackend(selected);
        } catch (_) {}
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Current Backend status card (top of page)
// ---------------------------------------------------------------------------
class _CurrentBackendCard extends StatefulWidget {
  const _CurrentBackendCard({
    required this.currentBackend,
    required this.onChangeBackend,
  });

  final String currentBackend;
  final VoidCallback onChangeBackend;

  @override
  State<_CurrentBackendCard> createState() => _CurrentBackendCardState();
}

class _CurrentBackendCardState extends State<_CurrentBackendCard> {
  Map<String, dynamic>? _status;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadStatus();
  }

  Future<void> _loadStatus() async {
    try {
      final conn = context.read<ConnectionProvider>();
      final result = await conn.api.getBackendStatus();
      if (mounted) setState(() { _status = result; _loading = false; });
    } catch (_) {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final backends = _status?['backends'] as Map<String, dynamic>? ?? {};
    final info = backends[widget.currentBackend] as Map<String, dynamic>? ?? {};
    final isConnected = info['authenticated'] == true;

    String label = widget.currentBackend;
    IconData icon = Icons.hub;
    for (final p in ProvidersPage._providers) {
      if (p.$1 == widget.currentBackend) {
        label = p.$2;
        icon = p.$3;
        break;
      }
    }

    return NeonCard(
      tint: isConnected ? JarvisTheme.green : JarvisTheme.accent,
      glowOnHover: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon,
                  color: isConnected ? JarvisTheme.green : JarvisTheme.accent,
                  size: 28),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(label,
                        style: theme.textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 2),
                    Row(
                      children: [
                        Icon(
                          isConnected ? Icons.check_circle : Icons.cancel,
                          size: 14,
                          color: isConnected
                              ? JarvisTheme.green
                              : JarvisTheme.red,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _loading
                              ? '...'
                              : isConnected
                                  ? l.connected
                                  : l.notInstalled,
                          style: theme.textTheme.labelSmall?.copyWith(
                            color: isConnected
                                ? JarvisTheme.green
                                : JarvisTheme.red,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              OutlinedButton.icon(
                onPressed: widget.onChangeBackend,
                icon: const Icon(Icons.swap_horiz, size: 18),
                label: Text(l.switchBackend),
                style: OutlinedButton.styleFrom(
                  foregroundColor: JarvisTheme.accent,
                  side: BorderSide(
                      color: JarvisTheme.accent.withValues(alpha: 0.4)),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Backend switch dialog (reusable from config page)
// ---------------------------------------------------------------------------
class _BackendSwitchDialog extends StatefulWidget {
  const _BackendSwitchDialog({required this.currentBackend});
  final String currentBackend;

  @override
  State<_BackendSwitchDialog> createState() => _BackendSwitchDialogState();
}

class _BackendSwitchDialogState extends State<_BackendSwitchDialog> {
  Map<String, dynamic>? _status;
  bool _loading = true;
  String? _selected;

  @override
  void initState() {
    super.initState();
    _selected = widget.currentBackend;
    _loadStatus();
  }

  Future<void> _loadStatus() async {
    try {
      final conn = context.read<ConnectionProvider>();
      final result = await conn.api.getBackendStatus();
      if (mounted) setState(() { _status = result; _loading = false; });
    } catch (_) {
      if (mounted) setState(() => _loading = false);
    }
  }

  bool _isAuth(String key) {
    final backends = _status?['backends'] as Map<String, dynamic>? ?? {};
    final info = backends[key] as Map<String, dynamic>? ?? {};
    return info['authenticated'] == true;
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    final options = <(String, String, IconData, Color)>[
      ('claude-code', l.claudeSubscription, Icons.psychology, JarvisTheme.sectionChat),
      ('ollama', l.ollamaLocal, Icons.computer, JarvisTheme.matrix),
      ('openai', l.openaiApi, Icons.auto_awesome, JarvisTheme.sectionChat),
      ('anthropic', l.anthropicApi, Icons.key, const Color(0xFFAB68FF)),
    ];

    return AlertDialog(
      title: Text(l.chooseBackend),
      content: SizedBox(
        width: 400,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : Column(
                mainAxisSize: MainAxisSize.min,
                children: options.map((o) {
                  final (key, label, icon, tint) = o;
                  final auth = _isAuth(key);
                  final isSel = _selected == key;
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: NeonCard(
                      tint: isSel ? tint : null,
                      glowOnHover: true,
                      onTap: () => setState(() => _selected = key),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 10),
                      child: Row(
                        children: [
                          Icon(icon,
                              color: isSel ? tint : JarvisTheme.textSecondary,
                              size: 22),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(label,
                                style: TextStyle(
                                    fontWeight:
                                        isSel ? FontWeight.w700 : null)),
                          ),
                          Icon(
                            auth ? Icons.check_circle : Icons.cancel,
                            size: 16,
                            color: auth
                                ? JarvisTheme.green
                                : JarvisTheme.red,
                          ),
                          if (isSel)
                            Padding(
                              padding: const EdgeInsets.only(left: 8),
                              child: Icon(Icons.check_circle,
                                  color: tint, size: 20),
                            ),
                        ],
                      ),
                    ),
                  );
                }).toList(),
              ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(null),
          child: Text(l.cancel),
        ),
        ElevatedButton(
          onPressed: _selected != null
              ? () => Navigator.of(context).pop(_selected)
              : null,
          child: Text(l.confirm),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Prominent backend selector widget
// ---------------------------------------------------------------------------
class _BackendSelector extends StatelessWidget {
  const _BackendSelector({
    required this.currentBackend,
    required this.onChanged,
  });

  final String currentBackend;
  final ValueChanged<String> onChanged;

  String _labelFor(String key) {
    for (final p in ProvidersPage._providers) {
      if (p.$1 == key) return p.$2;
    }
    return key;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: JarvisTheme.accent.withValues(alpha: 0.07),
        borderRadius: BorderRadius.circular(JarvisTheme.cardRadius),
        border: Border.all(color: JarvisTheme.accent.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.hub, color: JarvisTheme.accent, size: 22),
              const SizedBox(width: 8),
              Text(
                'LLM Backend',
                style: theme.textTheme.titleLarge?.copyWith(fontSize: 18),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Choose which LLM provider Jarvis uses for all AI requests. '
            'The active provider\'s card is expanded below so you can '
            'configure its connection settings.',
            style: theme.textTheme.bodySmall
                ?.copyWith(color: JarvisTheme.textSecondary, height: 1.4),
          ),
          const SizedBox(height: 12),
          JarvisSelectField.fromStrings(
            label: 'Active Provider',
            value: currentBackend,
            options: ProvidersPage._providers.map((p) => p.$1).toList(),
            onChanged: onChanged,
            description: 'Currently using: ${_labelFor(currentBackend)}',
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Individual provider card
// ---------------------------------------------------------------------------
class _ProviderCard extends StatelessWidget {
  const _ProviderCard({
    required this.cfg,
    required this.provider,
    required this.isActive,
  });

  final ConfigProvider cfg;
  final (String, String, IconData) provider;
  final bool isActive;

  @override
  Widget build(BuildContext context) {
    final (key, label, icon) = provider;

    final children = _fieldsFor(key);

    return Opacity(
      opacity: isActive ? 1.0 : 0.55,
      child: JarvisCollapsibleCard(
        title: label,
        icon: icon,
        badge: isActive ? 'ACTIVE PROVIDER' : null,
        initiallyExpanded: isActive,
        forceOpen: isActive,
        children: children,
      ),
    );
  }

  List<Widget> _fieldsFor(String key) {
    if (key == 'ollama') {
      final ollama = cfg.cfg['ollama'] as Map<String, dynamic>? ?? {};
      return [
        JarvisTextField(
          label: 'Base URL',
          value: (ollama['base_url'] ?? 'http://localhost:11434').toString(),
          onChanged: (v) => cfg.set('ollama.base_url', v),
        ),
        JarvisNumberField(
          label: 'Timeout (seconds)',
          value: (ollama['timeout_seconds'] as num?) ?? 120,
          onChanged: (v) => cfg.set('ollama.timeout_seconds', v),
          min: 10,
        ),
        JarvisTextField(
          label: 'Keep Alive',
          value: (ollama['keep_alive'] ?? '5m').toString(),
          onChanged: (v) => cfg.set('ollama.keep_alive', v),
        ),
      ];
    }

    if (key == 'claude-code') {
      return [
        const Padding(
          padding: EdgeInsets.all(8),
          child: Text(
            'Claude Code uses your existing Claude Pro/Max subscription. '
            'No API key needed -- just ensure Claude Code CLI is installed.\n\n'
            'Install: npm install -g @anthropic-ai/claude-code',
          ),
        ),
      ];
    }

    final apiKey = '${key}_api_key';
    final baseUrl = '${key}_base_url';

    return [
      JarvisTextField(
        label: 'API Key',
        value: (cfg.cfg[apiKey] ?? '').toString(),
        onChanged: (v) => cfg.set(apiKey, v),
        isPassword: true,
        isSecret: true,
      ),
      if (key == 'openai' || key == 'lmstudio')
        JarvisTextField(
          label: 'Base URL (optional)',
          value: (cfg.cfg[baseUrl] ?? '').toString(),
          onChanged: (v) => cfg.set(baseUrl, v),
          placeholder: key == 'lmstudio'
              ? 'http://localhost:1234/v1'
              : 'https://api.openai.com/v1',
        ),
      if (key == 'anthropic')
        JarvisNumberField(
          label: 'Max Tokens',
          value: (cfg.cfg['anthropic_max_tokens'] as num?) ?? 4096,
          onChanged: (v) => cfg.set('anthropic_max_tokens', v),
          min: 256,
        ),
    ];
  }
}
