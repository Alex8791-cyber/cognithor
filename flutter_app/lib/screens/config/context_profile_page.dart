import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/cognithor_toast.dart';

/// Sprint-23 — Context-Profile picker.
///
/// Backend: GET/POST /api/v1/context_profile (see
/// src/cognithor/channels/config_routes/profile.py).
class ContextProfilePage extends StatefulWidget {
  const ContextProfilePage({super.key});

  @override
  State<ContextProfilePage> createState() => _ContextProfilePageState();
}

class _ContextProfilePageState extends State<ContextProfilePage> {
  String? _active;
  Map<String, dynamic> _available = const {};
  bool _loading = true;
  bool _saving = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final data = await api.get('v1/context_profile');
      if (!mounted) return;
      setState(() {
        _active = data['active'] as String?;
        _available = (data['available'] as Map?)?.cast<String, dynamic>() ?? {};
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _setProfile(String? profile) async {
    setState(() => _saving = true);
    try {
      final api = context.read<ConnectionProvider>().api;
      final data = await api.post('v1/context_profile', {'profile': profile});
      if (!mounted) return;
      setState(() {
        _active = data['active'] as String?;
        _saving = false;
      });
      CognithorToast.show(
        context,
        profile == null
            ? 'Profil deaktiviert — Modell-Defaults aktiv'
            : 'Profil "$profile" aktiv',
        type: ToastType.success,
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _saving = false);
      CognithorToast.show(
        context,
        'Profil-Wechsel fehlgeschlagen: $e',
        type: ToastType.error,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline, size: 48, color: Colors.red.shade400),
            const SizedBox(height: 12),
            Text('Fehler: $_error', textAlign: TextAlign.center),
            const SizedBox(height: 12),
            FilledButton(onPressed: _load, child: const Text('Erneut laden')),
          ],
        ),
      );
    }

    final profileNames = _orderProfiles(_available.keys.toList());

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(
          'Kontext-Profil',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 8),
        const Text(
          'Wähle wie viel Kontext und wie viel Sampling-Spielraum die '
          'aktive Anfrage bekommt. Das Profil legt num_ctx, temperature '
          'und top_p für jeden LLM-Backend-Aufruf fest. Embedding-Modelle '
          'ignorieren das Profil.',
        ),
        const SizedBox(height: 16),
        _buildClearTile(),
        const SizedBox(height: 8),
        ...profileNames.map(_buildProfileTile),
        const SizedBox(height: 24),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Hinweise', style: Theme.of(context).textTheme.titleSmall),
                const SizedBox(height: 6),
                const Text(
                  '• arc_agi3 erfordert ein vLLM-Backend mit '
                  '--max-model-len ≥ 131072. Andere Backends (Anthropic, '
                  'Gemini) nutzen ihr modell-intrinsisches Fenster.',
                ),
                const SizedBox(height: 4),
                const Text(
                  '• Per-Request-Override im Code: '
                  'router.context_profile_scope("deep").',
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  List<String> _orderProfiles(List<String> raw) {
    const order = ['quick', 'default', 'deep', 'arc_agi3'];
    final ordered = <String>[];
    for (final name in order) {
      if (raw.contains(name)) ordered.add(name);
    }
    for (final name in raw) {
      if (!ordered.contains(name)) ordered.add(name);
    }
    return ordered;
  }

  Widget _buildClearTile() {
    final isActive = _active == null;
    return Card(
      color: isActive ? CognithorTheme.accent.withValues(alpha: 0.12) : null,
      child: ListTile(
        leading: Icon(
          isActive ? Icons.radio_button_checked : Icons.radio_button_unchecked,
          color: isActive ? CognithorTheme.accent : null,
        ),
        title: const Text('Kein Profil (Modell-Defaults)'),
        subtitle: const Text(
          'Jedes Modell nutzt sein in der Config hinterlegtes context_window, '
          'temperature, top_p.',
        ),
        trailing: _saving
            ? const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : null,
        onTap: _saving || isActive ? null : () => _setProfile(null),
      ),
    );
  }

  Widget _buildProfileTile(String name) {
    final spec = (_available[name] as Map).cast<String, dynamic>();
    final isActive = _active == name;
    final numCtx = spec['num_ctx'] as int;
    final temperature = (spec['temperature'] as num).toDouble();
    final topP = (spec['top_p'] as num).toDouble();
    final description = spec['description'] as String;

    return Card(
      color: isActive ? CognithorTheme.accent.withValues(alpha: 0.12) : null,
      child: ListTile(
        leading: Icon(
          isActive ? Icons.radio_button_checked : Icons.radio_button_unchecked,
          color: isActive ? CognithorTheme.accent : null,
        ),
        title: Row(
          children: [
            Text(name),
            const SizedBox(width: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _formatCtx(numCtx),
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ),
          ],
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 4),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(description),
              const SizedBox(height: 4),
              Text(
                'temperature ${temperature.toStringAsFixed(2)} · '
                'top_p ${topP.toStringAsFixed(2)}',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).hintColor,
                ),
              ),
            ],
          ),
        ),
        trailing: _saving && isActive
            ? const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : null,
        onTap: _saving || isActive ? null : () => _setProfile(name),
      ),
    );
  }

  String _formatCtx(int numCtx) {
    if (numCtx >= 1024) {
      final k = numCtx / 1024;
      return '${k.toStringAsFixed(k == k.roundToDouble() ? 0 : 1)}k ctx';
    }
    return '$numCtx ctx';
  }
}
