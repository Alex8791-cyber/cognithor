import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
// Inline confirmation dialog used for delete
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';

class CredentialsScreen extends StatefulWidget {
  const CredentialsScreen({super.key});

  @override
  State<CredentialsScreen> createState() => _CredentialsScreenState();
}

class _CredentialsScreenState extends State<CredentialsScreen> {
  List<Map<String, dynamic>> _credentials = [];
  bool _loading = true;
  String? _error;
  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _load();
    }
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.getCredentials();
      final list = result['credentials'] as List? ?? [];
      if (!mounted) return;
      setState(() {
        _credentials = list.map((e) => e as Map<String, dynamic>).toList();
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _add() async {
    final api = context.read<ConnectionProvider>().api;
    final result = await showDialog<Map<String, String>>(
      context: context,
      builder: (_) => const _AddCredentialDialog(),
    );
    if (result == null) return;

    await api.addCredential(result);
    await _load();
  }

  Future<void> _delete(String service, String key) async {
    final l = AppLocalizations.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.deleteCredential),
        content: Text('${l.delete} $service / $key?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: ElevatedButton.styleFrom(
              backgroundColor: CognithorTheme.red,
              foregroundColor: Colors.white,
            ),
            child: Text(l.delete),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    if (!mounted) return;
    final api = context.read<ConnectionProvider>().api;
    await api.deleteCredential(service, key);
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(l.credentialsTitle),
        actions: [
          IconButton(
            icon: Icon(Icons.add, color: CognithorTheme.accent),
            onPressed: _add,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? CognithorEmptyState(
              icon: Icons.error_outline,
              title: l.errorLabel,
              subtitle: _error,
              action: ElevatedButton(onPressed: _load, child: Text(l.retry)),
            )
          : _credentials.isEmpty
          ? CognithorEmptyState(
              icon: Icons.vpn_key_off,
              title: l.noCredentials,
              subtitle: l.addCredential,
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _credentials.length,
              itemBuilder: (context, i) {
                final c = _credentials[i];
                final service = (c['service'] ?? '').toString();
                final key = (c['key'] ?? '').toString();
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: NeonCard(
                    tint: CognithorTheme.sectionAdmin,
                    glowOnHover: true,
                    child: Row(
                      children: [
                        const Icon(
                          Icons.vpn_key,
                          size: 18,
                          color: CognithorTheme.sectionAdmin,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                service,
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                              const SizedBox(height: 4),
                              Text(
                                key,
                                style: Theme.of(context).textTheme.bodySmall
                                    ?.copyWith(fontFamily: 'monospace'),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.delete,
                            size: 18,
                            color: CognithorTheme.red,
                          ),
                          onPressed: () => _delete(service, key),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}

class _AddCredentialDialog extends StatefulWidget {
  const _AddCredentialDialog();

  @override
  State<_AddCredentialDialog> createState() => _AddCredentialDialogState();
}

class _AddCredentialDialogState extends State<_AddCredentialDialog> {
  final _service = TextEditingController();
  final _key = TextEditingController();
  final _value = TextEditingController();

  @override
  void dispose() {
    _service.dispose();
    _key.dispose();
    _value.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return AlertDialog(
      title: Text(l.addCredential),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: _service,
            decoration: InputDecoration(
              labelText: l.service,
              hintText: 'e.g. openai',
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _key,
            decoration: InputDecoration(
              labelText: l.key,
              hintText: 'e.g. api_key',
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _value,
            decoration: InputDecoration(labelText: l.value),
            obscureText: true,
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(l.cancel),
        ),
        ElevatedButton(
          onPressed: () {
            if (_service.text.isNotEmpty && _key.text.isNotEmpty) {
              Navigator.of(context).pop({
                'service': _service.text,
                'key': _key.text,
                'value': _value.text,
              });
            }
          },
          child: Text(l.addCredential),
        ),
      ],
    );
  }
}
