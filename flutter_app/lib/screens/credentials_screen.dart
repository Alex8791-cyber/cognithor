import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
// Inline confirmation dialog used for delete
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';

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
        _credentials =
            list.map((e) => e as Map<String, dynamic>).toList();
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
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Credential'),
        content: Text('Delete $service / $key?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: ElevatedButton.styleFrom(
              backgroundColor: JarvisTheme.red,
              foregroundColor: Colors.white,
            ),
            child: const Text('Delete'),
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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Credentials'),
        actions: [
          IconButton(
            icon: Icon(Icons.add, color: JarvisTheme.accent),
            onPressed: _add,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? JarvisEmptyState(
                  icon: Icons.error_outline,
                  title: 'Error',
                  subtitle: _error,
                  action: ElevatedButton(
                      onPressed: _load, child: const Text('Retry')),
                )
              : _credentials.isEmpty
                  ? const JarvisEmptyState(
                      icon: Icons.vpn_key_off,
                      title: 'No Credentials',
                      subtitle: 'Add credentials with the + button',
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: _credentials.length,
                      itemBuilder: (context, i) {
                        final c = _credentials[i];
                        final service =
                            (c['service'] ?? '').toString();
                        final key = (c['key'] ?? '').toString();
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: NeonCard(
                            tint: JarvisTheme.sectionAdmin,
                            glowOnHover: true,
                            child: Row(
                              children: [
                                Icon(Icons.vpn_key, size: 18, color: JarvisTheme.sectionAdmin),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(service, style: Theme.of(context).textTheme.titleMedium),
                                      const SizedBox(height: 4),
                                      Text(key,
                                          style: Theme.of(context)
                                              .textTheme
                                              .bodySmall
                                              ?.copyWith(fontFamily: 'monospace')),
                                    ],
                                  ),
                                ),
                                IconButton(
                                  icon: Icon(Icons.delete,
                                      size: 18, color: JarvisTheme.red),
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
    return AlertDialog(
      title: const Text('Add Credential'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: _service,
            decoration: const InputDecoration(
              labelText: 'Service',
              hintText: 'e.g. openai',
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _key,
            decoration: const InputDecoration(
              labelText: 'Key',
              hintText: 'e.g. api_key',
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _value,
            decoration: const InputDecoration(
              labelText: 'Value',
            ),
            obscureText: true,
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
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
          child: const Text('Add'),
        ),
      ],
    );
  }
}
