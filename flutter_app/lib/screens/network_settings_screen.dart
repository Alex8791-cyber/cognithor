/// Network Settings screen — manage network interfaces and API binding.
library;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';

class NetworkSettingsScreen extends StatefulWidget {
  const NetworkSettingsScreen({super.key});

  @override
  State<NetworkSettingsScreen> createState() => _NetworkSettingsScreenState();
}

class _NetworkSettingsScreenState extends State<NetworkSettingsScreen> {
  List<Map<String, dynamic>> _interfaces = [];
  bool _autoDetect = true;
  String? _bindHost;
  List<String> _activeIps = [];
  bool _loading = true;
  String? _error;
  bool _saving = false;
  bool _dirty = false;

  @override
  void initState() {
    super.initState();
    _loadInterfaces();
  }

  Future<void> _loadInterfaces() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.get('/network/interfaces');
      final list = res['interfaces'];
      setState(() {
        _interfaces = (list is List) ? list.cast<Map<String, dynamic>>() : [];
        _autoDetect = res['auto_detect'] as bool? ?? true;
        _bindHost = res['bind_host'] as String?;
        final active = res['active_ips'];
        _activeIps = (active is List) ? active.cast<String>() : [];
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _saveEndpoints() async {
    setState(() => _saving = true);
    try {
      final api = context.read<ConnectionProvider>().api;
      final enabledIps = _interfaces
          .where((i) => i['_enabled'] == true)
          .map((i) => i['ip'] as String)
          .toList();
      final res = await api.put('/network/endpoints', {
        'enabled_ips': enabledIps,
        'auto_detect': _autoDetect,
      });
      if (mounted) {
        final msg = res['message'] ?? 'Saved';
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(msg.toString())));
        setState(() {
          _bindHost = res['bind_host'] as String?;
          final active = res['active_ips'];
          _activeIps = (active is List) ? active.cast<String>() : [];
          _dirty = false;
          _saving = false;
        });
      }
    } catch (e) {
      setState(() => _saving = false);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(e.toString())));
      }
    }
  }

  String _interfaceTypeName(AppLocalizations l, String type) {
    switch (type) {
      case 'loopback':
        return l.interfaceLoopback;
      case 'lan':
        return l.interfaceLan;
      case 'tailscale':
        return l.interfaceTailscale;
      case 'zerotier':
        return l.interfaceZerotier;
      case 'wireguard':
        return l.interfaceWireguard;
      case 'cloudflare':
        return l.interfaceCloudflare;
      default:
        return l.interfaceUnknown;
    }
  }

  IconData _interfaceIcon(String type) {
    switch (type) {
      case 'loopback':
        return Icons.loop;
      case 'lan':
        return Icons.lan;
      case 'tailscale':
        return Icons.vpn_lock;
      case 'zerotier':
        return Icons.hub;
      case 'wireguard':
        return Icons.shield;
      case 'cloudflare':
        return Icons.cloud;
      default:
        return Icons.device_hub;
    }
  }

  Color _interfaceColor(String type) {
    switch (type) {
      case 'tailscale':
        return const Color(0xFF4A7DFF);
      case 'zerotier':
        return const Color(0xFFFFB800);
      case 'wireguard':
        return const Color(0xFF88171A);
      case 'cloudflare':
        return const Color(0xFFF6821F);
      case 'loopback':
        return Colors.grey;
      default:
        return CognithorTheme.accent;
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l.networkSettings),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadInterfaces,
          ),
        ],
      ),
      floatingActionButton: _dirty
          ? FloatingActionButton.extended(
              onPressed: _saving ? null : _saveEndpoints,
              icon: _saving
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.save),
              label: Text(l.save),
            )
          : null,
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(
              child: Text(_error!, style: const TextStyle(color: Colors.red)),
            )
          : _buildBody(l, theme),
    );
  }

  Widget _buildBody(AppLocalizations l, ThemeData theme) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Current bind host info
        if (_bindHost != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: NeonCard(
              tint: CognithorTheme.accent,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const Icon(Icons.dns, size: 24),
                    const SizedBox(width: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(l.bindHost, style: theme.textTheme.bodySmall),
                        Text(
                          _bindHost!,
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontFamily: 'monospace',
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),

        // Auto-detect toggle
        NeonCard(
          child: SwitchListTile(
            title: Text(l.autoDetect),
            subtitle: Text(l.autoDetectHint, style: theme.textTheme.bodySmall),
            value: _autoDetect,
            onChanged: (v) {
              setState(() {
                _autoDetect = v;
                _dirty = true;
              });
            },
          ),
        ),
        const SizedBox(height: 16),

        // Detected interfaces
        Text(
          l.detectedInterfaces,
          style: theme.textTheme.titleSmall?.copyWith(
            color: CognithorTheme.textSecondary,
          ),
        ),
        const SizedBox(height: 8),

        if (_interfaces.isEmpty)
          CognithorEmptyState(
            icon: Icons.wifi_off,
            title: l.noInterfacesDetected,
          )
        else
          ..._interfaces.map((iface) => _buildInterfaceCard(iface, l, theme)),

        // Restart hint
        if (_dirty) ...[
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.amber.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.amber.withValues(alpha: 0.3)),
            ),
            child: Row(
              children: [
                const Icon(Icons.info_outline, color: Colors.amber, size: 20),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    l.restartRequired,
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildInterfaceCard(
    Map<String, dynamic> iface,
    AppLocalizations l,
    ThemeData theme,
  ) {
    final ip = iface['ip'] as String? ?? '';
    final name = iface['name'] as String? ?? '';
    final type = iface['interface_type'] as String? ?? 'unknown';
    final trusted = iface['trusted'] as bool? ?? false;
    final isEnabled = iface['_enabled'] as bool? ?? _activeIps.contains(ip);

    // Sync local state
    iface['_enabled'] = isEnabled;

    final color = _interfaceColor(type);

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: NeonCard(
        tint: color,
        glowOnHover: true,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Row(
            children: [
              Icon(_interfaceIcon(type), color: color, size: 28),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          _interfaceTypeName(l, type),
                          style: theme.textTheme.titleSmall?.copyWith(
                            color: color,
                          ),
                        ),
                        if (name.isNotEmpty) ...[
                          const SizedBox(width: 8),
                          Text(
                            name,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: CognithorTheme.textSecondary,
                            ),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      ip,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(height: 2),
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: trusted
                                ? Colors.green.withValues(alpha: 0.15)
                                : Colors.orange.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            trusted ? l.trusted : l.untrusted,
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: trusted ? Colors.green : Colors.orange,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              Switch(
                value: isEnabled,
                activeThumbColor: color,
                onChanged: type == 'loopback'
                    ? null // Loopback always on
                    : (v) {
                        setState(() {
                          iface['_enabled'] = v;
                          _dirty = true;
                        });
                      },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
