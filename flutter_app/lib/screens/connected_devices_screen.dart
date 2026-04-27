/// Connected Devices screen — list, pair, and revoke mobile devices.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/screens/qr_scanner_screen.dart';

class ConnectedDevicesScreen extends StatefulWidget {
  const ConnectedDevicesScreen({super.key});

  @override
  State<ConnectedDevicesScreen> createState() => _ConnectedDevicesScreenState();
}

class _ConnectedDevicesScreenState extends State<ConnectedDevicesScreen> {
  List<Map<String, dynamic>> _devices = [];
  bool _loading = true;
  String? _error;

  // Pairing state
  bool _pairing = false;
  Map<String, dynamic>? _pairingResult;

  @override
  void initState() {
    super.initState();
    _loadDevices();
  }

  Future<void> _loadDevices() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.get('/devices');
      final list = res['devices'];
      setState(() {
        _devices = (list is List) ? list.cast<Map<String, dynamic>>() : [];
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _pairNewDevice() async {
    setState(() => _pairing = true);
    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.post('/devices/pair', {
        'name': 'Mobile ${DateTime.now().toIso8601String().substring(0, 10)}',
      });
      setState(() {
        _pairingResult = res;
        _pairing = false;
      });
      _loadDevices();
    } catch (e) {
      setState(() => _pairing = false);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(e.toString())));
      }
    }
  }

  Future<void> _revokeDevice(String deviceId) async {
    final l = AppLocalizations.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.revokeDevice),
        content: Text(l.revokeDeviceConfirm),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l.cancel),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: Text(l.revokeDevice),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    if (!mounted) return;
    final api = context.read<ConnectionProvider>().api;
    try {
      await api.delete('/devices/$deviceId');
      if (!mounted) return;
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(l.deviceRevoked)));
      }
      _loadDevices();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(e.toString())));
      }
    }
  }

  String _formatTimestamp(dynamic ts) {
    if (ts == null) return '—';
    if (ts is num) {
      final dt = DateTime.fromMillisecondsSinceEpoch(
        (ts * 1000).toInt(),
        isUtc: true,
      ).toLocal();
      return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')}';
    }
    return ts.toString();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l.connectedDevices),
        actions: [
          IconButton(
            icon: const Icon(Icons.qr_code_scanner),
            tooltip: l.scanQrCode,
            onPressed: () async {
              final result = await Navigator.of(context).push<bool>(
                MaterialPageRoute(builder: (_) => const QrScannerScreen()),
              );
              if (result == true) _loadDevices();
            },
          ),
          IconButton(icon: const Icon(Icons.refresh), onPressed: _loadDevices),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _pairing ? null : _pairNewDevice,
        icon: _pairing
            ? const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : const Icon(Icons.add),
        label: Text(l.pairNewDevice),
      ),
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
    return Column(
      children: [
        // QR pairing result banner
        if (_pairingResult != null) _buildPairingBanner(l, theme),
        // Device list
        Expanded(
          child: _devices.isEmpty
              ? CognithorEmptyState(
                  icon: Icons.devices_other,
                  title: l.noDevicesPaired,
                  subtitle: l.noDevicesHint,
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _devices.length,
                  itemBuilder: (ctx, i) =>
                      _buildDeviceCard(ctx, _devices[i], l, theme),
                ),
        ),
      ],
    );
  }

  Widget _buildPairingBanner(AppLocalizations l, ThemeData theme) {
    final qr = _pairingResult?['qr_payload'] ?? '';
    final token = _pairingResult?['token'] ?? '';
    return Container(
      margin: const EdgeInsets.all(16),
      child: NeonCard(
        tint: CognithorTheme.accent,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.qr_code, size: 28),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      l.pairingQrTitle,
                      style: theme.textTheme.titleMedium,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => setState(() => _pairingResult = null),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(l.pairingQrHint, style: theme.textTheme.bodySmall),
              const SizedBox(height: 12),
              // QR payload as copyable text (real QR rendering needs qr_flutter)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SelectableText(
                  qr.toString(),
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontFamily: 'monospace',
                    fontSize: 11,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  TextButton.icon(
                    onPressed: () {
                      Clipboard.setData(ClipboardData(text: qr.toString()));
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Copied to clipboard')),
                      );
                    },
                    icon: const Icon(Icons.copy, size: 16),
                    label: Text(l.copyQrPayload),
                  ),
                  const SizedBox(width: 8),
                  TextButton.icon(
                    onPressed: () {
                      Clipboard.setData(ClipboardData(text: token.toString()));
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Token copied')),
                      );
                    },
                    icon: const Icon(Icons.key, size: 16),
                    label: Text(l.copyToken),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDeviceCard(
    BuildContext context,
    Map<String, dynamic> device,
    AppLocalizations l,
    ThemeData theme,
  ) {
    final name = device['name'] ?? device['device_name'] ?? 'Unknown';
    final id = device['device_id'] ?? '';
    final pairedAt = _formatTimestamp(device['paired_at']);
    final expiresAt = _formatTimestamp(device['expires_at']);

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: NeonCard(
        glowOnHover: true,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              const Icon(Icons.phone_android, size: 32),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(name.toString(), style: theme.textTheme.titleSmall),
                    const SizedBox(height: 4),
                    Text(
                      '${l.deviceId}: ${id.toString().substring(0, (id.toString().length > 12 ? 12 : id.toString().length))}...',
                      style: theme.textTheme.bodySmall,
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${l.pairedAt}: $pairedAt  •  ${l.expiresAt}: $expiresAt',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: CognithorTheme.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.delete_outline, color: Colors.red),
                tooltip: l.revokeDevice,
                onPressed: () => _revokeDevice(id.toString()),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
