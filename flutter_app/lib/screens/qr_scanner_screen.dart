/// QR Pairing screen — generates a pairing QR from the server and displays it.
///
/// On web: shows the QR payload as copyable text (scan with phone camera app).
/// On native: shows camera placeholder for future mobile_scanner integration.
library;

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';

class QrScannerScreen extends StatefulWidget {
  const QrScannerScreen({super.key});

  @override
  State<QrScannerScreen> createState() => _QrScannerScreenState();
}

class _QrScannerScreenState extends State<QrScannerScreen> {
  bool _loading = true;
  String? _qrPayload;
  String? _deviceId;
  String? _error;
  bool _copied = false;

  @override
  void initState() {
    super.initState();
    _generatePairingQr();
  }

  Future<void> _generatePairingQr() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final api = context.read<ConnectionProvider>().api;
      final res = await api.post('devices/pair', {
        'name': 'Mobile ${DateTime.now().toIso8601String().substring(0, 10)}',
      });

      if (res.containsKey('error')) {
        throw Exception(res['error']);
      }

      setState(() {
        _qrPayload = res['qr_payload'] as String? ?? '';
        _deviceId = res['device_id'] as String? ?? '';
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _copyPayload() {
    if (_qrPayload == null || _qrPayload!.isEmpty) return;
    Clipboard.setData(ClipboardData(text: _qrPayload!));
    setState(() => _copied = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _copied = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: Text(l.pairingQrTitle)),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
            ? _buildError(l, theme)
            : _buildQrDisplay(l, theme),
      ),
    );
  }

  Widget _buildError(AppLocalizations l, ThemeData theme) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error, size: 64, color: Colors.red),
          const SizedBox(height: 16),
          Text(_error!, style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: _generatePairingQr,
            icon: const Icon(Icons.refresh),
            label: Text(l.retry),
          ),
        ],
      ),
    );
  }

  Widget _buildQrDisplay(AppLocalizations l, ThemeData theme) {
    // Pretty-print the JSON payload for display
    String displayPayload = _qrPayload ?? '';
    try {
      final parsed = jsonDecode(displayPayload);
      displayPayload = const JsonEncoder.withIndent('  ').convert(parsed);
    } catch (_) {}

    return SingleChildScrollView(
      child: Column(
        children: [
          Icon(Icons.qr_code_2, size: 80, color: CognithorTheme.accent),
          const SizedBox(height: 16),
          Text(l.pairingQrTitle, style: theme.textTheme.titleLarge),
          const SizedBox(height: 8),
          Text(
            l.pairingQrHint,
            style: theme.textTheme.bodyMedium,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),

          // QR payload card
          NeonCard(
            tint: CognithorTheme.accent,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.key, size: 18),
                      const SizedBox(width: 8),
                      Text(
                        'Device: $_deviceId',
                        style: theme.textTheme.bodySmall,
                      ),
                      const Spacer(),
                      TextButton.icon(
                        onPressed: _copyPayload,
                        icon: Icon(
                          _copied ? Icons.check : Icons.copy,
                          size: 16,
                        ),
                        label: Text(_copied ? 'Copied!' : 'Copy'),
                      ),
                    ],
                  ),
                  const Divider(),
                  SelectableText(
                    displayPayload,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontFamily: 'monospace',
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // Instructions
          NeonCard(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('How to pair:', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 8),
                  _step('1', 'Copy the payload above'),
                  _step('2', 'Open Cognithor on your phone'),
                  _step('3', 'Go to Settings > Connect to Server'),
                  _step('4', 'Paste the payload'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // Done button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: Text(l.confirm),
            ),
          ),
        ],
      ),
    );
  }

  Widget _step(String num, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          CircleAvatar(
            radius: 12,
            backgroundColor: CognithorTheme.accent.withValues(alpha: 0.2),
            child: Text(
              num,
              style: TextStyle(fontSize: 11, color: CognithorTheme.accent),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }
}
