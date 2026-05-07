// Hardware-Aware Runtime — drift banner for the main shell.
//
// Polls /api/system/health at startup + every 5 minutes. Shows a thin
// MaterialBanner if `drift_detected=true`. Two actions:
//   - "Konfiguration prüfen" → opens HardwareWizardScreen
//   - "Verstecken" → calls /api/system/dismiss-hardware-drift
//                    (server-side cooldown 30 days, see drift_detector.py).
//
// Fail-open: any HTTP / parse error → banner stays hidden. The drift
// signal is informational, never blocking.

import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/onboarding/hardware_wizard_screen.dart';

class HardwareDriftBanner extends StatefulWidget {
  const HardwareDriftBanner({super.key, this.child});

  /// Child to render below the banner (typically the rest of the shell).
  final Widget? child;

  @override
  State<HardwareDriftBanner> createState() => _HardwareDriftBannerState();
}

class _HardwareDriftBannerState extends State<HardwareDriftBanner> {
  static const _poll = Duration(minutes: 5);

  Timer? _timer;
  bool _drift = false;
  List<String> _components = const [];
  bool _dismissedThisSession = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_check());
      _timer = Timer.periodic(_poll, (_) => unawaited(_check()));
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _check() async {
    if (!mounted || _dismissedThisSession) return;
    final conn = context.read<ConnectionProvider>();
    try {
      final token = conn.api.token;
      final r = await http
          .get(
            Uri.parse('${conn.serverUrl}/api/system/health'),
            headers: {if (token != null) 'Authorization': 'Bearer $token'},
          )
          .timeout(const Duration(seconds: 4));
      if (r.statusCode != 200) return;
      final data = jsonDecode(r.body) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _drift = data['drift_detected'] as bool? ?? false;
        _components = ((data['drift_components'] as List?) ?? const [])
            .map((e) => e.toString())
            .toList();
      });
    } on Exception {
      // Silent fail — banner stays as-is.
    }
  }

  Future<void> _dismiss() async {
    setState(() => _dismissedThisSession = true);
    final conn = context.read<ConnectionProvider>();
    try {
      final token = conn.api.token;
      await http
          .post(
            Uri.parse('${conn.serverUrl}/api/system/dismiss-hardware-drift'),
            headers: {if (token != null) 'Authorization': 'Bearer $token'},
          )
          .timeout(const Duration(seconds: 4));
    } on Exception {
      // Local dismissal already applied; server-side cooldown is best-effort.
    }
  }

  void _openWizard() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const HardwareWizardScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final child = widget.child ?? const SizedBox.shrink();
    if (!_drift || _dismissedThisSession) {
      return child;
    }

    final theme = Theme.of(context);
    return Column(
      children: [
        Material(
          color: theme.colorScheme.tertiaryContainer,
          child: SafeArea(
            bottom: false,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  Icon(
                    Icons.warning_amber_outlined,
                    color: theme.colorScheme.onTertiaryContainer,
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Hardware-Konfiguration prüfen',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: theme.colorScheme.onTertiaryContainer,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        if (_components.isNotEmpty)
                          Text(
                            _components.join(', '),
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onTertiaryContainer,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                      ],
                    ),
                  ),
                  TextButton(
                    onPressed: _dismiss,
                    child: const Text('Verstecken'),
                  ),
                  const SizedBox(width: 4),
                  FilledButton(
                    onPressed: _openWizard,
                    child: const Text('Prüfen'),
                  ),
                ],
              ),
            ),
          ),
        ),
        Expanded(child: child),
      ],
    );
  }
}
