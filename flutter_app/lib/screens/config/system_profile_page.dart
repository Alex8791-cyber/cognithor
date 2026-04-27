import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class SystemProfilePage extends StatefulWidget {
  const SystemProfilePage({super.key});

  @override
  State<SystemProfilePage> createState() => _SystemProfilePageState();
}

class _SystemProfilePageState extends State<SystemProfilePage> {
  Map<String, dynamic>? _profile;
  bool _loading = true;
  bool _rescanning = false;
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
      final data = await api.get('system/profile');
      if (!mounted) return;
      setState(() {
        _profile = data;
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

  Future<void> _rescan() async {
    setState(() {
      _rescanning = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final data = await api.post('system/rescan', {});
      if (!mounted) return;
      setState(() {
        _profile = data;
        _rescanning = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _rescanning = false;
        _error = e.toString();
      });
    }
  }

  Color _tierColor(String tier) {
    return switch (tier.toLowerCase()) {
      'minimal' => CognithorTheme.red,
      'standard' => CognithorTheme.orange,
      'power' => CognithorTheme.green,
      'enterprise' => CognithorTheme.accent,
      _ => CognithorTheme.textSecondary,
    };
  }

  Color _statusColor(String status) {
    return switch (status.toLowerCase()) {
      'ok' => CognithorTheme.green,
      'warn' => CognithorTheme.orange,
      'fail' => CognithorTheme.red,
      _ => CognithorTheme.textSecondary,
    };
  }

  IconData _statusIcon(String status) {
    return switch (status.toLowerCase()) {
      'ok' => Icons.check_circle,
      'warn' => Icons.warning,
      'fail' => Icons.error,
      _ => Icons.help_outline,
    };
  }

  IconData _detectionIcon(String key) {
    return switch (key.toLowerCase()) {
      'cpu' => Icons.developer_board,
      'ram' => Icons.memory,
      'gpu' => Icons.videogame_asset,
      'disk' => Icons.storage,
      'network' => Icons.wifi,
      'ollama' => Icons.smart_toy,
      'lmstudio' => Icons.hub,
      'os' => Icons.computer,
      _ => Icons.info_outline,
    };
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null && _profile == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline, size: 48, color: CognithorTheme.red),
            const SizedBox(height: 16),
            Text(
              _error!,
              style: TextStyle(color: CognithorTheme.textSecondary),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _load,
              icon: const Icon(Icons.refresh, size: 16),
              label: Text(l.retry),
            ),
          ],
        ),
      );
    }

    final tier = (_profile?['tier'] as String?) ?? 'unknown';
    final recommendedMode =
        (_profile?['recommended_mode'] as String?) ?? 'unknown';
    final results =
        (_profile?['results'] as Map<String, dynamic>?) ?? <String, dynamic>{};

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Header card with tier badge and recommended mode
        _buildHeaderCard(l, tier, recommendedMode),
        const SizedBox(height: 16),
        // Detection results
        ...results.entries.map(
          (e) => _buildDetectionTile(e.key, e.value as Map<String, dynamic>),
        ),
        const SizedBox(height: 24),
        // Rescan button
        Center(
          child: ElevatedButton.icon(
            onPressed: _rescanning ? null : _rescan,
            icon: _rescanning
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh, size: 18),
            label: Text(l.systemRescan),
          ),
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildHeaderCard(
    AppLocalizations l,
    String tier,
    String recommendedMode,
  ) {
    final color = _tierColor(tier);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            // Tier badge
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    l.systemTier,
                    style: TextStyle(
                      fontSize: 12,
                      color: CognithorTheme.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: color.withValues(alpha: 0.4),
                        width: 1,
                      ),
                    ),
                    child: Text(
                      tier.toUpperCase(),
                      style: TextStyle(
                        color: color,
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                        letterSpacing: 1.2,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            // Recommended mode
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    l.systemRecommendedMode,
                    style: TextStyle(
                      fontSize: 12,
                      color: CognithorTheme.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    recommendedMode,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectionTile(String key, Map<String, dynamic> result) {
    final value = (result['value'] as String?) ?? '';
    final status = (result['status'] as String?) ?? 'unknown';
    final rawData =
        (result['raw_data'] as Map<String, dynamic>?) ?? <String, dynamic>{};
    final sColor = _statusColor(status);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ExpansionTile(
        leading: Icon(_detectionIcon(key), size: 22),
        title: Row(
          children: [
            Expanded(
              child: Text(
                key.toUpperCase(),
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                ),
              ),
            ),
            Flexible(
              child: Text(
                value,
                style: TextStyle(
                  fontSize: 13,
                  color: CognithorTheme.textSecondary,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        trailing: Icon(_statusIcon(status), color: sColor, size: 20),
        children: rawData.isEmpty
            ? [
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    'No raw data',
                    style: TextStyle(
                      color: CognithorTheme.textTertiary,
                      fontSize: 12,
                    ),
                  ),
                ),
              ]
            : rawData.entries
                  .map(
                    (e) => ListTile(
                      dense: true,
                      visualDensity: VisualDensity.compact,
                      title: Text(
                        e.key,
                        style: TextStyle(
                          fontSize: 12,
                          color: CognithorTheme.textSecondary,
                          fontFamily: 'monospace',
                        ),
                      ),
                      trailing: Text(
                        '${e.value}',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                  )
                  .toList(),
      ),
    );
  }
}
