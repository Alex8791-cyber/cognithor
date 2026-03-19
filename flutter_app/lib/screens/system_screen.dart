import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
import 'package:jarvis_ui/widgets/jarvis_confirmation_dialog.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_list_tile.dart';
import 'package:jarvis_ui/widgets/jarvis_loading_skeleton.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';
import 'package:jarvis_ui/widgets/jarvis_stat.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';

class SystemScreen extends StatefulWidget {
  const SystemScreen({super.key});

  @override
  State<SystemScreen> createState() => _SystemScreenState();
}

class _SystemScreenState extends State<SystemScreen> {
  bool _isLoading = true;
  String? _error;
  Map<String, dynamic>? _status;
  List<dynamic> _commands = [];
  List<dynamic> _connectors = [];
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
    final api = context.read<ConnectionProvider>().api;
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final results = await Future.wait([
        api.getSystemStatus(),
        api.getCommands(),
        api.getConnectors(),
      ]);
      if (!mounted) return;
      setState(() {
        _status = results[0];
        _commands = (results[1]['commands'] as List?) ?? [];
        _connectors = (results[2]['connectors'] as List?) ?? [];
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  String _formatUptime(num seconds) {
    final d = seconds ~/ 86400;
    final h = (seconds % 86400) ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    if (d > 0) return '${d}d ${h}h ${m}m';
    if (h > 0) return '${h}h ${m}m';
    return '${m}m';
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(title: Text(l.systemStatus)),
      body: _isLoading
          ? const Padding(
              padding: EdgeInsets.all(JarvisTheme.spacing),
              child: JarvisLoadingSkeleton(count: 6, height: 24),
            )
          : _error != null
              ? JarvisEmptyState(
                  icon: Icons.error_outline,
                  title: l.errorLabel,
                  subtitle: _error,
                  action: ElevatedButton.icon(
                    onPressed: _load,
                    icon: const Icon(Icons.refresh),
                    label: Text(l.retry),
                  ),
                )
              : RefreshIndicator(
                  onRefresh: _load,
                  child: ListView(
                    padding: const EdgeInsets.all(JarvisTheme.spacing),
                    children: [
                      _buildSystemInfo(l),
                      const SizedBox(height: JarvisTheme.spacing),
                      _buildChannels(l),
                      const SizedBox(height: JarvisTheme.spacing),
                      _buildCommands(l),
                      const SizedBox(height: JarvisTheme.spacing),
                      _buildConnectors(l),
                      const SizedBox(height: JarvisTheme.spacingLg),
                      _buildDangerZone(l),
                    ],
                  ),
                ),
    );
  }

  Widget _buildSystemInfo(AppLocalizations l) {
    final runtime = _status?['runtime'] as Map<String, dynamic>? ?? {};
    final uptime = (_status?['uptime_seconds'] ?? runtime['uptime_seconds'] ?? 0) as num;
    final version = context.read<ConnectionProvider>().backendVersion ?? '?';
    final owner = _status?['owner']?.toString() ?? '-';
    final backend = _status?['llm_backend']?.toString() ?? '-';
    final configVersion = _status?['config_version']?.toString() ?? '-';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        JarvisSection(title: l.systemOverview),
        const SizedBox(height: JarvisTheme.spacingSm),
        Wrap(
          spacing: JarvisTheme.spacingSm,
          runSpacing: JarvisTheme.spacingSm,
          children: [
            JarvisStat(label: l.uptime, value: _formatUptime(uptime)),
            JarvisStat(
              label: l.backendVersion,
              value: version,
              color: JarvisTheme.accent,
            ),
            JarvisStat(
              label: l.owner,
              value: owner,
              color: JarvisTheme.success,
            ),
            JarvisStat(
              label: l.llmBackend,
              value: backend,
              color: JarvisTheme.info,
            ),
          ],
        ),
        const SizedBox(height: JarvisTheme.spacingSm),
        NeonCard(
          tint: JarvisTheme.sectionAdmin,
          child: Row(
            children: [
              const Icon(Icons.settings, size: JarvisTheme.iconSizeSm),
              const SizedBox(width: JarvisTheme.spacingSm),
              Text('Config: $configVersion',
                  style: Theme.of(context).textTheme.bodyMedium),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildChannels(AppLocalizations l) {
    final channels =
        (_status?['active_channels'] as List?) ?? [];
    if (channels.isEmpty) {
      return JarvisEmptyState(
        icon: Icons.podcasts,
        title: l.channels,
        subtitle: l.noData,
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        JarvisSection(title: l.channels),
        const SizedBox(height: JarvisTheme.spacingSm),
        ...channels.map((ch) {
          final name = ch.toString();
          return JarvisListTile(
            title: name,
            leading: const Icon(Icons.podcasts, size: JarvisTheme.iconSizeMd),
            trailing: JarvisStatusBadge(
              label: l.enabled,
              color: JarvisTheme.success,
            ),
          );
        }),
      ],
    );
  }

  Widget _buildCommands(AppLocalizations l) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        JarvisSection(title: l.commandsTitle),
        const SizedBox(height: JarvisTheme.spacingSm),
        if (_commands.isEmpty)
          JarvisEmptyState(
            icon: Icons.terminal,
            title: l.commandsTitle,
            subtitle: l.noData,
          )
        else
          ..._commands.map((cmd) {
            final name = (cmd is Map ? cmd['name'] : cmd).toString();
            return JarvisListTile(
              title: name,
              leading:
                  const Icon(Icons.terminal, size: JarvisTheme.iconSizeMd),
            );
          }),
      ],
    );
  }

  Widget _buildConnectors(AppLocalizations l) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        JarvisSection(title: l.connectorsTitle),
        const SizedBox(height: JarvisTheme.spacingSm),
        if (_connectors.isEmpty)
          JarvisEmptyState(
            icon: Icons.cable,
            title: l.connectorsTitle,
            subtitle: l.noData,
          )
        else
          ..._connectors.map((con) {
            final name = (con is Map ? con['name'] : con).toString();
            final status = con is Map ? con['status']?.toString() : null;
            return JarvisListTile(
              title: name,
              trailing: status != null
                  ? JarvisStatusBadge(
                      label: status,
                      color: status == 'active'
                          ? JarvisTheme.success
                          : JarvisTheme.warning,
                    )
                  : null,
              leading: const Icon(Icons.cable, size: JarvisTheme.iconSizeMd),
            );
          }),
      ],
    );
  }

  Widget _buildDangerZone(AppLocalizations l) {
    return NeonCard(
      tint: JarvisTheme.red,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.warning_amber, color: JarvisTheme.error),
              const SizedBox(width: JarvisTheme.spacingSm),
              Text(
                l.dangerZone,
                style: Theme.of(context)
                    .textTheme
                    .titleLarge
                    ?.copyWith(color: JarvisTheme.error),
              ),
            ],
          ),
          const SizedBox(height: JarvisTheme.spacing),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final api = context.read<ConnectionProvider>().api;
                    await api.reloadConfig();
                    if (mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text(l.reload)),
                      );
                    }
                  },
                  icon: const Icon(Icons.refresh),
                  label: Text(l.reload),
                ),
              ),
              const SizedBox(width: JarvisTheme.spacingSm),
              Expanded(
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: JarvisTheme.error,
                  ),
                  onPressed: () async {
                    final confirmed = await JarvisConfirmationDialog.show(
                      context,
                      title: l.shutdownServer,
                      message: l.shutdownConfirm,
                      confirmColor: JarvisTheme.error,
                      icon: Icons.power_settings_new,
                    );
                    if (confirmed && mounted) {
                      final api = context.read<ConnectionProvider>().api;
                      await api.shutdownServer();
                    }
                  },
                  icon: const Icon(Icons.power_settings_new),
                  label: Text(l.shutdownServer),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
