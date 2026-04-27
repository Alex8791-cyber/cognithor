import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_confirmation_dialog.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_list_tile.dart';
import 'package:cognithor_ui/widgets/cognithor_loading_skeleton.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';
import 'package:cognithor_ui/widgets/cognithor_status_badge.dart';

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
              padding: EdgeInsets.all(CognithorTheme.spacing),
              child: CognithorLoadingSkeleton(count: 6, height: 24),
            )
          : _error != null
          ? CognithorEmptyState(
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
                padding: const EdgeInsets.all(CognithorTheme.spacing),
                children: [
                  _buildSystemInfo(l),
                  const SizedBox(height: CognithorTheme.spacing),
                  _buildChannels(l),
                  const SizedBox(height: CognithorTheme.spacing),
                  _buildCommands(l),
                  const SizedBox(height: CognithorTheme.spacing),
                  _buildConnectors(l),
                  const SizedBox(height: CognithorTheme.spacingLg),
                  _buildDangerZone(l),
                ],
              ),
            ),
    );
  }

  Widget _buildSystemInfo(AppLocalizations l) {
    final runtime = _status?['runtime'] as Map<String, dynamic>? ?? {};
    final uptime =
        (_status?['uptime_seconds'] ?? runtime['uptime_seconds'] ?? 0) as num;
    final version = context.read<ConnectionProvider>().backendVersion ?? '?';
    final owner = _status?['owner']?.toString() ?? '-';
    final backend = _status?['llm_backend']?.toString() ?? '-';
    final configVersion = _status?['config_version']?.toString() ?? '-';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        CognithorSection(title: l.systemOverview),
        const SizedBox(height: CognithorTheme.spacingSm),
        Wrap(
          spacing: CognithorTheme.spacingSm,
          runSpacing: CognithorTheme.spacingSm,
          children: [
            CognithorStat(label: l.uptime, value: _formatUptime(uptime)),
            CognithorStat(
              label: l.backendVersion,
              value: version,
              color: CognithorTheme.accent,
            ),
            CognithorStat(
              label: l.owner,
              value: owner,
              color: CognithorTheme.success,
            ),
            CognithorStat(
              label: l.llmBackend,
              value: backend,
              color: CognithorTheme.info,
            ),
          ],
        ),
        const SizedBox(height: CognithorTheme.spacingSm),
        NeonCard(
          tint: CognithorTheme.sectionAdmin,
          child: Row(
            children: [
              const Icon(Icons.settings, size: CognithorTheme.iconSizeSm),
              const SizedBox(width: CognithorTheme.spacingSm),
              Text(
                'Config: $configVersion',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildChannels(AppLocalizations l) {
    final channels = (_status?['active_channels'] as List?) ?? [];
    if (channels.isEmpty) {
      return CognithorEmptyState(
        icon: Icons.podcasts,
        title: l.channels,
        subtitle: l.noData,
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        CognithorSection(title: l.channels),
        const SizedBox(height: CognithorTheme.spacingSm),
        ...channels.map((ch) {
          final name = ch.toString();
          return CognithorListTile(
            title: name,
            leading: const Icon(
              Icons.podcasts,
              size: CognithorTheme.iconSizeMd,
            ),
            trailing: CognithorStatusBadge(
              label: l.enabled,
              color: CognithorTheme.success,
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
        CognithorSection(title: l.commandsTitle),
        const SizedBox(height: CognithorTheme.spacingSm),
        if (_commands.isEmpty)
          CognithorEmptyState(
            icon: Icons.terminal,
            title: l.commandsTitle,
            subtitle: l.noData,
          )
        else
          ..._commands.map((cmd) {
            final name = (cmd is Map ? cmd['name'] : cmd).toString();
            return CognithorListTile(
              title: name,
              leading: const Icon(
                Icons.terminal,
                size: CognithorTheme.iconSizeMd,
              ),
            );
          }),
      ],
    );
  }

  Widget _buildConnectors(AppLocalizations l) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        CognithorSection(title: l.connectorsTitle),
        const SizedBox(height: CognithorTheme.spacingSm),
        if (_connectors.isEmpty)
          CognithorEmptyState(
            icon: Icons.cable,
            title: l.connectorsTitle,
            subtitle: l.noData,
          )
        else
          ..._connectors.map((con) {
            final name = (con is Map ? con['name'] : con).toString();
            final status = con is Map ? con['status']?.toString() : null;
            return CognithorListTile(
              title: name,
              trailing: status != null
                  ? CognithorStatusBadge(
                      label: status,
                      color: status == 'active'
                          ? CognithorTheme.success
                          : CognithorTheme.warning,
                    )
                  : null,
              leading: const Icon(Icons.cable, size: CognithorTheme.iconSizeMd),
            );
          }),
      ],
    );
  }

  Widget _buildDangerZone(AppLocalizations l) {
    return NeonCard(
      tint: CognithorTheme.red,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.warning_amber, color: CognithorTheme.error),
              const SizedBox(width: CognithorTheme.spacingSm),
              Text(
                l.dangerZone,
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(color: CognithorTheme.error),
              ),
            ],
          ),
          const SizedBox(height: CognithorTheme.spacing),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final api = context.read<ConnectionProvider>().api;
                    await api.reloadConfig();
                    if (mounted) {
                      ScaffoldMessenger.of(
                        context,
                      ).showSnackBar(SnackBar(content: Text(l.reload)));
                    }
                  },
                  icon: const Icon(Icons.refresh),
                  label: Text(l.reload),
                ),
              ),
              const SizedBox(width: CognithorTheme.spacingSm),
              Expanded(
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: CognithorTheme.error,
                  ),
                  onPressed: () async {
                    final confirmed = await CognithorConfirmationDialog.show(
                      context,
                      title: l.shutdownServer,
                      message: l.shutdownConfirm,
                      confirmColor: CognithorTheme.error,
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
