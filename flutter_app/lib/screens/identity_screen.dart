import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/neon_glow.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';
import 'package:cognithor_ui/widgets/cognithor_status_badge.dart';

class IdentityScreen extends StatefulWidget {
  const IdentityScreen({super.key});

  @override
  State<IdentityScreen> createState() => _IdentityScreenState();
}

class _IdentityScreenState extends State<IdentityScreen> {
  Map<String, dynamic>? _state;
  bool _loading = true;
  bool _available = true;
  String? _error;

  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _loadState();
    }
  }

  Future<void> _loadState() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.getIdentityState();
      if (result.containsKey('error')) {
        setState(() {
          _error = result['error'] as String;
          _available = false;
          _loading = false;
        });
      } else {
        setState(() {
          _available = result['available'] as bool? ?? true;
          _state = result;
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _performAction(String action) async {
    final api = context.read<ConnectionProvider>().api;
    final messenger = ScaffoldMessenger.of(context);

    try {
      final result = await api.post('identity/$action');
      if (result.containsKey('error')) {
        messenger.showSnackBar(
          SnackBar(
            content: Text(result['error'] as String? ?? 'Error'),
            backgroundColor: CognithorTheme.red,
          ),
        );
      } else {
        await _loadState();
      }
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(e.toString()),
          backgroundColor: CognithorTheme.red,
        ),
      );
    }
  }

  Future<void> _confirmReset() async {
    final l = AppLocalizations.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.identityReset),
        content: Text(l.identityResetConfirm),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: ElevatedButton.styleFrom(
              backgroundColor: CognithorTheme.red,
            ),
            child: Text(l.identityReset),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await _performAction('reset');
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    if (_loading) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(l.loading),
          ],
        ),
      );
    }

    if (!_available || _state == null) {
      return CognithorEmptyState(
        icon: Icons.psychology_outlined,
        title: l.identityNotAvailable,
        subtitle: _error ?? l.identityInstallHint,
        action: ElevatedButton.icon(
          onPressed: _loadState,
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    final isFrozen =
        (_state!['is_frozen'] ?? _state!['frozen']) as bool? ?? false;
    final energy = (_state!['somatic_energy'] ?? _state!['energy'] ?? 0)
        .toString();
    final interactions =
        (_state!['total_interactions'] ?? _state!['interactions'] ?? 0)
            .toString();
    final memories = (_state!['vector_store_count'] ?? _state!['memories'] ?? 0)
        .toString();
    final characterStrength = (_state!['character_strength'] ?? 0).toString();
    final anchors = _state!['genesis_anchors'] as List<dynamic>? ?? [];

    return RefreshIndicator(
      onRefresh: _loadState,
      color: CognithorTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Status badge
          Row(
            children: [
              CognithorStatusBadge(
                label: isFrozen ? l.identityFrozen : l.identityActive,
                color: isFrozen ? CognithorTheme.orange : CognithorTheme.green,
                icon: isFrozen ? Icons.ac_unit : Icons.check_circle,
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Stats grid
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              CognithorStat(
                label: l.identityEnergy,
                value: energy,
                icon: Icons.bolt,
                color: CognithorTheme.accent,
              ),
              CognithorStat(
                label: l.identityInteractions,
                value: interactions,
                icon: Icons.forum,
                color: CognithorTheme.green,
              ),
              CognithorStat(
                label: l.identityMemories,
                value: memories,
                icon: Icons.memory,
                color: CognithorTheme.orange,
              ),
              CognithorStat(
                label: l.identityCharacterStrength,
                value: characterStrength,
                icon: Icons.shield,
                color: CognithorTheme.accent,
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Actions
          CognithorSection(title: l.identity),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              NeonGlow(
                color: CognithorTheme.sectionIdentity,
                intensity: 0.25,
                blurRadius: 10,
                child: ElevatedButton.icon(
                  onPressed: () => _performAction('dream'),
                  icon: const Icon(Icons.nightlight_round, size: 18),
                  label: Text(l.identityDream),
                ),
              ),
              if (isFrozen)
                OutlinedButton.icon(
                  onPressed: () => _performAction('unfreeze'),
                  icon: Icon(
                    Icons.lock_open,
                    size: 18,
                    color: CognithorTheme.green,
                  ),
                  label: Text(l.identityUnfreeze),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: CognithorTheme.green,
                    side: BorderSide(color: CognithorTheme.green),
                  ),
                )
              else
                OutlinedButton.icon(
                  onPressed: () => _performAction('freeze'),
                  icon: Icon(
                    Icons.ac_unit,
                    size: 18,
                    color: CognithorTheme.orange,
                  ),
                  label: Text(l.identityFreeze),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: CognithorTheme.orange,
                    side: BorderSide(color: CognithorTheme.orange),
                  ),
                ),
              OutlinedButton.icon(
                onPressed: _confirmReset,
                icon: Icon(
                  Icons.restart_alt,
                  size: 18,
                  color: CognithorTheme.red,
                ),
                label: Text(l.identityReset),
                style: OutlinedButton.styleFrom(
                  foregroundColor: CognithorTheme.red,
                  side: BorderSide(color: CognithorTheme.red),
                ),
              ),
            ],
          ),

          // Genesis anchors
          if (anchors.isNotEmpty) ...[
            const SizedBox(height: 24),
            CognithorSection(title: l.identityGenesisAnchors),
            NeonCard(
              tint: CognithorTheme.sectionIdentity,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: anchors.map<Widget>((anchor) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 3),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(
                          Icons.anchor,
                          size: 14,
                          color: CognithorTheme.sectionIdentity,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            anchor.toString(),
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        ),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
