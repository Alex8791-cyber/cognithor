import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';
import 'package:jarvis_ui/widgets/neon_glow.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';
import 'package:jarvis_ui/widgets/jarvis_stat.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';

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
            backgroundColor: JarvisTheme.red,
          ),
        );
      } else {
        await _loadState();
      }
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(e.toString()),
          backgroundColor: JarvisTheme.red,
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
              backgroundColor: JarvisTheme.red,
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
      return JarvisEmptyState(
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

    final isFrozen = (_state!['is_frozen'] ?? _state!['frozen']) as bool? ?? false;
    final energy = (_state!['somatic_energy'] ?? _state!['energy'] ?? 0).toString();
    final interactions = (_state!['total_interactions'] ?? _state!['interactions'] ?? 0).toString();
    final memories = (_state!['vector_store_count'] ?? _state!['memories'] ?? 0).toString();
    final characterStrength = (_state!['character_strength'] ?? 0).toString();
    final anchors = _state!['genesis_anchors'] as List<dynamic>? ?? [];

    return RefreshIndicator(
      onRefresh: _loadState,
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Status badge
          Row(
            children: [
              JarvisStatusBadge(
                label: isFrozen ? l.identityFrozen : l.identityActive,
                color: isFrozen ? JarvisTheme.orange : JarvisTheme.green,
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
              JarvisStat(
                label: l.identityEnergy,
                value: energy,
                icon: Icons.bolt,
                color: JarvisTheme.accent,
              ),
              JarvisStat(
                label: l.identityInteractions,
                value: interactions,
                icon: Icons.forum,
                color: JarvisTheme.green,
              ),
              JarvisStat(
                label: l.identityMemories,
                value: memories,
                icon: Icons.memory,
                color: JarvisTheme.orange,
              ),
              JarvisStat(
                label: l.identityCharacterStrength,
                value: characterStrength,
                icon: Icons.shield,
                color: JarvisTheme.accent,
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Actions
          JarvisSection(title: l.identity),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              NeonGlow(
                color: JarvisTheme.sectionIdentity,
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
                  icon: Icon(Icons.lock_open, size: 18, color: JarvisTheme.green),
                  label: Text(l.identityUnfreeze),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: JarvisTheme.green,
                    side: BorderSide(color: JarvisTheme.green),
                  ),
                )
              else
                OutlinedButton.icon(
                  onPressed: () => _performAction('freeze'),
                  icon: Icon(Icons.ac_unit, size: 18, color: JarvisTheme.orange),
                  label: Text(l.identityFreeze),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: JarvisTheme.orange,
                    side: BorderSide(color: JarvisTheme.orange),
                  ),
                ),
              OutlinedButton.icon(
                onPressed: _confirmReset,
                icon: Icon(Icons.restart_alt, size: 18, color: JarvisTheme.red),
                label: Text(l.identityReset),
                style: OutlinedButton.styleFrom(
                  foregroundColor: JarvisTheme.red,
                  side: BorderSide(color: JarvisTheme.red),
                ),
              ),
            ],
          ),

          // Genesis anchors
          if (anchors.isNotEmpty) ...[
            const SizedBox(height: 24),
            const JarvisSection(title: 'Genesis Anchors'),
            GlassPanel(
              tint: JarvisTheme.sectionIdentity,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: anchors.map<Widget>((anchor) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 3),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.anchor,
                          size: 14,
                          color: JarvisTheme.sectionIdentity,
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
