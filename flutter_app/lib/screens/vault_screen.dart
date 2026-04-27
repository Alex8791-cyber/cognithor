import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_list_tile.dart';
import 'package:cognithor_ui/widgets/cognithor_loading_skeleton.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';

class VaultScreen extends StatefulWidget {
  const VaultScreen({super.key});

  @override
  State<VaultScreen> createState() => _VaultScreenState();
}

class _VaultScreenState extends State<VaultScreen> {
  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _loadData();
    }
  }

  void _loadData() {
    final provider = context.read<AdminProvider>();
    final api = context.read<ConnectionProvider>().api;
    provider.setApi(api);
    provider.loadVaultStats();
    provider.loadVaultAgents();
  }

  Future<void> _refresh() async {
    final provider = context.read<AdminProvider>();
    await provider.loadVaultStats();
    await provider.loadVaultAgents();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Consumer<AdminProvider>(
      builder: (context, provider, _) {
        if (provider.isLoading &&
            provider.vaultStats == null &&
            provider.vaultAgents.isEmpty) {
          return const Center(
            child: Padding(
              padding: EdgeInsets.all(32),
              child: CognithorLoadingSkeleton(count: 4, height: 60),
            ),
          );
        }

        if (provider.error != null &&
            provider.vaultStats == null &&
            provider.vaultAgents.isEmpty) {
          return CognithorEmptyState(
            icon: Icons.lock_outline,
            title: l.noData,
            subtitle: provider.error,
            action: ElevatedButton.icon(
              onPressed: _refresh,
              icon: const Icon(Icons.refresh),
              label: Text(l.retry),
            ),
          );
        }

        final totalVaults =
            provider.vaultStats?['total_vaults']?.toString() ?? '0';
        final totalEntries =
            provider.vaultStats?['total_entries']?.toString() ?? '0';

        return RefreshIndicator(
          onRefresh: _refresh,
          color: CognithorTheme.accent,
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              // Stats
              CognithorSection(title: l.vaultStats),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: [
                  CognithorStat(
                    label: l.totalVaults,
                    value: totalVaults,
                    icon: Icons.lock,
                    color: CognithorTheme.accent,
                  ),
                  CognithorStat(
                    label: l.totalEntries,
                    value: totalEntries,
                    icon: Icons.key,
                    color: CognithorTheme.green,
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // Agent vaults
              CognithorSection(title: l.agentVaults),

              if (provider.vaultAgents.isEmpty)
                CognithorEmptyState(
                  icon: Icons.lock_outlined,
                  title: l.noVaults,
                )
              else
                ...provider.vaultAgents.map<Widget>((agent) {
                  final a = agent as Map<String, dynamic>;
                  final name =
                      a['agent']?.toString() ?? a['name']?.toString() ?? '';
                  final entries = a['entry_count']?.toString() ?? '0';
                  final lastAccessed = a['last_accessed']?.toString() ?? '';

                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: NeonCard(
                      tint: CognithorTheme.sectionAdmin,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                      child: CognithorListTile(
                        title: name,
                        subtitle: lastAccessed.isNotEmpty
                            ? '${l.totalEntries}: $entries  |  ${l.lastAccessed}: $lastAccessed'
                            : '${l.totalEntries}: $entries',
                        leading: Icon(
                          Icons.person_outline,
                          color: CognithorTheme.accent,
                          size: 20,
                        ),
                      ),
                    ),
                  );
                }),
            ],
          ),
        );
      },
    );
  }
}
