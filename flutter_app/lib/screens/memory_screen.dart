import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/memory_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
import 'package:jarvis_ui/widgets/neon_glow.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_list_tile.dart';
import 'package:jarvis_ui/widgets/jarvis_loading_skeleton.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';
import 'package:jarvis_ui/widgets/jarvis_stat.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';
import 'package:jarvis_ui/widgets/jarvis_tab_bar.dart';

class MemoryScreen extends StatefulWidget {
  const MemoryScreen({super.key});

  @override
  State<MemoryScreen> createState() => _MemoryScreenState();
}

class _MemoryScreenState extends State<MemoryScreen> {
  int _tabIndex = 0;
  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      final provider = context.read<MemoryProvider>();
      final api = context.read<ConnectionProvider>().api;
      provider.setApi(api);
      _loadAll(provider);
    }
  }

  void _loadAll(MemoryProvider provider) {
    provider.loadGraphStats();
    provider.loadEntities();
    provider.loadHygieneStats();
    provider.loadQuarantine();
    provider.loadExplainability();
    provider.loadTrails();
    provider.loadLowTrustTrails();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Consumer<MemoryProvider>(
      builder: (context, provider, _) {
        return Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: JarvisTabBar(
                tabs: [l.knowledgeGraph, l.hygiene, l.explainability],
                icons: const [
                  Icons.hub_outlined,
                  Icons.health_and_safety_outlined,
                  Icons.account_tree_outlined,
                ],
                selectedIndex: _tabIndex,
                onChanged: (i) => setState(() => _tabIndex = i),
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: _buildTabContent(provider, l),
            ),
          ],
        );
      },
    );
  }

  Widget _buildTabContent(MemoryProvider provider, AppLocalizations l) {
    if (provider.isLoading) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: JarvisLoadingSkeleton(count: 5, height: 80),
        ),
      );
    }

    if (provider.error != null) {
      return JarvisEmptyState(
        icon: Icons.error_outline,
        title: l.noData,
        subtitle: provider.error,
        action: ElevatedButton.icon(
          onPressed: () => _loadAll(provider),
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    return switch (_tabIndex) {
      0 => _buildGraphTab(provider, l),
      1 => _buildHygieneTab(provider, l),
      2 => _buildExplainabilityTab(provider, l),
      _ => const SizedBox.shrink(),
    };
  }

  Widget _buildGraphTab(MemoryProvider provider, AppLocalizations l) {
    final stats = provider.graphStats;
    final entityCount = stats?['entity_count']?.toString() ?? '0';
    final relationCount = stats?['relation_count']?.toString() ?? '0';
    final entityTypes = stats?['entity_types']?.toString() ?? '0';

    return RefreshIndicator(
      onRefresh: () async {
        await provider.loadGraphStats();
        await provider.loadEntities();
      },
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Stats row
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              JarvisStat(
                label: l.entities,
                value: entityCount,
                icon: Icons.circle,
                color: JarvisTheme.accent,
              ),
              JarvisStat(
                label: l.relations,
                value: relationCount,
                icon: Icons.link,
                color: JarvisTheme.green,
              ),
              JarvisStat(
                label: l.entityTypes,
                value: entityTypes,
                icon: Icons.category,
                color: JarvisTheme.orange,
              ),
            ],
          ),
          const SizedBox(height: 24),

          JarvisSection(title: l.entities),

          if (provider.entities.isEmpty)
            JarvisEmptyState(
              icon: Icons.hub_outlined,
              title: l.noEntities,
            )
          else
            ...provider.entities.map<Widget>((entity) {
              final e = entity as Map<String, dynamic>;
              final name = e['name']?.toString() ?? '';
              final type = e['type']?.toString() ?? '';
              final relations = e['relation_count']?.toString() ?? '0';

              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: NeonCard(
                tint: JarvisTheme.sectionAdmin,
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                child: JarvisListTile(
                  title: name,
                  subtitle: '$relations ${l.relations}',
                  leading: JarvisStatusBadge(
                    label: type,
                    color: JarvisTheme.accent,
                  ),
                  dense: true,
                ),
                ),
              );
            }),
        ],
      ),
    );
  }

  Widget _buildHygieneTab(MemoryProvider provider, AppLocalizations l) {
    final stats = provider.hygieneStats;
    final totalScans = stats?['total_scans']?.toString() ?? '0';
    final threats = stats?['threats']?.toString() ?? '0';
    final threatRate = stats?['threat_rate']?.toString() ?? '0%';
    final quarantinedCount = stats?['quarantined']?.toString() ?? '0';

    return RefreshIndicator(
      onRefresh: () async {
        await provider.loadHygieneStats();
        await provider.loadQuarantine();
      },
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Stats row
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              JarvisStat(
                label: l.totalScans,
                value: totalScans,
                icon: Icons.search,
                color: JarvisTheme.accent,
              ),
              JarvisStat(
                label: l.threats,
                value: threats,
                icon: Icons.warning_amber,
                color: JarvisTheme.red,
              ),
              JarvisStat(
                label: l.threatRate,
                value: threatRate,
                icon: Icons.percent,
                color: JarvisTheme.orange,
              ),
              JarvisStat(
                label: l.quarantine,
                value: quarantinedCount,
                icon: Icons.shield,
                color: JarvisTheme.orange,
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Scan button
          Center(
            child: NeonGlow(
              color: JarvisTheme.sectionAdmin,
              intensity: 0.25,
              blurRadius: 10,
              child: ElevatedButton.icon(
                onPressed: provider.isLoading ? null : _scanHygiene,
                icon: const Icon(Icons.play_arrow, size: 18),
                label: Text(l.scanNow),
              ),
            ),
          ),
          const SizedBox(height: 24),

          // Quarantine list
          JarvisSection(title: l.quarantine),

          if (provider.quarantined.isEmpty)
            JarvisEmptyState(
              icon: Icons.shield_outlined,
              title: l.noQuarantine,
            )
          else
            ...provider.quarantined.map<Widget>((item) {
              final q = item as Map<String, dynamic>;
              final name = q['name']?.toString() ?? '';
              final reason = q['reason']?.toString() ?? '';
              final timestamp = q['timestamp']?.toString() ?? '';

              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: NeonCard(
                tint: JarvisTheme.sectionAdmin,
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                child: JarvisListTile(
                  title: name,
                  subtitle: '$reason\n$timestamp',
                  leading: Icon(
                    Icons.warning_amber,
                    color: JarvisTheme.red,
                    size: 20,
                  ),
                  dense: true,
                ),
                ),
              );
            }),
        ],
      ),
    );
  }

  Future<void> _scanHygiene() async {
    final provider = context.read<MemoryProvider>();
    final messenger = ScaffoldMessenger.of(context);
    final l = AppLocalizations.of(context);
    await provider.scanHygiene();
    if (mounted && provider.error == null) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(l.scanComplete),
          backgroundColor: JarvisTheme.green,
        ),
      );
      // Reload hygiene stats and quarantine after successful scan
      await provider.loadHygieneStats();
      await provider.loadQuarantine();
    }
  }

  Widget _buildExplainabilityTab(
    MemoryProvider provider,
    AppLocalizations l,
  ) {
    final stats = provider.explainabilityStats;
    final totalRequests = stats?['total_requests']?.toString() ?? '0';
    final activeTrails = stats?['active_trails']?.toString() ?? '0';
    final completed = stats?['completed']?.toString() ?? '0';
    final avgConfidence = stats?['avg_confidence']?.toString() ?? '0';

    return RefreshIndicator(
      onRefresh: () async {
        await provider.loadExplainability();
        await provider.loadTrails();
        await provider.loadLowTrustTrails();
      },
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Stats
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              JarvisStat(
                label: l.totalRequests,
                value: totalRequests,
                icon: Icons.analytics,
                color: JarvisTheme.accent,
              ),
              JarvisStat(
                label: l.activeTrails,
                value: activeTrails,
                icon: Icons.route,
                color: JarvisTheme.green,
              ),
              JarvisStat(
                label: l.completedTrails,
                value: completed,
                icon: Icons.check_circle,
                color: JarvisTheme.green,
              ),
              JarvisStat(
                label: l.confidence,
                value: avgConfidence,
                icon: Icons.speed,
                color: JarvisTheme.orange,
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Decision trails
          JarvisSection(title: l.decisionTrails),

          if (provider.trails.isEmpty)
            JarvisEmptyState(
              icon: Icons.account_tree_outlined,
              title: l.noTrails,
            )
          else
            ...provider.trails.map<Widget>((trail) {
              final t = trail as Map<String, dynamic>;
              final id = t['id']?.toString() ?? '';
              final status = t['status']?.toString() ?? '';
              final confidence =
                  (t['confidence'] as num?)?.toDouble() ?? 0.0;

              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: NeonCard(
                tint: JarvisTheme.sectionAdmin,
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                child: JarvisListTile(
                  title: id,
                  subtitle: status,
                  trailing: JarvisStatusBadge(
                    label: '${(confidence * 100).toStringAsFixed(0)}%',
                    color: _confidenceColor(confidence),
                    icon: Icons.speed,
                  ),
                  dense: true,
                ),
                ),
              );
            }),

          // Low trust section
          if (provider.lowTrustTrails.isNotEmpty) ...[
            const SizedBox(height: 24),
            JarvisSection(title: l.lowTrust),
            ...provider.lowTrustTrails.map<Widget>((trail) {
              final t = trail as Map<String, dynamic>;
              final id = t['id']?.toString() ?? '';
              final status = t['status']?.toString() ?? '';
              final confidence =
                  (t['confidence'] as num?)?.toDouble() ?? 0.0;

              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: NeonCard(
                tint: JarvisTheme.sectionAdmin,
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                child: JarvisListTile(
                  title: id,
                  subtitle: status,
                  trailing: JarvisStatusBadge(
                    label: '${(confidence * 100).toStringAsFixed(0)}%',
                    color: JarvisTheme.red,
                    icon: Icons.warning_amber,
                  ),
                  dense: true,
                ),
                ),
              );
            }),
          ],
        ],
      ),
    );
  }

  Color _confidenceColor(double confidence) {
    if (confidence >= 0.8) return JarvisTheme.green;
    if (confidence >= 0.5) return JarvisTheme.orange;
    return JarvisTheme.red;
  }
}
