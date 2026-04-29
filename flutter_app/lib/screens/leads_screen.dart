import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/data/known_packs.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';
import 'package:cognithor_ui/widgets/leads/lead_card.dart';
import 'package:cognithor_ui/widgets/leads/lead_detail_sheet.dart';
import 'package:cognithor_ui/widgets/leads/lead_wizard.dart';
import 'package:cognithor_ui/widgets/packs/locked_pack_card.dart';
import 'package:cognithor_ui/widgets/packs/pack_preview_overlay.dart';

class LeadsScreen extends StatefulWidget {
  const LeadsScreen({super.key});

  @override
  State<LeadsScreen> createState() => _LeadsScreenState();
}

class _LeadsScreenState extends State<LeadsScreen> {
  bool _initialized = false;
  String? _activeSourceFilter;

  // Source chips shown in the filter bar (sourceId -> display label)
  static const List<(String, String)> _sourceChips = [
    ('', 'All'),
    ('reddit', 'Reddit'),
    ('hn', 'HN'),
    ('discord', 'Discord'),
    ('rss', 'RSS'),
  ];

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      final conn = context.read<ConnectionProvider>();
      if (conn.state == CognithorConnectionState.connected) {
        context.read<RedditLeadsProvider>().init(conn.api);
      }
    }
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<SourcesProvider>().refresh();
    });
  }

  void _openDetail(RedditLead lead) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (_) => LeadDetailSheet(lead: lead),
    );
  }

  Future<void> _scanNow() async {
    await context.read<RedditLeadsProvider>().scanNow();
  }

  void _openWizard() {
    final provider = context.read<RedditLeadsProvider>();
    final newLeads = provider.leads.where((l) => l.status == 'new').toList()
      ..sort((a, b) => b.intentScore.compareTo(a.intentScore));
    if (newLeads.isEmpty) return;
    Navigator.of(context)
        .push(
          MaterialPageRoute<void>(builder: (_) => LeadWizard(leads: newLeads)),
        )
        .then((_) => provider.fetchLeads());
  }

  void _archiveLead(String id) {
    context.read<RedditLeadsProvider>().updateLead(id, status: 'archived');
  }

  void _replyLead(String id) {
    context.read<RedditLeadsProvider>().replyToLead(id);
  }

  Widget _buildUpsellSection(SourcesProvider sources) {
    final installed = sources.sources.map((s) => s.sourceId).toSet();
    final lockedPacks = kKnownPacks
        .where((p) => !installed.contains(p.sourceId))
        .toList();
    if (lockedPacks.isEmpty) return const SizedBox.shrink();

    final paidLocked = lockedPacks
        .where((p) => p.listPriceBadge != null)
        .toList();
    final freeLocked = lockedPacks
        .where((p) => p.listPriceBadge == null)
        .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (paidLocked.isNotEmpty) ...[
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Text(
              'Premium packs available',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          ...paidLocked.map(
            (p) => Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: SizedBox(
                height: 220,
                child: PackPreviewOverlay(
                  pack: p,
                  child: _buildFakeLeadPreview(p),
                ),
              ),
            ),
          ),
        ],
        if (freeLocked.isNotEmpty) ...[
          Padding(
            padding: const EdgeInsets.only(bottom: 12, top: 8),
            child: Text(
              'Free sources (not yet configured)',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          ...freeLocked.map(
            (p) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: LockedPackCard(pack: p),
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildFakeLeadPreview(KnownPack pack) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(pack.icon, color: pack.accentColor, size: 20),
            const SizedBox(width: 8),
            Text(
              pack.displayName,
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        _fakeLeadRow(
          'Looking for a local LLM alternative to...',
          92,
          pack.accentColor,
        ),
        _fakeLeadRow(
          'Anyone tried self-hosted AI agents for...',
          87,
          pack.accentColor,
        ),
        _fakeLeadRow(
          'Switching from OpenAI to local — need...',
          78,
          pack.accentColor,
        ),
      ],
    );
  }

  Widget _fakeLeadRow(String title, int score, Color accent) {
    final theme = Theme.of(context);
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(
          color: theme.colorScheme.outline.withValues(alpha: 0.1),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: theme.textTheme.bodySmall,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: accent.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              '$score',
              style: TextStyle(
                color: accent,
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final sources = context.watch<SourcesProvider>();

    return Consumer<RedditLeadsProvider>(
      builder: (context, provider, _) {
        return Scaffold(
          body: Column(
            children: [
              // Stats bar
              _StatsBar(provider: provider, onProcessQueue: _openWizard),
              // Source filter chips
              _SourceFilterBar(
                sourceChips: _sourceChips,
                activeFilter: _activeSourceFilter,
                onFilterChanged: (v) => setState(() => _activeSourceFilter = v),
              ),
              // Status filter row
              _FilterRow(provider: provider),
              // Upsell section + lead list
              Expanded(
                child: provider.loading && provider.leads.isEmpty
                    ? const Center(child: CircularProgressIndicator())
                    : RefreshIndicator(
                        onRefresh: () async {
                          await Future.wait([
                            provider.fetchLeads(),
                            sources.refresh(),
                          ]);
                        },
                        // Migrated from `ListView(children: [...provider.leads.map(...)])`
                        // to a CustomScrollView so the dynamic lead list renders lazily
                        // via SliverList.builder. The fixed upsell + empty-state cards
                        // stay as SliverToBoxAdapters above the lazy list.
                        child: CustomScrollView(
                          slivers: [
                            SliverPadding(
                              padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                              sliver: SliverToBoxAdapter(
                                child: _buildUpsellSection(sources),
                              ),
                            ),
                            if (provider.leads.isEmpty)
                              SliverPadding(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 8,
                                ),
                                sliver: SliverToBoxAdapter(
                                  child: CognithorEmptyState(
                                    icon: Icons.track_changes,
                                    title: l.noLeadsFound,
                                    subtitle: l.noLeadsHint,
                                  ),
                                ),
                              )
                            else
                              SliverPadding(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 8,
                                ),
                                sliver: SliverList.builder(
                                  itemCount: provider.leads.length,
                                  itemBuilder: (context, index) {
                                    final lead = provider.leads[index];
                                    return LeadCard(
                                      lead: lead,
                                      onTap: () => _openDetail(lead),
                                      onReply: () => _replyLead(lead.id),
                                      onArchive: () => _archiveLead(lead.id),
                                    );
                                  },
                                ),
                              ),
                          ],
                        ),
                      ),
              ),
            ],
          ),
          floatingActionButton: FloatingActionButton.extended(
            onPressed: provider.scanning ? null : _scanNow,
            icon: provider.scanning
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.radar),
            label: Text(provider.scanning ? l.scanning : l.scanNow),
            backgroundColor: CognithorTheme.accent,
          ),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------

class _SourceFilterBar extends StatelessWidget {
  const _SourceFilterBar({
    required this.sourceChips,
    required this.activeFilter,
    required this.onFilterChanged,
  });

  final List<(String, String)> sourceChips;
  final String? activeFilter;
  final ValueChanged<String?> onFilterChanged;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 44,
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        children: [
          for (final (id, label) in sourceChips) ...[
            ChoiceChip(
              label: Text(label),
              selected: (activeFilter ?? '') == id,
              onSelected: (_) => onFilterChanged(id.isEmpty ? null : id),
              visualDensity: VisualDensity.compact,
            ),
            const SizedBox(width: 8),
          ],
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _StatsBar extends StatelessWidget {
  const _StatsBar({required this.provider, this.onProcessQueue});
  final RedditLeadsProvider provider;
  final VoidCallback? onProcessQueue;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Wrap(
        spacing: 12,
        runSpacing: 8,
        children: [
          CognithorStat(
            label: 'New',
            value: '${provider.newCount}',
            icon: Icons.fiber_new,
            color: CognithorTheme.accent,
          ),
          CognithorStat(
            label: 'Reviewed',
            value: '${provider.reviewedCount}',
            icon: Icons.check_circle_outline,
            color: Colors.orange,
          ),
          CognithorStat(
            label: 'Replied',
            value: '${provider.repliedCount}',
            icon: Icons.reply,
            color: CognithorTheme.green,
          ),
          if (provider.newCount > 0 && onProcessQueue != null)
            ElevatedButton.icon(
              onPressed: onProcessQueue,
              icon: const Icon(Icons.playlist_play, size: 18),
              label: Text(l.processQueue),
              style: ElevatedButton.styleFrom(
                backgroundColor: CognithorTheme.accent,
              ),
            ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _FilterRow extends StatelessWidget {
  const _FilterRow({required this.provider});
  final RedditLeadsProvider provider;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Row(
        children: [
          SegmentedButton<String>(
            segments: [
              ButtonSegment(value: '', label: Text(l.filterAll)),
              ButtonSegment(value: 'new', label: Text(l.leadNew)),
              ButtonSegment(value: 'reviewed', label: Text(l.leadReviewed)),
              ButtonSegment(value: 'replied', label: Text(l.leadReplied)),
            ],
            selected: {provider.statusFilter},
            onSelectionChanged: (s) => provider.setStatusFilter(s.first),
            style: ButtonStyle(
              visualDensity: VisualDensity.compact,
              textStyle: WidgetStatePropertyAll(
                Theme.of(context).textTheme.labelSmall,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
