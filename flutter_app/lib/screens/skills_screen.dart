import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/skills_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
import 'package:jarvis_ui/widgets/neon_glow.dart';
import 'package:jarvis_ui/widgets/jarvis_chip.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_search_bar.dart';
import 'package:jarvis_ui/widgets/shimmer_loading.dart';
import 'package:jarvis_ui/widgets/staggered_list.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';
import 'package:jarvis_ui/widgets/jarvis_tab_bar.dart';

class SkillsScreen extends StatefulWidget {
  const SkillsScreen({super.key});

  @override
  State<SkillsScreen> createState() => _SkillsScreenState();
}

class _SkillsScreenState extends State<SkillsScreen> {
  int _tabIndex = 0;
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    final provider = context.read<SkillsProvider>();
    final api = context.read<ConnectionProvider>().api;
    provider.setApi(api);
    provider.loadFeatured();
    provider.loadTrending();
    provider.loadInstalled();
  }

  void _onSearch(String query) {
    setState(() => _searchQuery = query);
    if (query.isNotEmpty) {
      context.read<SkillsProvider>().search(query);
    }
  }

  void _onClearSearch() {
    setState(() => _searchQuery = '');
  }

  List<dynamic> _filterSkills(List<dynamic> skills) {
    if (_searchQuery.isEmpty) return skills;
    final q = _searchQuery.toLowerCase();
    return skills.where((s) {
      final skill = s as Map<String, dynamic>;
      final name = (skill['name']?.toString() ?? '').toLowerCase();
      final desc = (skill['description']?.toString() ?? '').toLowerCase();
      final author = (skill['author']?.toString() ?? '').toLowerCase();
      return name.contains(q) || desc.contains(q) || author.contains(q);
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Consumer<SkillsProvider>(
      builder: (context, provider, _) {
        return Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: JarvisSearchBar(
                hintText: l.searchSkills,
                onChanged: _onSearch,
                onClear: _onClearSearch,
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: JarvisTabBar(
                tabs: [l.featured, l.trending, l.installed],
                icons: const [
                  Icons.star_outline,
                  Icons.trending_up,
                  Icons.check_circle_outline,
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

  Widget _buildTabContent(SkillsProvider provider, AppLocalizations l) {
    if (provider.isLoading) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: ShimmerLoading(count: 6, height: 120),
        ),
      );
    }

    if (provider.error != null) {
      return JarvisEmptyState(
        icon: Icons.error_outline,
        title: l.noSkills,
        subtitle: provider.error,
        action: ElevatedButton.icon(
          onPressed: _retryLoad,
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    return switch (_tabIndex) {
      0 => _buildSkillGrid(
          _searchQuery.isNotEmpty
              ? _filterSkills(provider.searchResults.isNotEmpty
                  ? provider.searchResults
                  : provider.featured)
              : provider.featured,
          l,
          isInstalled: false,
        ),
      1 => _buildSkillGrid(
          _filterSkills(provider.trending),
          l,
          isInstalled: false,
        ),
      2 => _buildInstalledList(
          _filterSkills(provider.installed),
          l,
        ),
      _ => const SizedBox.shrink(),
    };
  }

  void _retryLoad() {
    final provider = context.read<SkillsProvider>();
    provider.loadFeatured();
    provider.loadTrending();
    provider.loadInstalled();
  }

  Widget _buildSkillGrid(
    List<dynamic> skills,
    AppLocalizations l, {
    required bool isInstalled,
  }) {
    if (skills.isEmpty) {
      return JarvisEmptyState(
        icon: Icons.extension_outlined,
        title: l.noSkills,
        subtitle: l.browseMarketplace,
      );
    }

    return RefreshIndicator(
      onRefresh: () async {
        final provider = context.read<SkillsProvider>();
        if (_tabIndex == 0) {
          await provider.loadFeatured();
        } else {
          await provider.loadTrending();
        }
      },
      color: JarvisTheme.accent,
      child: GridView.builder(
        padding: const EdgeInsets.all(16),
        gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
          maxCrossAxisExtent: 400,
          mainAxisExtent: 220,
          crossAxisSpacing: 12,
          mainAxisSpacing: 12,
        ),
        itemCount: skills.length,
        itemBuilder: (context, index) {
          final skill = skills[index] as Map<String, dynamic>;
          return _SkillCard(
            skill: skill,
            isInstalled: isInstalled,
            onInstall: () => _installSkill(skill),
          );
        },
      ),
    );
  }

  Widget _buildInstalledList(List<dynamic> skills, AppLocalizations l) {
    if (skills.isEmpty) {
      return JarvisEmptyState(
        icon: Icons.extension_off_outlined,
        title: l.noSkills,
        subtitle: l.browseMarketplace,
      );
    }

    return RefreshIndicator(
      onRefresh: () => context.read<SkillsProvider>().loadInstalled(),
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          StaggeredList(
            children: skills.map<Widget>((s) {
              final skill = s as Map<String, dynamic>;
              return _SkillCard(
                skill: skill,
                isInstalled: true,
                onUninstall: () => _uninstallSkill(skill),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Future<void> _installSkill(Map<String, dynamic> skill) async {
    final id = skill['id']?.toString() ?? '';
    if (id.isEmpty) return;
    await context.read<SkillsProvider>().installSkill(id);
  }

  Future<void> _uninstallSkill(Map<String, dynamic> skill) async {
    final id = skill['id']?.toString() ?? '';
    if (id.isEmpty) return;
    final l = AppLocalizations.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.uninstallSkill),
        content: Text(skill['name']?.toString() ?? ''),
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
            child: Text(l.uninstallSkill),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      await context.read<SkillsProvider>().uninstallSkill(id);
    }
  }
}

class _SkillCard extends StatelessWidget {
  const _SkillCard({
    required this.skill,
    required this.isInstalled,
    this.onInstall,
    this.onUninstall,
  });

  final Map<String, dynamic> skill;
  final bool isInstalled;
  final VoidCallback? onInstall;
  final VoidCallback? onUninstall;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final name = skill['name']?.toString() ?? '';
    final author = skill['author']?.toString() ?? '';
    final description = skill['description']?.toString() ?? '';
    final category = skill['category']?.toString() ?? '';
    final rating = (skill['rating'] as num?)?.toDouble() ?? 0.0;
    final downloadCount = skill['downloads']?.toString() ?? '0';
    final isVerified = skill['verified'] as bool? ?? false;

    return NeonCard(
      tint: JarvisTheme.sectionSkills,
      glowOnHover: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title row
          Row(
            children: [
              Expanded(
                child: Text(
                  name,
                  style: theme.textTheme.bodyLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (isVerified)
                JarvisStatusBadge(
                  label: l.verified,
                  color: JarvisTheme.green,
                  icon: Icons.verified,
                ),
            ],
          ),
          const SizedBox(height: 2),

          // Author
          Text(
            author,
            style: theme.textTheme.bodySmall,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 8),

          // Description
          Expanded(
            child: Text(
              description,
              style: theme.textTheme.bodyMedium,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(height: 8),

          // Metadata row
          Row(
            children: [
              if (category.isNotEmpty) ...[
                JarvisChip(label: category),
                const SizedBox(width: 8),
              ],
              if (rating > 0) ...[
                Icon(Icons.star, size: 14, color: JarvisTheme.orange),
                const SizedBox(width: 2),
                Text(
                  rating.toStringAsFixed(1),
                  style: theme.textTheme.bodySmall,
                ),
                const SizedBox(width: 8),
              ],
              Icon(Icons.download, size: 14, color: JarvisTheme.textSecondary),
              const SizedBox(width: 2),
              Text(downloadCount, style: theme.textTheme.bodySmall),
              const Spacer(),
              if (isInstalled)
                SizedBox(
                  height: 30,
                  child: OutlinedButton(
                    onPressed: onUninstall,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: JarvisTheme.red,
                      side: BorderSide(color: JarvisTheme.red),
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      textStyle: const TextStyle(fontSize: 12),
                    ),
                    child: Text(l.uninstallSkill),
                  ),
                )
              else
                NeonGlow(
                  color: JarvisTheme.sectionSkills,
                  intensity: 0.2,
                  blurRadius: 8,
                  child: SizedBox(
                    height: 30,
                    child: ElevatedButton(
                      onPressed: onInstall,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        textStyle: const TextStyle(fontSize: 12),
                      ),
                      child: Text(l.installSkill),
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }
}
