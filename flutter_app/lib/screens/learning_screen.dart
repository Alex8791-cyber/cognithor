import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/cognithor_toggle_field.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/neon_glow.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';
import 'package:cognithor_ui/widgets/cognithor_status_badge.dart';
import 'package:cognithor_ui/widgets/cognithor_tab_bar.dart';

class LearningScreen extends StatefulWidget {
  const LearningScreen({super.key});

  @override
  State<LearningScreen> createState() => _LearningScreenState();
}

class _LearningScreenState extends State<LearningScreen> {
  int _tabIndex = 0;

  // Overview data
  Map<String, dynamic>? _stats;
  List<Map<String, dynamic>> _confidenceHistory = [];

  // Knowledge gaps data
  List<Map<String, dynamic>> _gaps = [];

  // Exploration queue data
  List<Map<String, dynamic>> _queue = [];

  // Watch directories data
  List<Map<String, dynamic>> _directories = [];

  // Q&A data
  List<Map<String, dynamic>> _qaPairs = [];
  String _qaSearch = '';

  // Lineage data
  List<Map<String, dynamic>> _lineageEntries = [];
  String _lineageFilter = '';

  bool _loading = true;
  String? _error;
  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _loadAll();
    }
  }

  Future<void> _loadAll() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final results = await Future.wait([
        api.getLearningStats(),
        api.getLearningGaps(),
        api.getConfidenceHistory(),
        api.getLearningQueue(),
        api.getLearningDirectories(),
        api.getQAPairs(),
        api.getRecentLineage(),
      ]);

      if (!mounted) return;

      final statsResult = results[0];
      final gapsResult = results[1];
      final historyResult = results[2];
      final queueResult = results[3];
      final dirsResult = results[4];
      final qaResult = results[5];
      final lineageResult = results[6];

      setState(() {
        _stats = statsResult.containsKey('error') ? null : statsResult;
        _gaps = _parseList(gapsResult, 'gaps');
        _confidenceHistory = _parseList(historyResult, 'history');
        _queue = _parseList(queueResult, 'tasks');
        _directories = _parseList(dirsResult, 'directories');
        _qaPairs = _parseList(qaResult, 'pairs');
        _lineageEntries = _parseList(lineageResult, 'entries');
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  List<Map<String, dynamic>> _parseList(Map<String, dynamic> data, String key) {
    return (data[key] as List?)
            ?.map((e) => e as Map<String, dynamic>)
            .toList() ??
        [];
  }

  Future<void> _dismissGap(String gapId) async {
    try {
      final api = context.read<ConnectionProvider>().api;
      await api.dismissGap(gapId);
      setState(() {
        _gaps.removeWhere((g) => g['id']?.toString() == gapId);
      });
    } catch (_) {}
  }

  Future<void> _exploreGap(String gapId) async {
    try {
      final api = context.read<ConnectionProvider>().api;
      await api.triggerExploration(gapId);
      await _loadAll();
    } catch (_) {}
  }

  Future<void> _toggleDirectory(int index, bool enabled) async {
    final dir = Map<String, dynamic>.from(_directories[index]);
    dir['enabled'] = enabled;
    setState(() => _directories[index] = dir);
    try {
      final api = context.read<ConnectionProvider>().api;
      await api.updateLearningDirectories(_directories);
    } catch (_) {
      // Revert on failure.
      dir['enabled'] = !enabled;
      if (mounted) setState(() => _directories[index] = dir);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l.learningTitle),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadAll,
            tooltip: l.refresh,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(_error!, style: TextStyle(color: CognithorTheme.red)),
                  const SizedBox(height: 12),
                  ElevatedButton(onPressed: _loadAll, child: Text(l.retry)),
                ],
              ),
            )
          : Column(
              children: [
                CognithorTabBar(
                  tabs: [
                    l.dashboard,
                    l.knowledgeGaps,
                    l.explorationQueue,
                    l.qaKnowledgeBase,
                    l.lineage,
                  ],
                  icons: const [
                    Icons.dashboard,
                    Icons.help_outline,
                    Icons.explore,
                    Icons.quiz,
                    Icons.account_tree,
                  ],
                  selectedIndex: _tabIndex,
                  onChanged: (i) => setState(() => _tabIndex = i),
                ),
                Expanded(
                  child: switch (_tabIndex) {
                    0 => _buildOverview(l),
                    1 => _buildGaps(l),
                    2 => _buildQueue(l),
                    3 => _buildQA(l),
                    4 => _buildLineage(l),
                    _ => const SizedBox.shrink(),
                  },
                ),
              ],
            ),
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 1: Overview
  // ---------------------------------------------------------------------------

  Widget _buildOverview(AppLocalizations l) {
    final theme = Theme.of(context);

    return ListView(
      padding: const EdgeInsets.all(CognithorTheme.spacing),
      children: [
        // Stats row
        Wrap(
          spacing: CognithorTheme.spacingSm,
          runSpacing: CognithorTheme.spacingSm,
          children: [
            CognithorStat(
              label: l.filesProcessed,
              value: '${_stats?['files_processed'] ?? 0}',
              icon: Icons.insert_drive_file,
              color: CognithorTheme.accent,
            ),
            CognithorStat(
              label: l.entitiesCreated,
              value: '${_stats?['entities_created'] ?? 0}',
              icon: Icons.bubble_chart,
              color: CognithorTheme.green,
            ),
            CognithorStat(
              label: l.confidenceUpdates,
              value: '${_stats?['confidence_updates'] ?? 0}',
              icon: Icons.trending_up,
              color: CognithorTheme.orange,
            ),
            CognithorStat(
              label: l.openGaps,
              value: '${_stats?['open_gaps'] ?? _gaps.length}',
              icon: Icons.help_outline,
              color: CognithorTheme.red,
            ),
          ],
        ),
        const SizedBox(height: CognithorTheme.spacingLg),

        // Learning activity chart
        NeonCard(
          tint: CognithorTheme.sectionDashboard,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(
                    Icons.show_chart,
                    size: 18,
                    color: CognithorTheme.sectionDashboard,
                  ),
                  const SizedBox(width: 8),
                  Text(l.learningTitle, style: theme.textTheme.titleMedium),
                ],
              ),
              const SizedBox(height: 12),
              SizedBox(height: 200, child: _buildActivityChart(theme)),
            ],
          ),
        ),
        const SizedBox(height: 12),

        // Confidence history
        NeonCard(
          tint: CognithorTheme.sectionDashboard,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(
                    Icons.history,
                    size: 18,
                    color: CognithorTheme.sectionDashboard,
                  ),
                  const SizedBox(width: 8),
                  Text(l.confidenceHistory, style: theme.textTheme.titleMedium),
                ],
              ),
              const SizedBox(height: 8),
              _confidenceHistory.isEmpty
                  ? Padding(
                      padding: const EdgeInsets.all(CognithorTheme.spacing),
                      child: Text(l.noData, style: theme.textTheme.bodySmall),
                    )
                  : Column(
                      children: _confidenceHistory.take(10).map((entry) {
                        final entity =
                            (entry['entity'] ?? entry['entity_id'] ?? '')
                                .toString();
                        final oldConf = (entry['old_confidence'] ?? 0)
                            .toString();
                        final newConf = (entry['new_confidence'] ?? 0)
                            .toString();
                        final increased =
                            (entry['new_confidence'] ?? 0) >
                            (entry['old_confidence'] ?? 0);

                        return ListTile(
                          dense: true,
                          leading: Icon(
                            increased
                                ? Icons.arrow_upward
                                : Icons.arrow_downward,
                            color: increased
                                ? CognithorTheme.green
                                : CognithorTheme.red,
                            size: 18,
                          ),
                          title: Text(
                            entity,
                            style: theme.textTheme.bodyMedium,
                          ),
                          subtitle: Text(
                            '$oldConf -> $newConf',
                            style: theme.textTheme.bodySmall,
                          ),
                          trailing: Text(
                            (entry['timestamp'] ?? '').toString(),
                            style: theme.textTheme.bodySmall,
                          ),
                        );
                      }).toList(),
                    ),
            ],
          ),
        ),
        const SizedBox(height: 12),

        // Watch directories
        NeonCard(
          tint: CognithorTheme.sectionDashboard,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(
                    Icons.folder_open,
                    size: 18,
                    color: CognithorTheme.sectionDashboard,
                  ),
                  const SizedBox(width: 8),
                  Text(l.watchDirectories, style: theme.textTheme.titleMedium),
                ],
              ),
              const SizedBox(height: 8),
              if (_directories.isEmpty)
                Padding(
                  padding: const EdgeInsets.all(CognithorTheme.spacing),
                  child: Text(l.noData, style: theme.textTheme.bodySmall),
                )
              else
                ...List.generate(
                  _directories.length,
                  (i) => _buildDirectoryRow(theme, l, i),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildDirectoryRow(ThemeData theme, AppLocalizations l, int index) {
    final dir = _directories[index];
    final path = (dir['path'] ?? '').toString();
    final enabled = dir['enabled'] == true;
    final exists = dir['exists'] == true;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: CognithorTheme.spacingSm),
      child: Row(
        children: [
          Tooltip(
            message: exists ? l.directoryExists : l.directoryMissing,
            child: Icon(
              exists ? Icons.check_circle : Icons.error_outline,
              color: exists ? CognithorTheme.green : CognithorTheme.red,
              size: 18,
            ),
          ),
          const SizedBox(width: CognithorTheme.spacingSm),
          Expanded(
            child: CognithorToggleField(
              label: path,
              value: enabled,
              onChanged: (v) => _toggleDirectory(index, v),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActivityChart(ThemeData theme) {
    // Build spots from stats or use placeholder structure
    final dataPoints =
        (_stats?['activity_chart'] as List?)
            ?.whereType<Map<String, dynamic>>()
            .toList() ??
        [];

    if (dataPoints.isEmpty) {
      return Center(
        child: Text(
          AppLocalizations.of(context).noData,
          style: theme.textTheme.bodySmall,
        ),
      );
    }

    final spots = <FlSpot>[];
    for (var i = 0; i < dataPoints.length; i++) {
      final y = (dataPoints[i]['count'] ?? 0).toDouble();
      spots.add(FlSpot(i.toDouble(), y));
    }

    return LineChart(
      LineChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 1,
          getDrawingHorizontalLine: (_) =>
              FlLine(color: theme.dividerColor, strokeWidth: 0.5),
        ),
        titlesData: const FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(showTitles: true, reservedSize: 32),
          ),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: CognithorTheme.accent,
            barWidth: 2,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: CognithorTheme.accent.withValues(alpha: 0.1),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 2: Knowledge Gaps
  // ---------------------------------------------------------------------------

  Widget _buildGaps(AppLocalizations l) {
    if (_gaps.isEmpty) {
      return CognithorEmptyState(
        icon: Icons.check_circle_outline,
        title: l.noGaps,
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(CognithorTheme.spacing),
      itemCount: _gaps.length,
      itemBuilder: (context, index) {
        final gap = _gaps[index];
        return _GapCard(
          gap: gap,
          l: l,
          onDismiss: () => _dismissGap((gap['id'] ?? '').toString()),
          onExplore: () => _exploreGap((gap['id'] ?? '').toString()),
        );
      },
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 3: Exploration Queue
  // ---------------------------------------------------------------------------

  Widget _buildQueue(AppLocalizations l) {
    if (_queue.isEmpty) {
      return CognithorEmptyState(icon: Icons.explore_off, title: l.noTasks);
    }

    final theme = Theme.of(context);

    return ListView.builder(
      padding: const EdgeInsets.all(CognithorTheme.spacing),
      itemCount: _queue.length,
      itemBuilder: (context, index) {
        final task = _queue[index];
        final query = (task['query'] ?? '').toString();
        final sources =
            (task['sources'] as List?)?.map((s) => s.toString()).toList() ?? [];
        final priority = (task['priority'] ?? 'normal').toString();
        final status = (task['status'] ?? 'pending').toString();

        final priorityColor = switch (priority) {
          'high' => CognithorTheme.red,
          'medium' => CognithorTheme.orange,
          _ => CognithorTheme.green,
        };

        final statusColor = switch (status) {
          'running' => CognithorTheme.accent,
          'completed' => CognithorTheme.green,
          'failed' => CognithorTheme.red,
          _ => CognithorTheme.textSecondary,
        };

        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: NeonCard(
            tint: CognithorTheme.sectionDashboard,
            glowOnHover: true,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        query,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    CognithorStatusBadge(label: status, color: statusColor),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    CognithorStatusBadge(
                      label: '${l.priority}: $priority',
                      color: priorityColor,
                      icon: Icons.flag,
                    ),
                    if (sources.isNotEmpty) ...[
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          sources.join(', '),
                          style: theme.textTheme.bodySmall,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }
  // ---------------------------------------------------------------------------
  // Tab 4: Q&A Knowledge Base
  // ---------------------------------------------------------------------------

  Future<void> _searchQA(String query) async {
    setState(() => _qaSearch = query);
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.getQAPairs(query: query.isEmpty ? null : query);
      if (!mounted) return;
      setState(() => _qaPairs = _parseList(result, 'pairs'));
    } catch (_) {}
  }

  Future<void> _verifyQA(String id) async {
    try {
      final api = context.read<ConnectionProvider>().api;
      await api.verifyQA(id);
      await _searchQA(_qaSearch);
    } catch (_) {}
  }

  Future<void> _deleteQA(String id) async {
    try {
      final api = context.read<ConnectionProvider>().api;
      await api.deleteQA(id);
      setState(() => _qaPairs.removeWhere((p) => p['id']?.toString() == id));
    } catch (_) {}
  }

  Future<void> _showAddQADialog(AppLocalizations l) async {
    final questionCtrl = TextEditingController();
    final answerCtrl = TextEditingController();
    final topicCtrl = TextEditingController();

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.addQA),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: questionCtrl,
                decoration: InputDecoration(labelText: l.question),
                maxLines: 2,
              ),
              const SizedBox(height: 12),
              TextField(
                controller: answerCtrl,
                decoration: InputDecoration(labelText: l.answer),
                maxLines: 4,
              ),
              const SizedBox(height: 12),
              TextField(
                controller: topicCtrl,
                decoration: InputDecoration(labelText: l.topic),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(l.save),
          ),
        ],
      ),
    );

    if (confirmed == true && questionCtrl.text.isNotEmpty && mounted) {
      try {
        final api = context.read<ConnectionProvider>().api;
        await api.addQAPair({
          'question': questionCtrl.text,
          'answer': answerCtrl.text,
          'topic': topicCtrl.text,
        });
        await _searchQA(_qaSearch);
      } catch (_) {}
    }

    questionCtrl.dispose();
    answerCtrl.dispose();
    topicCtrl.dispose();
  }

  Widget _buildQA(AppLocalizations l) {
    final theme = Theme.of(context);

    return Stack(
      children: [
        Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(CognithorTheme.spacing),
              child: TextField(
                decoration: InputDecoration(
                  hintText: l.search,
                  prefixIcon: const Icon(Icons.search),
                  border: const OutlineInputBorder(),
                  isDense: true,
                ),
                onChanged: _searchQA,
              ),
            ),
            Expanded(
              child: _qaPairs.isEmpty
                  ? CognithorEmptyState(
                      icon: Icons.quiz_outlined,
                      title: l.noQAPairs,
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(
                        horizontal: CognithorTheme.spacing,
                      ),
                      itemCount: _qaPairs.length,
                      itemBuilder: (context, index) {
                        final qa = _qaPairs[index];
                        final question = (qa['question'] ?? '').toString();
                        final answer = (qa['answer'] ?? '').toString();
                        final topic = (qa['topic'] ?? '').toString();
                        final conf = ((qa['confidence'] ?? 0) as num)
                            .toDouble();
                        final source = (qa['source'] ?? '').toString();
                        final isVerified = qa['verified'] == true;
                        final id = (qa['id'] ?? '').toString();

                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: NeonCard(
                            tint: CognithorTheme.sectionDashboard,
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  question,
                                  style: theme.textTheme.bodyMedium?.copyWith(
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(answer, style: theme.textTheme.bodySmall),
                                const SizedBox(height: 8),
                                Row(
                                  children: [
                                    if (topic.isNotEmpty)
                                      CognithorStatusBadge(
                                        label: topic,
                                        color: CognithorTheme.accent,
                                      ),
                                    if (isVerified) ...[
                                      const SizedBox(width: 8),
                                      CognithorStatusBadge(
                                        label: l.verified,
                                        color: CognithorTheme.green,
                                        icon: Icons.check,
                                      ),
                                    ],
                                    const Spacer(),
                                    if (source.isNotEmpty)
                                      Text(
                                        '${l.source}: $source',
                                        style: theme.textTheme.bodySmall,
                                      ),
                                  ],
                                ),
                                const SizedBox(height: 6),
                                // Confidence bar
                                Row(
                                  children: [
                                    SizedBox(
                                      width: 80,
                                      child: Text(
                                        l.confidence,
                                        style: theme.textTheme.bodySmall,
                                      ),
                                    ),
                                    Expanded(
                                      child: ClipRRect(
                                        borderRadius: BorderRadius.circular(4),
                                        child: LinearProgressIndicator(
                                          value: conf.clamp(0.0, 1.0),
                                          backgroundColor: theme.dividerColor,
                                          color: CognithorTheme.accent,
                                          minHeight: 6,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      '${(conf * 100).toInt()}%',
                                      style: theme.textTheme.bodySmall,
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.end,
                                  children: [
                                    if (!isVerified)
                                      TextButton.icon(
                                        onPressed: () => _verifyQA(id),
                                        icon: const Icon(
                                          Icons.thumb_up,
                                          size: 16,
                                        ),
                                        label: Text(l.verify),
                                      ),
                                    const SizedBox(width: 8),
                                    TextButton.icon(
                                      onPressed: () => _deleteQA(id),
                                      icon: const Icon(Icons.delete, size: 16),
                                      label: Text(l.delete),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
        Positioned(
          bottom: CognithorTheme.spacing,
          right: CognithorTheme.spacing,
          child: FloatingActionButton(
            onPressed: () => _showAddQADialog(l),
            child: const Icon(Icons.add),
          ),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 5: Lineage
  // ---------------------------------------------------------------------------

  Future<void> _loadLineage({String? entityId}) async {
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = entityId != null && entityId.isNotEmpty
          ? await api.getEntityLineage(entityId)
          : await api.getRecentLineage();
      if (!mounted) return;
      setState(() {
        _lineageEntries = _parseList(result, 'entries');
        _lineageFilter = entityId ?? '';
      });
    } catch (_) {}
  }

  IconData _lineageActionIcon(String action) {
    return switch (action) {
      'created' => Icons.add_circle_outline,
      'updated' => Icons.edit,
      'decayed' => Icons.trending_down,
      _ => Icons.circle_outlined,
    };
  }

  Color _lineageActionColor(String action) {
    return switch (action) {
      'created' => CognithorTheme.green,
      'updated' => CognithorTheme.accent,
      'decayed' => CognithorTheme.red,
      _ => CognithorTheme.textSecondary,
    };
  }

  IconData _sourceTypeIcon(String sourceType) {
    return switch (sourceType) {
      'file' => Icons.insert_drive_file,
      'web' => Icons.language,
      'conversation' => Icons.chat,
      'exploration' => Icons.explore,
      _ => Icons.source,
    };
  }

  Widget _buildLineage(AppLocalizations l) {
    final theme = Theme.of(context);

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(CognithorTheme.spacing),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    hintText: l.entityLineage,
                    prefixIcon: const Icon(Icons.filter_alt),
                    border: const OutlineInputBorder(),
                    isDense: true,
                  ),
                  onSubmitted: (value) =>
                      _loadLineage(entityId: value.isEmpty ? null : value),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton.icon(
                onPressed: () async {
                  final api = context.read<ConnectionProvider>().api;
                  await api.triggerExplorationBatch();
                  if (mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(l.explorationComplete)),
                    );
                    _loadLineage(
                      entityId: _lineageFilter.isEmpty ? null : _lineageFilter,
                    );
                  }
                },
                icon: const Icon(Icons.play_arrow, size: 16),
                label: Text(l.runExploration),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(
            horizontal: CognithorTheme.spacing,
          ),
          child: Align(
            alignment: Alignment.centerLeft,
            child: Text(
              _lineageFilter.isEmpty
                  ? l.recentChanges
                  : '${l.entityLineage}: $_lineageFilter',
              style: theme.textTheme.titleSmall,
            ),
          ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: _lineageEntries.isEmpty
              ? CognithorEmptyState(
                  icon: Icons.account_tree_outlined,
                  title: l.noLineage,
                )
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(
                    horizontal: CognithorTheme.spacing,
                  ),
                  itemCount: _lineageEntries.length,
                  itemBuilder: (context, index) {
                    final entry = _lineageEntries[index];
                    final entity = (entry['entity'] ?? entry['entity_id'] ?? '')
                        .toString();
                    final action = (entry['action'] ?? '').toString();
                    final sourceType = (entry['source_type'] ?? '').toString();
                    final ts = (entry['timestamp'] ?? '').toString();

                    final actionLabel = switch (action) {
                      'created' => l.created,
                      'updated' => l.updated,
                      'decayed' => l.decayed,
                      _ => action,
                    };

                    return Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Timeline dot + line
                          Column(
                            children: [
                              Icon(
                                _lineageActionIcon(action),
                                color: _lineageActionColor(action),
                                size: 20,
                              ),
                              if (index < _lineageEntries.length - 1)
                                Container(
                                  width: 2,
                                  height: 32,
                                  color: theme.dividerColor,
                                ),
                            ],
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: NeonCard(
                              tint: CognithorTheme.sectionDashboard,
                              child: Row(
                                children: [
                                  Icon(
                                    _sourceTypeIcon(sourceType),
                                    size: 16,
                                    color: CognithorTheme.textSecondary,
                                  ),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          entity,
                                          style: theme.textTheme.bodyMedium
                                              ?.copyWith(
                                                fontWeight: FontWeight.w600,
                                              ),
                                        ),
                                        Text(
                                          actionLabel,
                                          style: TextStyle(
                                            color: _lineageActionColor(action),
                                            fontSize: 12,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Text(ts, style: theme.textTheme.bodySmall),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Gap card widget
// -----------------------------------------------------------------------------

class _GapCard extends StatelessWidget {
  const _GapCard({
    required this.gap,
    required this.l,
    required this.onDismiss,
    required this.onExplore,
  });

  final Map<String, dynamic> gap;
  final AppLocalizations l;
  final VoidCallback onDismiss;
  final VoidCallback onExplore;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final question = (gap['question'] ?? '').toString();
    final topic = (gap['topic'] ?? '').toString();
    final importance = ((gap['importance'] ?? 0) as num).toDouble();
    final curiosityScore = ((gap['curiosity_score'] ?? 0) as num).toDouble();
    final status = (gap['status'] ?? 'open').toString();

    final statusColor = switch (status) {
      'exploring' => CognithorTheme.accent,
      'resolved' => CognithorTheme.green,
      'dismissed' => CognithorTheme.textSecondary,
      _ => CognithorTheme.orange,
    };

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: NeonCard(
        tint: CognithorTheme.sectionDashboard,
        glowOnHover: true,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Question and status
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Text(
                    question,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                CognithorStatusBadge(label: status, color: statusColor),
              ],
            ),
            if (topic.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(topic, style: theme.textTheme.bodySmall),
            ],
            const SizedBox(height: 12),

            // Importance bar
            Row(
              children: [
                SizedBox(
                  width: 80,
                  child: Text(l.importance, style: theme.textTheme.bodySmall),
                ),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: importance.clamp(0.0, 1.0),
                      backgroundColor: theme.dividerColor,
                      color: CognithorTheme.accent,
                      minHeight: 6,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '${(importance * 100).toInt()}%',
                  style: theme.textTheme.bodySmall,
                ),
              ],
            ),
            const SizedBox(height: 6),

            // Curiosity score bar
            Row(
              children: [
                SizedBox(
                  width: 80,
                  child: Text(l.curiosity, style: theme.textTheme.bodySmall),
                ),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: curiosityScore.clamp(0.0, 1.0),
                      backgroundColor: theme.dividerColor,
                      color: CognithorTheme.orange,
                      minHeight: 6,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '${(curiosityScore * 100).toInt()}%',
                  style: theme.textTheme.bodySmall,
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Actions
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton.icon(
                  onPressed: onDismiss,
                  icon: const Icon(Icons.close, size: 16),
                  label: Text(l.dismiss),
                ),
                const SizedBox(width: 8),
                NeonGlow(
                  color: CognithorTheme.sectionDashboard,
                  intensity: 0.2,
                  blurRadius: 8,
                  child: ElevatedButton.icon(
                    onPressed: onExplore,
                    icon: const Icon(Icons.explore, size: 16),
                    label: Text(l.explore),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
