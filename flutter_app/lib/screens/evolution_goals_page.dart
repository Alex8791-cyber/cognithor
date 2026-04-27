import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/evolution_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';

class EvolutionGoalsPage extends StatefulWidget {
  const EvolutionGoalsPage({super.key});

  @override
  State<EvolutionGoalsPage> createState() => _EvolutionGoalsPageState();
}

class _EvolutionGoalsPageState extends State<EvolutionGoalsPage>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final conn = context.read<ConnectionProvider>();
      final evo = context.read<EvolutionProvider>();
      evo.setApi(conn.api);
      evo.fetchAll();
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _createGoal() async {
    final l = AppLocalizations.of(context);
    final titleCtrl = TextEditingController();
    final descCtrl = TextEditingController();
    int priority = 3;

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => AlertDialog(
          title: Text(l.newLearningGoal),
          content: SizedBox(
            width: 400,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: titleCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Goal',
                    border: OutlineInputBorder(),
                  ),
                  autofocus: true,
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: descCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Description',
                    border: OutlineInputBorder(),
                  ),
                  maxLines: 3,
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<int>(
                  initialValue: priority,
                  decoration: const InputDecoration(
                    labelText: 'Priority',
                    border: OutlineInputBorder(),
                  ),
                  items: [1, 2, 3, 4, 5]
                      .map(
                        (p) => DropdownMenuItem(value: p, child: Text('P$p')),
                      )
                      .toList(),
                  onChanged: (v) => setDialogState(() => priority = v ?? 3),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: Text(l.cancel),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: Text(l.create),
            ),
          ],
        ),
      ),
    );

    if (result == true && titleCtrl.text.trim().isNotEmpty && mounted) {
      await context.read<EvolutionProvider>().createGoal(
        title: titleCtrl.text.trim(),
        description: descCtrl.text.trim(),
        priority: priority,
      );
    }
    titleCtrl.dispose();
    descCtrl.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(l.evolutionEngine),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.flag_outlined), text: 'Goals'),
            Tab(icon: Icon(Icons.school_outlined), text: 'Plans'),
            Tab(icon: Icon(Icons.history_edu_outlined), text: 'Journal'),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => context.read<EvolutionProvider>().fetchAll(),
          ),
        ],
      ),
      body: Consumer<EvolutionProvider>(
        builder: (context, evo, _) {
          if (evo.loading && evo.goals.isEmpty) {
            return const Center(child: CircularProgressIndicator());
          }
          return TabBarView(
            controller: _tabController,
            children: [
              _GoalsTab(
                goals: evo.goals,
                stats: evo.stats,
                onUpdate: evo.updateGoal,
                onDelete: evo.deleteGoal,
              ),
              _PlansTab(plans: evo.plans),
              _JournalTab(journal: evo.journal),
            ],
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _createGoal,
        child: const Icon(Icons.add),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Goals tab
// ---------------------------------------------------------------------------

class _GoalsTab extends StatelessWidget {
  final List<EvolutionGoal> goals;
  final Map<String, dynamic> stats;
  final Future<bool> Function(
    String, {
    String? title,
    String? description,
    String? status,
    int? priority,
  })
  onUpdate;
  final Future<bool> Function(String) onDelete;

  const _GoalsTab({
    required this.goals,
    required this.stats,
    required this.onUpdate,
    required this.onDelete,
  });

  static const _statusColors = <String, Color>{
    'active': Colors.blue,
    'paused': Colors.orange,
    'completed': Colors.green,
    'mastered': Colors.green,
    'abandoned': Colors.grey,
  };

  static const _statusIcons = <String, IconData>{
    'active': Icons.play_arrow,
    'paused': Icons.pause,
    'completed': Icons.check_circle,
    'mastered': Icons.star,
    'abandoned': Icons.cancel,
  };

  Future<void> _editGoal(BuildContext context, EvolutionGoal goal) async {
    final titleCtrl = TextEditingController(text: goal.title);
    final descCtrl = TextEditingController(text: goal.description);
    int priority = goal.priority;

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => AlertDialog(
          title: const Text('Lernziel bearbeiten'),
          content: SizedBox(
            width: 400,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: titleCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Goal',
                    border: OutlineInputBorder(),
                  ),
                  autofocus: true,
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: descCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Description',
                    border: OutlineInputBorder(),
                  ),
                  maxLines: 3,
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<int>(
                  initialValue: priority,
                  decoration: const InputDecoration(
                    labelText: 'Priority',
                    border: OutlineInputBorder(),
                  ),
                  items: [1, 2, 3, 4, 5]
                      .map(
                        (p) => DropdownMenuItem(value: p, child: Text('P$p')),
                      )
                      .toList(),
                  onChanged: (v) => setDialogState(() => priority = v ?? 3),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Abbrechen'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Speichern'),
            ),
          ],
        ),
      ),
    );

    if (result == true && titleCtrl.text.trim().isNotEmpty) {
      await onUpdate(
        goal.id,
        title: titleCtrl.text.trim(),
        description: descCtrl.text.trim(),
        priority: priority,
      );
    }
    titleCtrl.dispose();
    descCtrl.dispose();
  }

  Future<void> _confirmDelete(BuildContext context, EvolutionGoal goal) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Lernziel loeschen?'),
        content: Text('"${goal.title}" wirklich loeschen?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Abbrechen'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Loeschen'),
          ),
        ],
      ),
    );
    if (confirm == true) {
      await onDelete(goal.id);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (goals.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.school_outlined,
              size: 64,
              color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
            ),
            const SizedBox(height: 16),
            Text('No learning goals yet.', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(
              'Create one with the + button.',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: goals.length,
      itemBuilder: (context, index) {
        final goal = goals[index];
        final color = _statusColors[goal.status] ?? Colors.grey;
        final icon = _statusIcons[goal.status] ?? Icons.flag;

        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          child: ListTile(
            leading: CircleAvatar(
              backgroundColor: color.withValues(alpha: 0.15),
              child: Icon(icon, color: color, size: 20),
            ),
            title: Text(
              goal.title,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (goal.description.isNotEmpty)
                  Text(
                    goal.description,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Expanded(
                      child: LinearProgressIndicator(
                        value: goal.progress,
                        backgroundColor: color.withValues(alpha: 0.15),
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${(goal.progress * 100).toInt()}%',
                      style: TextStyle(fontSize: 12, color: color),
                    ),
                  ],
                ),
              ],
            ),
            trailing: PopupMenuButton<String>(
              onSelected: (action) {
                switch (action) {
                  case 'edit':
                    _editGoal(context, goal);
                  case 'pause':
                    onUpdate(goal.id, status: 'paused');
                  case 'resume':
                    onUpdate(goal.id, status: 'active');
                  case 'complete':
                    onUpdate(goal.id, status: 'completed');
                  case 'delete':
                    _confirmDelete(context, goal);
                }
              },
              itemBuilder: (_) => [
                const PopupMenuItem(
                  value: 'edit',
                  child: ListTile(
                    leading: Icon(Icons.edit, size: 20),
                    title: Text('Bearbeiten'),
                    dense: true,
                    contentPadding: EdgeInsets.zero,
                  ),
                ),
                if (goal.status == 'active')
                  const PopupMenuItem(value: 'pause', child: Text('Pause')),
                if (goal.status == 'paused')
                  const PopupMenuItem(value: 'resume', child: Text('Resume')),
                if (goal.status != 'completed' && goal.status != 'mastered')
                  const PopupMenuItem(
                    value: 'complete',
                    child: Text('Mark Complete'),
                  ),
                const PopupMenuDivider(),
                const PopupMenuItem(
                  value: 'delete',
                  child: ListTile(
                    leading: Icon(Icons.delete, size: 20, color: Colors.red),
                    title: Text(
                      'Loeschen',
                      style: TextStyle(color: Colors.red),
                    ),
                    dense: true,
                    contentPadding: EdgeInsets.zero,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Plans tab
// ---------------------------------------------------------------------------

class _PlansTab extends StatelessWidget {
  final List<EvolutionPlan> plans;

  const _PlansTab({required this.plans});

  static const _stateColors = <String, Color>{
    'learning': Colors.blue,
    'examining': Colors.orange,
    'mastered': Colors.green,
    'stagnating': Colors.red,
  };

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (plans.isEmpty) {
      return Center(
        child: Text(
          'No learning plans active.',
          style: theme.textTheme.titleMedium,
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: plans.length,
      itemBuilder: (context, index) {
        final plan = plans[index];
        final stateColor = _stateColors[plan.cycleState] ?? Colors.grey;

        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        plan.goal,
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    Chip(
                      label: Text(
                        plan.cycleState.toUpperCase(),
                        style: TextStyle(fontSize: 10, color: stateColor),
                      ),
                      backgroundColor: stateColor.withValues(alpha: 0.15),
                      visualDensity: VisualDensity.compact,
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    _MetricChip(
                      label: 'Sub-Goals',
                      value: '${plan.subGoalsPassed}/${plan.subGoalsTotal}',
                    ),
                    const SizedBox(width: 8),
                    _MetricChip(
                      label: 'Coverage',
                      value: '${(plan.coverageScore * 100).toInt()}%',
                    ),
                    const SizedBox(width: 8),
                    _MetricChip(
                      label: 'Quality',
                      value: '${(plan.qualityScore * 100).toInt()}%',
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(
                  value: plan.completionPercent,
                  backgroundColor: stateColor.withValues(alpha: 0.15),
                  valueColor: AlwaysStoppedAnimation<Color>(stateColor),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _MetricChip extends StatelessWidget {
  final String label;
  final String value;

  const _MetricChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
          Text(
            label,
            style: TextStyle(
              fontSize: 9,
              color: Theme.of(
                context,
              ).colorScheme.onSurface.withValues(alpha: 0.5),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Journal tab
// ---------------------------------------------------------------------------

class _JournalTab extends StatelessWidget {
  final String journal;

  const _JournalTab({required this.journal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (journal.isEmpty) {
      return Center(
        child: Text(
          'No journal entries yet.',
          style: theme.textTheme.titleMedium,
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: SelectableText(
        journal,
        style: theme.textTheme.bodyMedium?.copyWith(
          fontFamily: 'monospace',
          height: 1.5,
        ),
      ),
    );
  }
}
