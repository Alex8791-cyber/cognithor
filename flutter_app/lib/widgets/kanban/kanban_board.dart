import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/widgets/kanban/kanban_column.dart';

class KanbanBoard extends StatelessWidget {
  const KanbanBoard({super.key});

  static const _columnOrder = [
    'todo',
    'in_progress',
    'pending_review',
    'verifying',
    'done',
    'blocked',
  ];

  static const _columnColors = {
    'todo': Colors.grey,
    'in_progress': Colors.blueAccent,
    'pending_review': Colors.amber,
    'verifying': Colors.orange,
    'done': Colors.green,
    'blocked': Colors.red,
  };

  Map<String, String> _columnLabels(AppLocalizations l) => {
    'todo': l.toDo,
    'in_progress': l.kanbanInProgress,
    'pending_review': l.pendingReview,
    'verifying': l.verifying,
    'done': l.kanbanDone,
    'blocked': l.kanbanBlocked,
  };

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final labels = _columnLabels(l);

    return Consumer<KanbanProvider>(
      builder: (context, kanban, _) {
        if (kanban.loading && kanban.tasks.isEmpty) {
          return const Center(child: CircularProgressIndicator());
        }

        final grouped = kanban.tasksByStatus;

        return SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.all(12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: _columnOrder.map((status) {
              return Padding(
                padding: const EdgeInsets.only(right: 12),
                child: KanbanColumn(
                  status: status,
                  label: labels[status] ?? status,
                  color: _columnColors[status] ?? Colors.grey,
                  tasks: grouped[status] ?? [],
                ),
              );
            }).toList(),
          ),
        );
      },
    );
  }
}
