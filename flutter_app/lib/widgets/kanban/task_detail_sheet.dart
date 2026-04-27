import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';

/// Bottom sheet showing full task details, subtasks, and history.
class TaskDetailSheet extends StatefulWidget {
  final KanbanTask task;

  const TaskDetailSheet({super.key, required this.task});

  @override
  State<TaskDetailSheet> createState() => _TaskDetailSheetState();
}

class _TaskDetailSheetState extends State<TaskDetailSheet> {
  List<Map<String, dynamic>> _history = [];
  bool _loadingHistory = false;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() => _loadingHistory = true);
    _history = await context.read<KanbanProvider>().getHistory(widget.task.id);
    if (mounted) setState(() => _loadingHistory = false);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final l = AppLocalizations.of(context);
    final task = widget.task;

    return DraggableScrollableSheet(
      initialChildSize: 0.6,
      minChildSize: 0.3,
      maxChildSize: 0.9,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
          ),
          child: ListView(
            controller: scrollController,
            padding: const EdgeInsets.all(20),
            children: [
              // Handle
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              // Title
              Text(task.title, style: theme.textTheme.headlineSmall),
              const SizedBox(height: 8),
              // Status + Priority + Agent
              Wrap(
                spacing: 8,
                runSpacing: 4,
                children: [
                  Chip(label: Text(task.statusDisplay)),
                  Chip(
                    label: Text(task.priority),
                    avatar: Icon(_priorityIcon(task.priority), size: 16),
                  ),
                  if (task.assignedAgent.isNotEmpty)
                    Chip(
                      label: Text(task.assignedAgent),
                      avatar: const Icon(Icons.smart_toy_outlined, size: 16),
                    ),
                ],
              ),
              // Labels
              if (task.labels.isNotEmpty) ...[
                const SizedBox(height: 8),
                Wrap(
                  spacing: 4,
                  children: task.labels
                      .map((l) => Chip(label: Text(l)))
                      .toList(),
                ),
              ],
              // Description
              if (task.description.isNotEmpty) ...[
                const SizedBox(height: 16),
                Text(l.kanbanDescription, style: theme.textTheme.titleMedium),
                const SizedBox(height: 4),
                Text(task.description),
              ],
              // Result
              if (task.resultSummary.isNotEmpty) ...[
                const SizedBox(height: 16),
                Text(l.kanbanResult, style: theme.textTheme.titleMedium),
                const SizedBox(height: 4),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.green.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(task.resultSummary),
                ),
              ],
              // Subtasks
              if (task.subtasks.isNotEmpty) ...[
                const SizedBox(height: 16),
                Text(
                  l.kanbanSubtasks(task.subtasks.length),
                  style: theme.textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                ...task.subtasks.map(
                  (sub) => ListTile(
                    leading: Icon(
                      sub.status == 'done'
                          ? Icons.check_circle
                          : Icons.radio_button_unchecked,
                      color: sub.status == 'done' ? Colors.green : Colors.grey,
                    ),
                    title: Text(sub.title),
                    subtitle: Text(sub.statusDisplay),
                    dense: true,
                  ),
                ),
              ],
              // History
              const SizedBox(height: 16),
              Text(l.kanbanHistory, style: theme.textTheme.titleMedium),
              const SizedBox(height: 4),
              if (_loadingHistory)
                const Center(child: CircularProgressIndicator())
              else if (_history.isEmpty)
                Text(l.kanbanNoHistory)
              else
                ..._history.map(
                  (h) => ListTile(
                    leading: const Icon(Icons.history, size: 18),
                    title: Text('${h["old_status"]} -> ${h["new_status"]}'),
                    subtitle: Text(
                      'by ${h["changed_by"]} - ${h["changed_at"] ?? ""}',
                    ),
                    dense: true,
                  ),
                ),
              // Metadata
              const SizedBox(height: 16),
              Text(l.kanbanMetadata, style: theme.textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(
                l.kanbanSource(task.source),
                style: theme.textTheme.bodySmall,
              ),
              Text(
                l.kanbanCreated(task.createdAt),
                style: theme.textTheme.bodySmall,
              ),
              Text(
                l.kanbanUpdated(task.updatedAt),
                style: theme.textTheme.bodySmall,
              ),
              if (task.completedAt.isNotEmpty)
                Text(
                  l.kanbanCompleted(task.completedAt),
                  style: theme.textTheme.bodySmall,
                ),
              const SizedBox(height: 32),
            ],
          ),
        );
      },
    );
  }

  IconData _priorityIcon(String priority) {
    switch (priority) {
      case 'urgent':
        return Icons.priority_high;
      case 'high':
        return Icons.arrow_upward;
      case 'low':
        return Icons.arrow_downward;
      default:
        return Icons.remove;
    }
  }
}
