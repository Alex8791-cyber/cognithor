import 'package:flutter/material.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';

class KanbanCard extends StatelessWidget {
  final KanbanTask task;
  final Color columnColor;

  const KanbanCard({super.key, required this.task, required this.columnColor});

  static const _priorityIcons = {
    'urgent': Icons.priority_high,
    'high': Icons.arrow_upward,
    'medium': Icons.remove,
    'low': Icons.arrow_downward,
  };

  static const _priorityColors = {
    'urgent': Colors.red,
    'high': Colors.orange,
    'medium': Colors.grey,
    'low': Colors.blue,
  };

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Draggable<KanbanTask>(
      data: task,
      feedback: Material(
        elevation: 8,
        borderRadius: BorderRadius.circular(8),
        child: SizedBox(width: 220, child: _buildCard(theme, dragging: true)),
      ),
      childWhenDragging: Opacity(opacity: 0.3, child: _buildCard(theme)),
      child: _buildCard(theme),
    );
  }

  Widget _buildCard(ThemeData theme, {bool dragging = false}) {
    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(8),
        border: Border(left: BorderSide(color: columnColor, width: 3)),
        boxShadow: dragging
            ? [
                BoxShadow(
                  color: columnColor.withValues(alpha: 0.3),
                  blurRadius: 12,
                ),
              ]
            : [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.1),
                  blurRadius: 2,
                ),
              ],
      ),
      padding: const EdgeInsets.all(10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title + priority
          Row(
            children: [
              Expanded(
                child: Text(
                  task.title,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                    decoration: task.status == 'done'
                        ? TextDecoration.lineThrough
                        : null,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Icon(
                _priorityIcons[task.priority] ?? Icons.remove,
                size: 14,
                color: _priorityColors[task.priority] ?? Colors.grey,
              ),
            ],
          ),
          if (task.assignedAgent.isNotEmpty) ...[
            const SizedBox(height: 4),
            Row(
              children: [
                Icon(
                  Icons.smart_toy_outlined,
                  size: 12,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                ),
                const SizedBox(width: 4),
                Text(
                  task.assignedAgent,
                  style: TextStyle(
                    fontSize: 11,
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                  ),
                ),
              ],
            ),
          ],
          if (task.labels.isNotEmpty) ...[
            const SizedBox(height: 6),
            Wrap(
              spacing: 4,
              runSpacing: 2,
              children: task.labels.take(3).map((label) {
                return Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 1,
                  ),
                  decoration: BoxDecoration(
                    color: columnColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    label,
                    style: TextStyle(fontSize: 10, color: columnColor),
                  ),
                );
              }).toList(),
            ),
          ],
          if (task.subtasks.isNotEmpty) ...[
            const SizedBox(height: 4),
            Row(
              children: [
                Icon(
                  Icons.account_tree_outlined,
                  size: 12,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.4),
                ),
                const SizedBox(width: 4),
                Text(
                  '${task.subtasks.where((s) => s.status == "done").length}/${task.subtasks.length}',
                  style: TextStyle(
                    fontSize: 10,
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.4),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}
