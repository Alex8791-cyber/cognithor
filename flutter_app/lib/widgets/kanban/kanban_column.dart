import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/widgets/kanban/kanban_card.dart';

class KanbanColumn extends StatelessWidget {
  final String status;
  final String label;
  final Color color;
  final List<KanbanTask> tasks;

  const KanbanColumn({
    super.key,
    required this.status,
    required this.label,
    required this.color,
    required this.tasks,
  });

  void _onReorder(BuildContext context, int oldIndex, int newIndex) {
    if (oldIndex < newIndex) newIndex -= 1;
    final reordered = List<KanbanTask>.from(tasks);
    final item = reordered.removeAt(oldIndex);
    reordered.insert(newIndex, item);
    // Batch-update sort_order
    final provider = context.read<KanbanProvider>();
    final batch = <Map<String, dynamic>>[];
    for (int i = 0; i < reordered.length; i++) {
      reordered[i].sortOrder = i;
      batch.add({'id': reordered[i].id, 'sort_order': i});
    }
    provider.reorderTasks(batch);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return DragTarget<KanbanTask>(
      onWillAcceptWithDetails: (details) => details.data.status != status,
      onAcceptWithDetails: (details) {
        context.read<KanbanProvider>().moveTask(details.data.id, status);
      },
      builder: (context, candidateData, rejectedData) {
        final isHovering = candidateData.isNotEmpty;

        return AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          width: 240,
          constraints: const BoxConstraints(minHeight: 400),
          decoration: BoxDecoration(
            color: isHovering
                ? color.withValues(alpha: 0.15)
                : theme.colorScheme.surfaceContainerHighest.withValues(
                    alpha: 0.3,
                  ),
            borderRadius: BorderRadius.circular(12),
            border: isHovering
                ? Border.all(color: color, width: 2)
                : Border.all(color: Colors.transparent),
          ),
          padding: const EdgeInsets.all(8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Column header
              Row(
                children: [
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      color: color,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    label,
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const Spacer(),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      '${tasks.length}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: color,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              const Divider(height: 1),
              const SizedBox(height: 8),
              // Task cards — reorderable within column
              if (tasks.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 24),
                  child: Center(
                    child: Text(
                      'No tasks',
                      style: TextStyle(
                        color: theme.colorScheme.onSurface.withValues(
                          alpha: 0.4,
                        ),
                        fontSize: 12,
                      ),
                    ),
                  ),
                )
              else
                ReorderableListView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  buildDefaultDragHandles: false,
                  itemCount: tasks.length,
                  onReorder: (oldIdx, newIdx) =>
                      _onReorder(context, oldIdx, newIdx),
                  proxyDecorator: (child, index, animation) {
                    return AnimatedBuilder(
                      animation: animation,
                      builder: (context, child) => Material(
                        elevation: 4,
                        color: Colors.transparent,
                        borderRadius: BorderRadius.circular(8),
                        child: child,
                      ),
                      child: child,
                    );
                  },
                  itemBuilder: (context, index) {
                    final task = tasks[index];
                    return ReorderableDragStartListener(
                      key: ValueKey(task.id),
                      index: index,
                      child: Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: KanbanCard(task: task, columnColor: color),
                      ),
                    );
                  },
                ),
            ],
          ),
        );
      },
    );
  }
}
