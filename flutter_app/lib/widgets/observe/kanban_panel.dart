import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class KanbanPanel extends StatelessWidget {
  const KanbanPanel({super.key, required this.entries});

  final List<Map<String, dynamic>> entries;

  @override
  Widget build(BuildContext context) {
    final columns = {
      'To Do': <Map<String, dynamic>>[],
      'In Progress': <Map<String, dynamic>>[],
      'Verifying': <Map<String, dynamic>>[],
      'Done': <Map<String, dynamic>>[],
    };

    for (final e in entries) {
      final status = (e['status'] ?? '').toString().toLowerCase();
      if (status == 'running' || status == 'active') {
        columns['In Progress']!.add(e);
      } else if (status == 'complete' || status == 'done') {
        columns['Done']!.add(e);
      } else if (status == 'verifying') {
        columns['Verifying']!.add(e);
      } else {
        columns['To Do']!.add(e);
      }
    }

    final columnColors = {
      'To Do': CognithorTheme.textSecondary,
      'In Progress': CognithorTheme.accent,
      'Verifying': CognithorTheme.orange,
      'Done': CognithorTheme.green,
    };

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.all(8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: columns.entries.map((col) {
          final color = columnColors[col.key] ?? CognithorTheme.textSecondary;
          final count = col.value.length;
          return Container(
            width: 160,
            margin: const EdgeInsets.only(right: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Column header with count badge
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          col.key,
                          style: TextStyle(
                            color: color,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      const SizedBox(width: 4),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 1,
                        ),
                        decoration: BoxDecoration(
                          color: color.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '$count',
                          style: TextStyle(
                            color: color,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 6),
                // Cards
                ...col.value.map((item) {
                  final name = (item['phase'] ?? item['name'] ?? '').toString();
                  final tool = (item['tool'] ?? '').toString();
                  return Container(
                    margin: const EdgeInsets.only(bottom: 4),
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Theme.of(context).cardColor,
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(color: Theme.of(context).dividerColor),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          name,
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        if (tool.isNotEmpty)
                          Text(
                            tool,
                            style: TextStyle(
                              fontSize: 10,
                              color: CognithorTheme.textSecondary,
                            ),
                          ),
                      ],
                    ),
                  );
                }),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }
}
