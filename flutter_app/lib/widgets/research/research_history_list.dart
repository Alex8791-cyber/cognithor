import 'package:flutter/material.dart';
import 'package:cognithor_ui/providers/research_provider.dart';

/// A scrollable list of past research sessions.
///
/// Each item shows the query, relative date, hop count, and a confidence
/// badge (green >= 80%, yellow >= 50%, red < 50%).
class ResearchHistoryList extends StatelessWidget {
  final List<ResearchSummary> history;
  final void Function(String id) onSelect;
  final void Function(String id) onDelete;

  const ResearchHistoryList({
    super.key,
    required this.history,
    required this.onSelect,
    required this.onDelete,
  });

  Color _confidenceColor(double confidence) {
    if (confidence >= 0.8) return const Color(0xFF00E676);
    if (confidence >= 0.5) return const Color(0xFFFFAB40);
    return const Color(0xFFFF5252);
  }

  String _confidenceLabel(double confidence) {
    return '${(confidence * 100).toStringAsFixed(0)}%';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (history.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.history,
              size: 48,
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.4),
            ),
            const SizedBox(height: 12),
            Text(
              'No research history yet.',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      itemCount: history.length,
      itemBuilder: (context, index) {
        final item = history[index];
        final confColor = _confidenceColor(item.confidenceAvg);

        return Dismissible(
          key: ValueKey(item.id),
          direction: DismissDirection.endToStart,
          background: Container(
            alignment: Alignment.centerRight,
            padding: const EdgeInsets.only(right: 16),
            color: const Color(0xFFFF5252).withValues(alpha: 0.15),
            child: const Icon(Icons.delete_outline, color: Color(0xFFFF5252)),
          ),
          onDismissed: (_) => onDelete(item.id),
          child: ListTile(
            onTap: () => onSelect(item.id),
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 12,
              vertical: 4,
            ),
            title: Text(
              item.query,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w500,
              ),
            ),
            subtitle: Row(
              children: [
                Icon(
                  Icons.schedule,
                  size: 11,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                const SizedBox(width: 3),
                Text(
                  item.timeAgo,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
                const SizedBox(width: 8),
                Icon(
                  Icons.account_tree_outlined,
                  size: 11,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                const SizedBox(width: 3),
                Text(
                  '${item.hops} hops',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
            trailing: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: confColor.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: confColor.withValues(alpha: 0.4)),
              ),
              child: Text(
                _confidenceLabel(item.confidenceAvg),
                style: TextStyle(
                  color: confColor,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
