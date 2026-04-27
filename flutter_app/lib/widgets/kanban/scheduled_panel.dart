import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/cron_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class ScheduledPanel extends StatelessWidget {
  const ScheduledPanel({super.key});

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Consumer<CronProvider>(
      builder: (context, cron, _) {
        if (cron.loading && cron.jobs.isEmpty) {
          return const Center(child: CircularProgressIndicator());
        }

        if (cron.jobs.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.schedule,
                  size: 48,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
                ),
                const SizedBox(height: 12),
                Text(
                  l.noScheduledTasks,
                  style: theme.textTheme.bodyLarge?.copyWith(
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  l.scheduledTasksHint,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          );
        }

        final enabled = cron.jobs.where((j) => j.enabled).toList();
        final paused = cron.jobs.where((j) => !j.enabled).toList();

        return RefreshIndicator(
          onRefresh: cron.fetchJobs,
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (enabled.isNotEmpty) ...[
                _sectionHeader(
                  theme,
                  l.activeJobs,
                  enabled.length,
                  Colors.green,
                ),
                const SizedBox(height: 8),
                ...enabled.map((j) => _CronJobCard(job: j)),
              ],
              if (paused.isNotEmpty) ...[
                const SizedBox(height: 20),
                _sectionHeader(
                  theme,
                  l.pausedJobs,
                  paused.length,
                  Colors.orange,
                ),
                const SizedBox(height: 8),
                ...paused.map((j) => _CronJobCard(job: j)),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _sectionHeader(ThemeData theme, String label, int count, Color color) {
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 8),
        Text(
          label,
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(width: 8),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            '$count',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ),
      ],
    );
  }
}

class _CronJobCard extends StatelessWidget {
  final CronJob job;
  const _CronJobCard({required this.job});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final l = AppLocalizations.of(context);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  _iconForAction(job.name),
                  size: 20,
                  color: job.enabled
                      ? CognithorTheme.accent
                      : theme.colorScheme.onSurface.withValues(alpha: 0.4),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    job.name.replaceAll('_', ' ').toUpperCase(),
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: job.enabled
                          ? null
                          : theme.colorScheme.onSurface.withValues(alpha: 0.5),
                    ),
                  ),
                ),
                // Pause/Resume toggle
                Switch(
                  value: job.enabled,
                  onChanged: (_) =>
                      context.read<CronProvider>().toggleJob(job.name),
                  activeThumbColor: CognithorTheme.accent,
                ),
              ],
            ),
            const SizedBox(height: 6),
            // Schedule + next run
            Row(
              children: [
                Icon(
                  Icons.access_time,
                  size: 14,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.4),
                ),
                const SizedBox(width: 4),
                Text(
                  job.scheduleLabel,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                  ),
                ),
                if (job.nextRun != null) ...[
                  const SizedBox(width: 12),
                  Icon(
                    Icons.play_arrow,
                    size: 14,
                    color: Colors.green.withValues(alpha: 0.6),
                  ),
                  const SizedBox(width: 2),
                  Text(
                    '${l.nextRun}: ${_formatNextRun(job.nextRun!)}',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: Colors.green.withValues(alpha: 0.8),
                      fontSize: 11,
                    ),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 4),
            // Prompt preview
            Text(
              job.prompt.length > 80
                  ? '${job.prompt.substring(0, 80)}...'
                  : job.prompt,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurface.withValues(alpha: 0.4),
                fontSize: 11,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 4),
            // Tags: channel + model
            Wrap(
              spacing: 6,
              children: [
                _tag(theme, job.channel, Icons.send, Colors.blue),
                _tag(theme, job.model, Icons.psychology, Colors.purple),
                if (job.agent.isNotEmpty)
                  _tag(theme, job.agent, Icons.smart_toy, Colors.teal),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _tag(ThemeData theme, String label, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 10, color: color.withValues(alpha: 0.7)),
          const SizedBox(width: 3),
          Text(
            label,
            style: TextStyle(fontSize: 10, color: color.withValues(alpha: 0.8)),
          ),
        ],
      ),
    );
  }

  IconData _iconForAction(String name) {
    if (name.contains('briefing') || name.contains('morning')) {
      return Icons.wb_sunny;
    }
    if (name.contains('review')) {
      return Icons.rate_review;
    }
    if (name.contains('memory') || name.contains('maintenance')) {
      return Icons.memory;
    }
    if (name.contains('scan') || name.contains('reddit')) {
      return Icons.radar;
    }
    if (name.contains('backup')) {
      return Icons.backup;
    }
    return Icons.schedule;
  }

  String _formatNextRun(String iso) {
    try {
      final dt = DateTime.parse(iso);
      final now = DateTime.now();
      final diff = dt.difference(now);
      if (diff.isNegative) return 'overdue';
      if (diff.inMinutes < 60) return '${diff.inMinutes}m';
      if (diff.inHours < 24) return '${diff.inHours}h ${diff.inMinutes % 60}m';
      return '${diff.inDays}d ${diff.inHours % 24}h';
    } catch (_) {
      return iso;
    }
  }
}
