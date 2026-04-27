import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';

class LeadCard extends StatelessWidget {
  const LeadCard({
    super.key,
    required this.lead,
    this.onTap,
    this.onReply,
    this.onArchive,
  });

  final RedditLead lead;
  final VoidCallback? onTap;
  final VoidCallback? onReply;
  final VoidCallback? onArchive;

  Color _scoreColor(int score) {
    if (score >= 80) return CognithorTheme.green;
    if (score >= 60) return CognithorTheme.accent;
    if (score >= 40) return Colors.orange;
    return CognithorTheme.red;
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'new':
        return CognithorTheme.accent;
      case 'reviewed':
        return Colors.orange;
      case 'replied':
        return CognithorTheme.green;
      case 'archived':
        return CognithorTheme.textSecondary;
      default:
        return CognithorTheme.textSecondary;
    }
  }

  String _statusLabel(AppLocalizations l, String status) {
    switch (status) {
      case 'new':
        return l.leadNew;
      case 'reviewed':
        return l.leadReviewed;
      case 'replied':
        return l.leadReplied;
      case 'archived':
        return l.leadArchived;
      default:
        return status;
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final scoreColor = _scoreColor(lead.intentScore);

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: NeonCard(
        tint: scoreColor,
        glowOnHover: true,
        onTap: onTap,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header: score badge + subreddit + status chip
            Row(
              children: [
                // Score badge
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: scoreColor.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                      color: scoreColor.withValues(alpha: 0.4),
                    ),
                  ),
                  child: Text(
                    '${lead.intentScore}',
                    style: TextStyle(
                      color: scoreColor,
                      fontWeight: FontWeight.w800,
                      fontSize: 14,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                // Subreddit
                Text(
                  'r/${lead.subreddit}',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: CognithorTheme.textSecondary,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                // Status chip
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: _statusColor(lead.status).withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    _statusLabel(l, lead.status),
                    style: TextStyle(
                      color: _statusColor(lead.status),
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            // Title
            Text(
              lead.title,
              style: theme.textTheme.titleSmall,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 4),
            // Meta row
            DefaultTextStyle(
              style:
                  theme.textTheme.bodySmall?.copyWith(
                    color: CognithorTheme.textSecondary,
                    fontSize: 11,
                  ) ??
                  const TextStyle(fontSize: 11),
              child: Row(
                children: [
                  Text('u/${lead.author}'),
                  const SizedBox(width: 8),
                  Text('${lead.upvotes} upvotes'),
                  const SizedBox(width: 8),
                  Text('${lead.numComments} comments'),
                  const Spacer(),
                  Text(lead.timeAgo),
                ],
              ),
            ),
            // Action buttons
            if (lead.status != 'archived') ...[
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  if (lead.status == 'new')
                    TextButton.icon(
                      onPressed: onArchive,
                      icon: Icon(
                        Icons.archive_outlined,
                        size: 14,
                        color: CognithorTheme.textSecondary,
                      ),
                      label: Text(
                        l.archiveLead,
                        style: TextStyle(
                          fontSize: 11,
                          color: CognithorTheme.textSecondary,
                        ),
                      ),
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 8),
                      ),
                    ),
                  if (lead.effectiveReply.isNotEmpty)
                    TextButton.icon(
                      onPressed: onReply,
                      icon: Icon(Icons.reply, size: 14, color: scoreColor),
                      label: Text(
                        l.copyReply,
                        style: TextStyle(fontSize: 11, color: scoreColor),
                      ),
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 8),
                      ),
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
