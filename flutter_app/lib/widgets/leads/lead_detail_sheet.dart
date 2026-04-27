import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/cognithor_toast.dart';
import 'package:cognithor_ui/widgets/leads/performance_badge.dart';
import 'package:cognithor_ui/widgets/leads/feedback_dialog.dart';
import 'package:cognithor_ui/widgets/leads/refine_panel.dart';

class LeadDetailSheet extends StatefulWidget {
  const LeadDetailSheet({super.key, required this.lead});
  final RedditLead lead;

  @override
  State<LeadDetailSheet> createState() => _LeadDetailSheetState();
}

class _LeadDetailSheetState extends State<LeadDetailSheet> {
  late final TextEditingController _replyCtrl;
  bool _posting = false;

  @override
  void initState() {
    super.initState();
    _replyCtrl = TextEditingController(text: widget.lead.effectiveReply);
  }

  @override
  void dispose() {
    _replyCtrl.dispose();
    super.dispose();
  }

  Future<void> _copyReply() async {
    await Clipboard.setData(ClipboardData(text: _replyCtrl.text));
    if (mounted) {
      final l = AppLocalizations.of(context);
      CognithorToast.show(context, l.copyReply, type: ToastType.success);
    }
  }

  Future<void> _openOnReddit() async {
    final uri = Uri.tryParse(widget.lead.url);
    if (uri != null) await launchUrl(uri, mode: LaunchMode.externalApplication);
  }

  Future<void> _postReply() async {
    setState(() => _posting = true);
    final provider = context.read<RedditLeadsProvider>();
    // Save edited reply first
    if (_replyCtrl.text != widget.lead.effectiveReply) {
      await provider.updateLead(widget.lead.id, replyFinal: _replyCtrl.text);
    }
    final ok = await provider.replyToLead(widget.lead.id);
    setState(() => _posting = false);
    if (ok && mounted) {
      final l = AppLocalizations.of(context);
      CognithorToast.show(context, l.postReply, type: ToastType.success);
      Navigator.of(context).pop();
    }
  }

  Future<void> _markReviewed() async {
    await context.read<RedditLeadsProvider>().updateLead(
      widget.lead.id,
      status: 'reviewed',
    );
    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _archive() async {
    await context.read<RedditLeadsProvider>().updateLead(
      widget.lead.id,
      status: 'archived',
    );
    if (mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final lead = widget.lead;

    return DraggableScrollableSheet(
      initialChildSize: 0.7,
      minChildSize: 0.3,
      maxChildSize: 0.95,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
          ),
          child: Column(
            children: [
              Expanded(
                child: ListView(
                  controller: scrollController,
                  padding: const EdgeInsets.all(20),
                  children: [
                    // Drag handle
                    Center(
                      child: Container(
                        width: 40,
                        height: 4,
                        margin: const EdgeInsets.only(bottom: 16),
                        decoration: BoxDecoration(
                          color: CognithorTheme.textSecondary.withValues(
                            alpha: 0.3,
                          ),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),

                    // Score + subreddit header
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: CognithorTheme.accent.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            '${lead.intentScore}/100',
                            style: TextStyle(
                              color: CognithorTheme.accent,
                              fontWeight: FontWeight.w800,
                              fontSize: 16,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          'r/${lead.subreddit}',
                          style: theme.textTheme.titleSmall?.copyWith(
                            color: CognithorTheme.textSecondary,
                          ),
                        ),
                        const Spacer(),
                        Text(
                          'u/${lead.author}',
                          style: theme.textTheme.bodySmall,
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),

                    // Title
                    Text(lead.title, style: theme.textTheme.titleLarge),
                    const SizedBox(height: 8),

                    // Body
                    if (lead.body.isNotEmpty) ...[
                      Text(lead.body, style: theme.textTheme.bodyMedium),
                      const SizedBox(height: 12),
                    ],

                    // Score reason
                    if (lead.scoreReason.isNotEmpty) ...[
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: CognithorTheme.accent.withValues(alpha: 0.08),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              l.scoreReason,
                              style: theme.textTheme.labelSmall?.copyWith(
                                color: CognithorTheme.accent,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              lead.scoreReason,
                              style: theme.textTheme.bodySmall,
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],

                    // Reply editor
                    Text(l.editReply, style: theme.textTheme.titleSmall),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _replyCtrl,
                      maxLines: 6,
                      decoration: InputDecoration(
                        border: const OutlineInputBorder(),
                        hintText: l.draftReply,
                      ),
                    ),
                    const SizedBox(height: 8),
                    RefinePanel(
                      leadId: lead.id,
                      currentDraft: _replyCtrl.text,
                      onAccept: (text) =>
                          setState(() => _replyCtrl.text = text),
                    ),
                    const SizedBox(height: 16),

                    // Metadata
                    const SizedBox(height: 16),
                    DefaultTextStyle(
                      style:
                          theme.textTheme.bodySmall?.copyWith(
                            color: CognithorTheme.textSecondary.withValues(
                              alpha: 0.6,
                            ),
                            fontSize: 10,
                          ) ??
                          const TextStyle(fontSize: 10),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${lead.upvotes} upvotes  |  ${lead.numComments} comments  |  ${lead.timeAgo}',
                          ),
                          Text('Post ID: ${lead.postId}'),
                        ],
                      ),
                    ),

                    // Performance + feedback (replied leads only)
                    if (lead.status == 'replied') ...[
                      const Divider(height: 24),
                      Row(
                        children: [
                          Text(
                            l.engagementScore,
                            style: theme.textTheme.titleSmall,
                          ),
                          const SizedBox(width: 8),
                          FutureBuilder<Map<String, dynamic>>(
                            future: context
                                .read<RedditLeadsProvider>()
                                .getPerformance(lead.id),
                            builder: (context, snap) {
                              if (!snap.hasData) return const SizedBox.shrink();
                              final perf =
                                  snap.data?['performance']
                                      as Map<String, dynamic>?;
                              if (perf == null)
                                return const Text('Not tracked yet');
                              final score =
                                  (perf['engagement_score'] as num?)?.toInt() ??
                                  0;
                              return PerformanceBadge(score: score);
                            },
                          ),
                          const Spacer(),
                          TextButton.icon(
                            onPressed: () async {
                              final provider = context
                                  .read<RedditLeadsProvider>();
                              final tag = await showDialog<String>(
                                context: context,
                                builder: (_) => const FeedbackDialog(),
                              );
                              if (tag != null && mounted) {
                                await provider.setFeedback(lead.id, tag: tag);
                                if (!mounted) return;
                                // ignore: use_build_context_synchronously
                                CognithorToast.show(
                                  context,
                                  'Feedback saved',
                                  type: ToastType.success,
                                );
                              }
                            },
                            icon: const Icon(Icons.feedback, size: 16),
                            label: Text(
                              l.feedbackTitle,
                              style: const TextStyle(fontSize: 11),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
              // Sticky footer with action buttons
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surface,
                  border: Border(
                    top: BorderSide(color: theme.dividerColor, width: 0.5),
                  ),
                ),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  alignment: WrapAlignment.end,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _posting ? null : _postReply,
                      icon: _posting
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.reply, size: 18),
                      label: Text(l.postReply),
                    ),
                    OutlinedButton.icon(
                      onPressed: _copyReply,
                      icon: const Icon(Icons.copy, size: 18),
                      label: Text(l.copyReply),
                    ),
                    OutlinedButton.icon(
                      onPressed: _openOnReddit,
                      icon: const Icon(Icons.open_in_new, size: 18),
                      label: Text(l.openOnReddit),
                    ),
                    if (lead.status == 'new')
                      TextButton.icon(
                        onPressed: _markReviewed,
                        icon: const Icon(Icons.check, size: 18),
                        label: Text(l.markReviewed),
                      ),
                    if (lead.status != 'archived')
                      TextButton.icon(
                        onPressed: _archive,
                        icon: Icon(
                          Icons.archive,
                          size: 18,
                          color: CognithorTheme.textSecondary,
                        ),
                        label: Text(
                          l.archiveLead,
                          style: TextStyle(color: CognithorTheme.textSecondary),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
