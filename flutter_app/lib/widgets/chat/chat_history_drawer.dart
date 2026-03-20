import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/jarvis_confirmation_dialog.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';

/// Sidebar drawer showing past chat sessions, ordered by last activity.
class ChatHistoryDrawer extends StatelessWidget {
  const ChatHistoryDrawer({
    super.key,
    required this.sessions,
    required this.activeSessionId,
    required this.onSelectSession,
    required this.onNewChat,
    required this.onDeleteSession,
  });

  final List<Map<String, dynamic>> sessions;
  final String? activeSessionId;
  final ValueChanged<String> onSelectSession;
  final VoidCallback onNewChat;
  final ValueChanged<String> onDeleteSession;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Drawer(
      backgroundColor: theme.scaffoldBackgroundColor,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.horizontal(right: Radius.circular(16)),
      ),
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 8, 8),
              child: Row(
                children: [
                  const Icon(
                    Icons.history,
                    color: JarvisTheme.sectionChat,
                    size: JarvisTheme.iconSizeMd,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      l.chatHistory,
                      style: theme.textTheme.titleLarge?.copyWith(
                        color: JarvisTheme.sectionChat,
                      ),
                    ),
                  ),
                  FilledButton.icon(
                    onPressed: onNewChat,
                    icon: const Icon(Icons.add, size: 18),
                    label: Text(l.newChat),
                    style: FilledButton.styleFrom(
                      backgroundColor:
                          JarvisTheme.sectionChat.withValues(alpha: 0.15),
                      foregroundColor: JarvisTheme.sectionChat,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 8),
                      shape: RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius.circular(JarvisTheme.buttonRadius),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const Divider(height: 1),

            // Sessions list
            Expanded(
              child: sessions.isEmpty
                  ? Center(
                      child: Text(
                        l.noMessages,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.textTheme.bodySmall?.color,
                        ),
                      ),
                    )
                  : ListView.separated(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                      itemCount: sessions.length,
                      separatorBuilder: (context, index) => const SizedBox(height: 6),
                      itemBuilder: (context, index) {
                        final session = sessions[index];
                        final sessionId =
                            session['session_id']?.toString() ?? '';
                        final isActive = sessionId == activeSessionId;

                        return _SessionCard(
                          session: session,
                          isActive: isActive,
                          onTap: () {
                            onSelectSession(sessionId);
                            Navigator.of(context).pop();
                          },
                          onDelete: () async {
                            final confirmed =
                                await JarvisConfirmationDialog.show(
                              context,
                              title: l.deleteChat,
                              message: l.confirmDeleteChat,
                              confirmLabel: l.delete,
                              icon: Icons.delete_outline,
                            );
                            if (confirmed && context.mounted) {
                              onDeleteSession(sessionId);
                            }
                          },
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SessionCard extends StatelessWidget {
  const _SessionCard({
    required this.session,
    required this.isActive,
    required this.onTap,
    required this.onDelete,
  });

  final Map<String, dynamic> session;
  final bool isActive;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final title =
        session['title']?.toString().trim().isNotEmpty == true
            ? session['title'].toString()
            : l.untitledChat;
    final messageCount = session['message_count'] as int? ?? 0;
    final lastActivity = session['last_activity']?.toString();

    return NeonCard(
      tint: isActive ? JarvisTheme.sectionChat : null,
      glowOnHover: true,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      onTap: onTap,
      child: Row(
        children: [
          // Chat icon
          Icon(
            Icons.chat_bubble_outline,
            size: 18,
            color: isActive
                ? JarvisTheme.sectionChat
                : Theme.of(context).iconTheme.color,
          ),
          const SizedBox(width: 10),

          // Title + meta
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: isActive ? FontWeight.w600 : null,
                        color: isActive ? JarvisTheme.sectionChat : null,
                      ),
                ),
                const SizedBox(height: 2),
                Row(
                  children: [
                    if (lastActivity != null) ...[
                      Text(
                        _formatRelativeTime(lastActivity, l),
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      const SizedBox(width: 8),
                    ],
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 6, vertical: 1),
                      decoration: BoxDecoration(
                        color: JarvisTheme.sectionChat.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        l.messagesCount(messageCount.toString()),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              fontSize: 11,
                              color: JarvisTheme.sectionChat,
                            ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Delete button
          IconButton(
            icon: const Icon(Icons.delete_outline, size: 18),
            onPressed: onDelete,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
            color: Theme.of(context).textTheme.bodySmall?.color,
          ),
        ],
      ),
    );
  }

  String _formatRelativeTime(String isoTimestamp, AppLocalizations l) {
    try {
      final dt = DateTime.parse(isoTimestamp);
      final now = DateTime.now();
      final diff = now.difference(dt);

      if (diff.inSeconds < 60) return l.justNow;
      if (diff.inMinutes < 60) return l.minutesAgo(diff.inMinutes.toString());
      if (diff.inHours < 24) return l.hoursAgo(diff.inHours.toString());
      return l.daysAgo(diff.inDays.toString());
    } catch (_) {
      return '';
    }
  }
}
