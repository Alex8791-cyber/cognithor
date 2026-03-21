import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/jarvis_confirmation_dialog.dart';

/// Sidebar drawer showing past chat sessions grouped by folder.
class ChatHistoryDrawer extends StatefulWidget {
  const ChatHistoryDrawer({
    super.key,
    required this.sessions,
    required this.folders,
    required this.activeSessionId,
    required this.onSelectSession,
    required this.onNewChat,
    this.onNewIncognitoChat,
    required this.onDeleteSession,
    required this.onRenameSession,
    required this.onMoveToFolder,
    this.searchResults = const [],
    this.onSearchChanged,
    this.sessionsByProject = const {},
  });

  final List<Map<String, dynamic>> sessions;
  final List<String> folders;
  final String? activeSessionId;
  final ValueChanged<String> onSelectSession;
  final VoidCallback onNewChat;
  final VoidCallback? onNewIncognitoChat;
  final ValueChanged<String> onDeleteSession;
  final void Function(String sessionId, String newTitle) onRenameSession;
  final void Function(String sessionId, String folder) onMoveToFolder;
  final List<Map<String, dynamic>> searchResults;
  final void Function(String query)? onSearchChanged;
  final Map<String, List<Map<String, dynamic>>> sessionsByProject;

  @override
  State<ChatHistoryDrawer> createState() => _ChatHistoryDrawerState();
}

class _ChatHistoryDrawerState extends State<ChatHistoryDrawer> {
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    // Use sessionsByProject if available, otherwise fall back to folder grouping
    final Map<String, List<Map<String, dynamic>>> grouped;
    if (widget.sessionsByProject.isNotEmpty) {
      grouped = widget.sessionsByProject;
    } else {
      grouped = {};
      for (final session in widget.sessions) {
        final folder = session['folder']?.toString().trim() ?? '';
        grouped.putIfAbsent(folder, () => []).add(session);
      }
    }

    // Sort: 'Allgemein' first, then named folders alphabetically, then unfiled ('')
    final sortedFolders = grouped.keys.toList()
      ..sort((a, b) {
        if (a == 'Allgemein') return -1;
        if (b == 'Allgemein') return 1;
        if (a.isEmpty && b.isEmpty) return 0;
        if (a.isEmpty) return 1;
        if (b.isEmpty) return -1;
        return a.compareTo(b);
      });

    final hasSearchResults = widget.searchResults.isNotEmpty;

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
                  if (widget.onNewIncognitoChat != null)
                    IconButton(
                      icon: const Icon(Icons.visibility_off),
                      tooltip: 'Inkognito Chat',
                      onPressed: widget.onNewIncognitoChat,
                    ),
                  FilledButton.icon(
                    onPressed: widget.onNewChat,
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

            // Search bar
            if (widget.onSearchChanged != null)
              Padding(
                padding: const EdgeInsets.all(12),
                child: TextField(
                  controller: _searchController,
                  decoration: InputDecoration(
                    hintText: 'Chats durchsuchen...',
                    prefixIcon: const Icon(Icons.search, size: 20),
                    suffixIcon: _searchController.text.isNotEmpty
                        ? IconButton(
                            icon: const Icon(Icons.clear, size: 18),
                            onPressed: () {
                              _searchController.clear();
                              widget.onSearchChanged!('');
                              setState(() {});
                            },
                          )
                        : null,
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(vertical: 8),
                  ),
                  onChanged: (query) {
                    widget.onSearchChanged!(query);
                    setState(() {});
                  },
                ),
              ),

            // Sessions list — search results or grouped by project
            Expanded(
              child: hasSearchResults
                  ? ListView(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                      children: widget.searchResults.map((r) => ListTile(
                        dense: true,
                        title: Text(
                          r['session_title'] as String? ?? '',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
                        ),
                        subtitle: Text(
                          r['content'] as String? ?? '',
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontSize: 11),
                        ),
                        onTap: () {
                          final sid = r['session_id'] as String?;
                          if (sid != null) {
                            widget.onSelectSession(sid);
                            Navigator.of(context).pop();
                          }
                        },
                      )).toList(),
                    )
                  : widget.sessions.isEmpty
                      ? Center(
                          child: Text(
                            l.noMessages,
                            style: theme.textTheme.bodyMedium?.copyWith(
                              color: theme.textTheme.bodySmall?.color,
                            ),
                          ),
                        )
                      : ListView(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 8,
                          ),
                          children: [
                            for (final folderName in sortedFolders)
                              ExpansionTile(
                                title: Text(
                                  folderName.isEmpty ? l.noFolder : folderName,
                                  style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                    color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
                                  ),
                                ),
                                initiallyExpanded: folderName == 'Allgemein' || folderName.isEmpty,
                                dense: true,
                                tilePadding: const EdgeInsets.symmetric(horizontal: 12),
                                children: grouped[folderName]!.map((s) {
                                  final sessionId = s['session_id']?.toString() ??
                                      s['id']?.toString() ??
                                      '';
                                  final isActive = sessionId == widget.activeSessionId;
                                  final isIncognito = s['incognito'] == true;
                                  return ListTile(
                                    dense: true,
                                    selected: isActive,
                                    leading: isIncognito
                                        ? const Icon(Icons.visibility_off, size: 16, color: Colors.purple)
                                        : null,
                                    title: Text(
                                      (s['title'] as String?)?.isNotEmpty == true
                                          ? s['title'] as String
                                          : l.untitledChat,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: TextStyle(
                                        fontSize: 13,
                                        fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
                                      ),
                                    ),
                                    subtitle: Text(
                                      '${s['message_count'] ?? 0} Nachrichten',
                                      style: const TextStyle(fontSize: 11),
                                    ),
                                    onTap: () {
                                      widget.onSelectSession(sessionId);
                                      Navigator.of(context).pop();
                                    },
                                    onLongPress: () => _showSessionMenu(context, s),
                                  );
                                }).toList(),
                              ),
                          ],
                        ),
            ),
          ],
        ),
      ),
    );
  }

  void _showSessionMenu(BuildContext context, Map<String, dynamic> session) {
    final l = AppLocalizations.of(context);
    final sessionId = session['session_id']?.toString() ??
        session['id']?.toString() ??
        '';

    showModalBottomSheet<String>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.edit, size: 18),
              title: Text(l.renameChat),
              onTap: () {
                Navigator.of(ctx).pop();
                _showRenameDialog(context, sessionId, session);
              },
            ),
            ListTile(
              leading: const Icon(Icons.folder_outlined, size: 18),
              title: Text(l.moveToFolder),
              onTap: () {
                Navigator.of(ctx).pop();
                _showMoveDialog(context, sessionId, session);
              },
            ),
            ListTile(
              leading: Icon(Icons.delete_outline, size: 18, color: JarvisTheme.red),
              title: Text(l.delete, style: TextStyle(color: JarvisTheme.red)),
              onTap: () async {
                Navigator.of(ctx).pop();
                final confirmed = await JarvisConfirmationDialog.show(
                  context,
                  title: l.deleteChat,
                  message: l.confirmDeleteChat,
                  confirmLabel: l.delete,
                  icon: Icons.delete_outline,
                );
                if (confirmed && context.mounted) {
                  widget.onDeleteSession(sessionId);
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showRenameDialog(
    BuildContext context,
    String sessionId,
    Map<String, dynamic> session,
  ) {
    final l = AppLocalizations.of(context);
    final currentTitle = session['title']?.toString().trim() ?? '';
    final controller = TextEditingController(text: currentTitle);

    showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Theme.of(context).cardColor,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(JarvisTheme.cardRadius),
          side: BorderSide(color: Theme.of(context).dividerColor),
        ),
        icon: const Icon(Icons.edit, color: JarvisTheme.sectionChat,
            size: JarvisTheme.iconSizeLg),
        title: Text(l.editTitle),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: InputDecoration(
            hintText: l.untitledChat,
            border: const OutlineInputBorder(),
          ),
          onSubmitted: (value) => Navigator.of(ctx).pop(value.trim()),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(controller.text.trim()),
            style: ElevatedButton.styleFrom(
              backgroundColor: JarvisTheme.sectionChat,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(JarvisTheme.buttonRadius),
              ),
            ),
            child: Text(l.save),
          ),
        ],
      ),
    ).then((newTitle) {
      if (newTitle != null && newTitle.isNotEmpty && newTitle != currentTitle) {
        widget.onRenameSession(sessionId, newTitle);
      }
    });
  }

  void _showMoveDialog(
    BuildContext context,
    String sessionId,
    Map<String, dynamic> session,
  ) {
    showDialog<String>(
      context: context,
      builder: (ctx) => _MoveToFolderDialog(
        folders: widget.folders,
        currentFolder: session['folder']?.toString().trim() ?? '',
      ),
    ).then((folder) {
      if (folder != null) {
        widget.onMoveToFolder(sessionId, folder);
      }
    });
  }
}

class _MoveToFolderDialog extends StatefulWidget {
  const _MoveToFolderDialog({
    required this.folders,
    required this.currentFolder,
  });

  final List<String> folders;
  final String currentFolder;

  @override
  State<_MoveToFolderDialog> createState() => _MoveToFolderDialogState();
}

class _MoveToFolderDialogState extends State<_MoveToFolderDialog> {
  bool _creatingNew = false;
  final _newFolderController = TextEditingController();

  @override
  void dispose() {
    _newFolderController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return AlertDialog(
      backgroundColor: theme.cardColor,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(JarvisTheme.cardRadius),
        side: BorderSide(color: theme.dividerColor),
      ),
      icon: const Icon(Icons.folder_outlined, color: JarvisTheme.sectionChat,
          size: JarvisTheme.iconSizeLg),
      title: Text(l.moveToFolder),
      content: SizedBox(
        width: 280,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // "Unfiled" option to remove from folder
            if (widget.currentFolder.isNotEmpty)
              ListTile(
                leading: const Icon(Icons.remove_circle_outline, size: 20),
                title: Text(l.noFolder),
                dense: true,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                onTap: () => Navigator.of(context).pop(''),
              ),

            // Existing folders
            ...widget.folders
                .where((f) => f != widget.currentFolder)
                .map((folder) => ListTile(
                      leading: const Icon(Icons.folder, size: 20),
                      title: Text(folder),
                      dense: true,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      onTap: () => Navigator.of(context).pop(folder),
                    )),

            const Divider(),

            // New folder creation
            if (_creatingNew)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: TextField(
                  controller: _newFolderController,
                  autofocus: true,
                  decoration: InputDecoration(
                    hintText: l.folderName,
                    border: const OutlineInputBorder(),
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.check),
                      onPressed: _submitNewFolder,
                    ),
                  ),
                  onSubmitted: (_) => _submitNewFolder(),
                ),
              )
            else
              ListTile(
                leading: const Icon(Icons.create_new_folder,
                    size: 20, color: JarvisTheme.sectionChat),
                title: Text(l.newFolder,
                    style: const TextStyle(color: JarvisTheme.sectionChat)),
                dense: true,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                onTap: () => setState(() => _creatingNew = true),
              ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(l.cancel),
        ),
      ],
    );
  }

  void _submitNewFolder() {
    final name = _newFolderController.text.trim();
    if (name.isNotEmpty) {
      Navigator.of(context).pop(name);
    }
  }
}
