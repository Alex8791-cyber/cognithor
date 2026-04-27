import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/models/chat_node.dart';
import 'package:cognithor_ui/providers/tree_provider.dart';

/// Resizable sidebar showing the full conversation tree.
/// Nodes are clickable — clicking scrolls the chat to that message.
/// Drag the right edge to resize.
class TreeSidebar extends StatefulWidget {
  const TreeSidebar({super.key, this.onNodeTap});

  /// Called when a node is tapped — passes the node index in the active path.
  final void Function(String nodeId)? onNodeTap;

  @override
  State<TreeSidebar> createState() => _TreeSidebarState();
}

class _TreeSidebarState extends State<TreeSidebar> {
  double _width = 260;
  static const _minWidth = 180.0;
  static const _maxWidth = 450.0;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Stack(
      children: [
        // Sidebar content
        Consumer<TreeProvider>(
          builder: (context, tree, _) {
            return Container(
              width: _width,
              decoration: BoxDecoration(
                color: theme.scaffoldBackgroundColor,
                border: Border(
                  right: BorderSide(color: cs.outlineVariant, width: 1),
                ),
              ),
              child: Column(
                children: [
                  // Header
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: cs.surface,
                      border: Border(
                        bottom: BorderSide(color: cs.outlineVariant, width: 1),
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.account_tree, size: 16, color: cs.primary),
                        const SizedBox(width: 8),
                        Text(
                          'Conversation Tree',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: cs.primary,
                          ),
                        ),
                        const Spacer(),
                        if (tree.hasTree)
                          Text(
                            '${tree.nodes.length}',
                            style: TextStyle(
                              fontSize: 11,
                              color: cs.onSurface.withValues(alpha: 0.5),
                            ),
                          ),
                      ],
                    ),
                  ),
                  // Tree content
                  Expanded(
                    child: !tree.hasTree
                        ? Center(
                            child: Text(
                              'No conversation yet',
                              style: TextStyle(
                                fontSize: 12,
                                color: cs.onSurface.withValues(alpha: 0.4),
                              ),
                            ),
                          )
                        : ListView(
                            padding: const EdgeInsets.all(8),
                            children: [
                              for (final root in _getRoots(tree))
                                _buildNode(context, tree, root, 0),
                            ],
                          ),
                  ),
                ],
              ),
            );
          },
        ),
        // Drag handle for resize
        Positioned(
          right: 0,
          top: 0,
          bottom: 0,
          child: MouseRegion(
            cursor: SystemMouseCursors.resizeColumn,
            child: GestureDetector(
              onHorizontalDragUpdate: (details) {
                setState(() {
                  _width = (_width + details.delta.dx).clamp(
                    _minWidth,
                    _maxWidth,
                  );
                });
              },
              child: Container(width: 6, color: Colors.transparent),
            ),
          ),
        ),
      ],
    );
  }

  List<ChatNode> _getRoots(TreeProvider tree) {
    return tree.nodes.values.where((n) => n.parentId == null).toList()
      ..sort((a, b) => a.createdAt.compareTo(b.createdAt));
  }

  Widget _buildNode(
    BuildContext context,
    TreeProvider tree,
    ChatNode node,
    int depth,
  ) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final isActive = tree.activePath.contains(node.id);
    final isFork = tree.isForkPoint(node.id);
    final children =
        tree.nodes.values.where((n) => n.parentId == node.id).toList()
          ..sort((a, b) => a.branchIndex.compareTo(b.branchIndex));

    final displayText = node.text.length > 40
        ? '${node.text.substring(0, 40)}...'
        : node.text;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Tooltip(
          message: node.text.length > 40 ? node.text.substring(0, 200) : '',
          waitDuration: const Duration(milliseconds: 500),
          child: InkWell(
            borderRadius: BorderRadius.circular(6),
            onTap: () {
              // Navigate to this node's branch
              if (node.parentId != null) {
                tree.switchBranch(node.parentId!, node.branchIndex);
              }
              widget.onNodeTap?.call(node.id);
            },
            child: Container(
              margin: EdgeInsets.only(left: depth * 14.0, bottom: 2),
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
              decoration: BoxDecoration(
                color: isActive
                    ? cs.primary.withValues(alpha: 0.12)
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(6),
                border: isActive
                    ? Border.all(color: cs.primary.withValues(alpha: 0.3))
                    : null,
              ),
              child: Row(
                children: [
                  Container(
                    width: 18,
                    height: 18,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: node.isUser
                          ? cs.primary.withValues(alpha: 0.2)
                          : Colors.green.withValues(alpha: 0.2),
                    ),
                    child: Icon(
                      node.isUser ? Icons.person : Icons.smart_toy,
                      size: 11,
                      color: node.isUser ? cs.primary : Colors.green,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      displayText,
                      style: TextStyle(
                        fontSize: 11,
                        color: isActive
                            ? cs.onSurface
                            : cs.onSurface.withValues(alpha: 0.7),
                        fontWeight: isActive
                            ? FontWeight.w500
                            : FontWeight.normal,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (isFork) ...[
                    const SizedBox(width: 4),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 4,
                        vertical: 1,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.orange.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        '${tree.getChildCount(node.id)}',
                        style: const TextStyle(
                          fontSize: 9,
                          color: Colors.orange,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
        for (final child in children)
          _buildNode(context, tree, child, depth + 1),
      ],
    );
  }
}
