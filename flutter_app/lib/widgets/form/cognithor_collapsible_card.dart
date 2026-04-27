import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorCollapsibleCard extends StatefulWidget {
  const CognithorCollapsibleCard({
    super.key,
    required this.title,
    required this.children,
    this.initiallyExpanded = false,
    this.forceOpen = false,
    this.badge,
    this.icon,
  });

  final String title;
  final List<Widget> children;
  final bool initiallyExpanded;
  final bool forceOpen;
  final String? badge;
  final IconData? icon;

  @override
  State<CognithorCollapsibleCard> createState() =>
      _CognithorCollapsibleCardState();
}

class _CognithorCollapsibleCardState extends State<CognithorCollapsibleCard> {
  late bool _expanded;

  @override
  void initState() {
    super.initState();
    _expanded = widget.initiallyExpanded || widget.forceOpen;
  }

  @override
  void didUpdateWidget(CognithorCollapsibleCard old) {
    super.didUpdateWidget(old);
    if (widget.forceOpen && !_expanded) _expanded = true;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isOpen = _expanded || widget.forceOpen;

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: theme.cardColor,
        borderRadius: BorderRadius.circular(CognithorTheme.cardRadius),
        border: Border.all(color: theme.dividerColor),
      ),
      child: Column(
        children: [
          InkWell(
            borderRadius: BorderRadius.only(
              topLeft: const Radius.circular(CognithorTheme.cardRadius),
              topRight: const Radius.circular(CognithorTheme.cardRadius),
              bottomLeft: Radius.circular(
                isOpen ? 0 : CognithorTheme.cardRadius,
              ),
              bottomRight: Radius.circular(
                isOpen ? 0 : CognithorTheme.cardRadius,
              ),
            ),
            onTap: widget.forceOpen
                ? null
                : () => setState(() => _expanded = !_expanded),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  if (widget.icon != null) ...[
                    Icon(widget.icon, size: 18, color: CognithorTheme.accent),
                    const SizedBox(width: 8),
                  ],
                  Expanded(
                    child: Text(
                      widget.title,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  if (widget.badge != null) ...[
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: CognithorTheme.accent.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        widget.badge!,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: CognithorTheme.accent,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                  ],
                  if (!widget.forceOpen)
                    Icon(
                      isOpen ? Icons.expand_less : Icons.expand_more,
                      size: 20,
                      color: theme.brightness == Brightness.dark
                          ? CognithorTheme.textSecondary
                          : const Color(0xFF6B6B80),
                    ),
                ],
              ),
            ),
          ),
          if (isOpen)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: widget.children,
              ),
            ),
        ],
      ),
    );
  }
}
