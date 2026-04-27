import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorListField extends StatefulWidget {
  const CognithorListField({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.description,
    this.placeholder,
  });

  final String label;
  final List<String> value;
  final ValueChanged<List<String>> onChanged;
  final String? description;
  final String? placeholder;

  @override
  State<CognithorListField> createState() => _CognithorListFieldState();
}

class _CognithorListFieldState extends State<CognithorListField> {
  final _ctrl = TextEditingController();

  void _add() {
    final text = _ctrl.text.trim();
    if (text.isEmpty) return;
    widget.onChanged([...widget.value, text]);
    _ctrl.clear();
  }

  void _remove(int index) {
    final copy = List<String>.from(widget.value);
    copy.removeAt(index);
    widget.onChanged(copy);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(widget.label, style: theme.textTheme.bodyMedium),
          if (widget.description != null) ...[
            const SizedBox(height: 2),
            Text(
              widget.description!,
              style: theme.textTheme.bodySmall?.copyWith(
                color: CognithorTheme.textSecondary,
              ),
            ),
          ],
          const SizedBox(height: 6),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _ctrl,
                  decoration: InputDecoration(
                    hintText: widget.placeholder ?? 'Add item...',
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                  ),
                  onSubmitted: (_) => _add(),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                icon: Icon(Icons.add, color: CognithorTheme.accent),
                onPressed: _add,
                tooltip: AppLocalizations.of(context).addItem,
              ),
            ],
          ),
          const SizedBox(height: 4),
          ...List.generate(widget.value.length, (i) {
            return Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: theme.cardColor,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: theme.dividerColor),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        widget.value[i],
                        style: theme.textTheme.bodySmall?.copyWith(
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                    InkWell(
                      onTap: () => _remove(i),
                      child: Icon(
                        Icons.close,
                        size: 16,
                        color: CognithorTheme.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }),
        ],
      ),
    );
  }
}
