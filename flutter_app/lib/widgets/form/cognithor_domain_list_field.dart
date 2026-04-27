import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorDomainListField extends StatefulWidget {
  const CognithorDomainListField({
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
  State<CognithorDomainListField> createState() =>
      _CognithorDomainListFieldState();
}

class _CognithorDomainListFieldState extends State<CognithorDomainListField> {
  final _ctrl = TextEditingController();
  String? _error;

  static final _domainRegex = RegExp(
    r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$',
  );

  String? _validate(String text) {
    if (text.startsWith('http://') || text.startsWith('https://')) {
      return 'No scheme (http://) allowed';
    }
    if (text.contains('/')) return 'No paths allowed';
    if (text.contains('*')) return 'No wildcards allowed';
    if (!_domainRegex.hasMatch(text)) return 'Invalid domain format';
    return null;
  }

  void _add() {
    final text = _ctrl.text.trim();
    if (text.isEmpty) return;
    final err = _validate(text);
    if (err != null) {
      setState(() => _error = err);
      return;
    }
    setState(() => _error = null);
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
                    hintText: widget.placeholder ?? 'example.com',
                    errorText: _error,
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                  ),
                  onSubmitted: (_) => _add(),
                  onChanged: (_) {
                    if (_error != null) setState(() => _error = null);
                  },
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                icon: Icon(Icons.add, color: CognithorTheme.accent),
                onPressed: _add,
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
