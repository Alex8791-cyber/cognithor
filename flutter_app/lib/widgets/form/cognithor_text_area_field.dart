import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorTextAreaField extends StatefulWidget {
  const CognithorTextAreaField({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.description,
    this.rows = 6,
    this.mono = true,
    this.error,
    this.onReset,
    this.resetLabel,
  });

  final String label;
  final String value;
  final ValueChanged<String> onChanged;
  final String? description;
  final int rows;
  final bool mono;
  final String? error;
  final VoidCallback? onReset;
  final String? resetLabel;

  @override
  State<CognithorTextAreaField> createState() => _CognithorTextAreaFieldState();
}

class _CognithorTextAreaFieldState extends State<CognithorTextAreaField> {
  late final TextEditingController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(text: widget.value);
  }

  @override
  void didUpdateWidget(CognithorTextAreaField old) {
    super.didUpdateWidget(old);
    if (old.value != widget.value && _ctrl.text != widget.value) {
      _ctrl.text = widget.value;
    }
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
          Row(
            children: [
              Expanded(
                child: Text(widget.label, style: theme.textTheme.bodyMedium),
              ),
              if (widget.onReset != null)
                TextButton.icon(
                  onPressed: widget.onReset,
                  icon: const Icon(Icons.restore, size: 14),
                  label: Text(
                    widget.resetLabel ?? 'Reset',
                    style: const TextStyle(fontSize: 12),
                  ),
                ),
            ],
          ),
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
          TextField(
            controller: _ctrl,
            maxLines: widget.rows,
            style: widget.mono
                ? theme.textTheme.bodyMedium?.copyWith(
                    fontFamily: 'monospace',
                    fontSize: 13,
                  )
                : null,
            decoration: InputDecoration(
              errorText: widget.error,
              isDense: true,
              contentPadding: const EdgeInsets.all(12),
            ),
            onChanged: widget.onChanged,
          ),
        ],
      ),
    );
  }
}
