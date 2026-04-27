import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class SelectOption {
  const SelectOption({required this.value, required this.label});
  final String value;
  final String label;
}

class CognithorSelectField extends StatelessWidget {
  const CognithorSelectField({
    super.key,
    required this.label,
    required this.value,
    required this.options,
    required this.onChanged,
    this.description,
  });

  final String label;
  final String value;
  final List<SelectOption> options;
  final ValueChanged<String> onChanged;
  final String? description;

  /// Convenience: create from a simple list of strings.
  factory CognithorSelectField.fromStrings({
    Key? key,
    required String label,
    required String value,
    required List<String> options,
    required ValueChanged<String> onChanged,
    String? description,
  }) {
    return CognithorSelectField(
      key: key,
      label: label,
      value: value,
      options: options.map((s) => SelectOption(value: s, label: s)).toList(),
      onChanged: onChanged,
      description: description,
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: theme.textTheme.bodyMedium),
          if (description != null) ...[
            const SizedBox(height: 2),
            Text(
              description!,
              style: theme.textTheme.bodySmall?.copyWith(
                color: CognithorTheme.textSecondary,
              ),
            ),
          ],
          const SizedBox(height: 6),
          InputDecorator(
            decoration: const InputDecoration(
              isDense: true,
              contentPadding: EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 10,
              ),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: options.any((o) => o.value == value) ? value : null,
                isExpanded: true,
                isDense: true,
                items: options
                    .map(
                      (o) => DropdownMenuItem(
                        value: o.value,
                        child: Text(o.label),
                      ),
                    )
                    .toList(),
                onChanged: (v) {
                  if (v != null) onChanged(v);
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}
