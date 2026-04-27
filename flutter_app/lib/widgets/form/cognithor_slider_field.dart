import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorSliderField extends StatefulWidget {
  const CognithorSliderField({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.min = 0.0,
    this.max = 1.0,
    this.step = 0.01,
    this.description,
  });

  final String label;
  final double value;
  final ValueChanged<double> onChanged;
  final double min;
  final double max;
  final double step;
  final String? description;

  @override
  State<CognithorSliderField> createState() => _CognithorSliderFieldState();
}

class _CognithorSliderFieldState extends State<CognithorSliderField> {
  bool _editing = false;
  late TextEditingController _ctrl;

  double get _clampedValue => widget.value.clamp(widget.min, widget.max);

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final clamped = _clampedValue;
    final divisions = ((widget.max - widget.min) / widget.step).round();

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
              if (_editing)
                SizedBox(
                  width: 60,
                  child: TextField(
                    controller: _ctrl,
                    autofocus: true,
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    style: theme.textTheme.bodySmall,
                    decoration: const InputDecoration(
                      isDense: true,
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 4,
                      ),
                    ),
                    onSubmitted: (text) {
                      final v = double.tryParse(text);
                      if (v != null) {
                        widget.onChanged(v.clamp(widget.min, widget.max));
                      }
                      setState(() => _editing = false);
                    },
                  ),
                )
              else
                GestureDetector(
                  onTap: () {
                    _ctrl.text = clamped.toStringAsFixed(2);
                    setState(() => _editing = true);
                  },
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: theme.cardColor,
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: theme.dividerColor),
                    ),
                    child: Text(
                      clamped.toStringAsFixed(2),
                      style: theme.textTheme.bodySmall?.copyWith(
                        fontFamily: 'monospace',
                      ),
                    ),
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
          Slider(
            value: clamped,
            min: widget.min,
            max: widget.max,
            divisions: divisions > 0 ? divisions : null,
            onChanged: widget.onChanged,
          ),
        ],
      ),
    );
  }
}
