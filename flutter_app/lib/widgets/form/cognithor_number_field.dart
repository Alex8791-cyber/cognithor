import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorNumberField extends StatefulWidget {
  const CognithorNumberField({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.description,
    this.min,
    this.max,
    this.step = 1,
    this.decimal = false,
    this.error,
  });

  final String label;
  final num value;
  final ValueChanged<num> onChanged;
  final String? description;
  final num? min;
  final num? max;
  final num step;
  final bool decimal;
  final String? error;

  @override
  State<CognithorNumberField> createState() => _CognithorNumberFieldState();
}

class _CognithorNumberFieldState extends State<CognithorNumberField> {
  late final TextEditingController _ctrl;
  String? _localError;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(text: _format(widget.value));
  }

  @override
  void didUpdateWidget(CognithorNumberField old) {
    super.didUpdateWidget(old);
    final formatted = _format(widget.value);
    if (old.value != widget.value && _ctrl.text != formatted) {
      _ctrl.text = formatted;
    }
  }

  String _format(num v) =>
      widget.decimal ? v.toStringAsFixed(2) : v.toInt().toString();

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _onChanged(String text) {
    final parsed = widget.decimal ? double.tryParse(text) : int.tryParse(text);
    if (parsed == null) {
      setState(() => _localError = 'Invalid number');
      return;
    }
    if (widget.min != null && parsed < widget.min!) {
      setState(() => _localError = 'Min: ${widget.min}');
      return;
    }
    if (widget.max != null && parsed > widget.max!) {
      setState(() => _localError = 'Max: ${widget.max}');
      return;
    }
    setState(() => _localError = null);
    widget.onChanged(parsed);
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
          TextField(
            controller: _ctrl,
            keyboardType: TextInputType.numberWithOptions(
              decimal: widget.decimal,
              signed: true,
            ),
            inputFormatters: [
              FilteringTextInputFormatter.allow(
                RegExp(widget.decimal ? r'[\d.\-]' : r'[\d\-]'),
              ),
            ],
            decoration: InputDecoration(
              errorText: widget.error ?? _localError,
              isDense: true,
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 10,
              ),
            ),
            onChanged: _onChanged,
          ),
        ],
      ),
    );
  }
}
