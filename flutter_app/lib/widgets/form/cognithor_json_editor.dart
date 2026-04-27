import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorJsonEditor extends StatefulWidget {
  const CognithorJsonEditor({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.description,
    this.rows = 8,
  });

  final String label;
  final dynamic value;
  final ValueChanged<dynamic> onChanged;
  final String? description;
  final int rows;

  @override
  State<CognithorJsonEditor> createState() => _CognithorJsonEditorState();
}

class _CognithorJsonEditorState extends State<CognithorJsonEditor> {
  late final TextEditingController _ctrl;
  late final FocusNode _focusNode;
  String? _error;
  bool _hasFocus = false;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(text: _encode(widget.value));
    _focusNode = FocusNode();
    _focusNode.addListener(_onFocusChanged);
  }

  void _onFocusChanged() {
    setState(() => _hasFocus = _focusNode.hasFocus);
  }

  @override
  void didUpdateWidget(CognithorJsonEditor old) {
    super.didUpdateWidget(old);
    // Do not update controller text while the user is actively editing
    // to prevent cursor jumps.
    if (_hasFocus) return;
    final encoded = _encode(widget.value);
    if (_ctrl.text != encoded && _error == null) {
      _ctrl.text = encoded;
    }
  }

  String _encode(dynamic v) {
    try {
      const encoder = JsonEncoder.withIndent('  ');
      return encoder.convert(v);
    } catch (_) {
      return v?.toString() ?? '';
    }
  }

  void _onChanged(String text) {
    try {
      final parsed = jsonDecode(text);
      setState(() => _error = null);
      widget.onChanged(parsed);
    } catch (e) {
      setState(() => _error = e.toString().split('\n').first);
    }
  }

  @override
  void dispose() {
    _focusNode.removeListener(_onFocusChanged);
    _focusNode.dispose();
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
          TextField(
            controller: _ctrl,
            focusNode: _focusNode,
            maxLines: widget.rows,
            style: theme.textTheme.bodyMedium?.copyWith(
              fontFamily: 'monospace',
              fontSize: 13,
            ),
            decoration: InputDecoration(
              errorText: _error,
              errorMaxLines: 2,
              isDense: true,
              contentPadding: const EdgeInsets.all(12),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(
                  color: _error != null
                      ? CognithorTheme.red
                      : theme.dividerColor,
                ),
              ),
            ),
            onChanged: _onChanged,
          ),
        ],
      ),
    );
  }
}
