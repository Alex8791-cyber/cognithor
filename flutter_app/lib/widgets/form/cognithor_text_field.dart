import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class CognithorTextField extends StatefulWidget {
  const CognithorTextField({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.description,
    this.placeholder,
    this.isPassword = false,
    this.isSecret = false,
    this.mono = false,
    this.error,
    this.enabled = true,
  });

  final String label;
  final String value;
  final ValueChanged<String> onChanged;
  final String? description;
  final String? placeholder;
  final bool isPassword;
  final bool isSecret;
  final bool mono;
  final String? error;
  final bool enabled;

  @override
  State<CognithorTextField> createState() => _CognithorTextFieldState();
}

class _CognithorTextFieldState extends State<CognithorTextField> {
  static const _maskedDisplay =
      '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022'; // 24 bullets

  late final TextEditingController _ctrl;
  bool _obscured = true;

  /// True when the field shows a backend-masked placeholder (not the real key).
  bool get _isMasked => _ctrl.text == '***' || _ctrl.text == _maskedDisplay;

  @override
  void initState() {
    super.initState();
    // Replace short '***' mask with a longer visual placeholder
    final initial = (widget.isSecret && widget.value == '***')
        ? _maskedDisplay
        : widget.value;
    _ctrl = TextEditingController(text: initial);
  }

  @override
  void didUpdateWidget(CognithorTextField old) {
    super.didUpdateWidget(old);
    // Only sync if the value changed externally (not from user typing)
    final effective = (widget.isSecret && widget.value == '***')
        ? _maskedDisplay
        : widget.value;
    if (old.value != widget.value && _ctrl.text != effective) {
      final sel = _ctrl.selection;
      _ctrl.text = effective;
      // Restore cursor if possible
      if (sel.isValid && sel.baseOffset <= _ctrl.text.length) {
        _ctrl.selection = sel;
      }
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
              Text(widget.label, style: theme.textTheme.bodyMedium),
              if (widget.isSecret && widget.value == '***') ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: CognithorTheme.green.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    AppLocalizations.of(context).saved,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: CognithorTheme.green,
                      fontSize: 10,
                    ),
                  ),
                ),
              ],
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
            enabled: widget.enabled,
            obscureText: widget.isPassword && _obscured,
            onTap: () {
              // Clear masked placeholder so user can type a new value
              if (_isMasked) {
                _ctrl.clear();
              }
            },
            style: widget.mono
                ? theme.textTheme.bodyMedium?.copyWith(fontFamily: 'monospace')
                : null,
            decoration: InputDecoration(
              hintText: widget.placeholder,
              errorText: widget.error,
              isDense: true,
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 10,
              ),
              // Hide eye button when value is backend-masked (not revealable)
              suffixIcon: widget.isPassword && !_isMasked
                  ? IconButton(
                      icon: Icon(
                        _obscured ? Icons.visibility_off : Icons.visibility,
                        size: 18,
                      ),
                      onPressed: () => setState(() => _obscured = !_obscured),
                    )
                  : null,
            ),
            onChanged: widget.onChanged,
          ),
        ],
      ),
    );
  }
}
