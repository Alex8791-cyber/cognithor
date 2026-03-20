import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

class JarvisTextField extends StatefulWidget {
  const JarvisTextField({
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
  State<JarvisTextField> createState() => _JarvisTextFieldState();
}

class _JarvisTextFieldState extends State<JarvisTextField> {
  late final TextEditingController _ctrl;
  bool _obscured = true;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(text: widget.value);
  }

  @override
  void didUpdateWidget(JarvisTextField old) {
    super.didUpdateWidget(old);
    // Only sync if the value changed externally (not from user typing)
    if (old.value != widget.value && _ctrl.text != widget.value) {
      final sel = _ctrl.selection;
      _ctrl.text = widget.value;
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
                  padding:
                      const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: JarvisTheme.green.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(AppLocalizations.of(context).saved,
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: JarvisTheme.green, fontSize: 10)),
                ),
              ],
            ],
          ),
          if (widget.description != null) ...[
            const SizedBox(height: 2),
            Text(widget.description!,
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: JarvisTheme.textSecondary)),
          ],
          const SizedBox(height: 6),
          TextField(
            controller: _ctrl,
            enabled: widget.enabled,
            obscureText: widget.isPassword && _obscured,
            onTap: () {
              // Clear masked placeholder so user can type a new value
              if (_ctrl.text == '***') {
                _ctrl.clear();
              }
            },
            style: widget.mono
                ? theme.textTheme.bodyMedium
                    ?.copyWith(fontFamily: 'monospace')
                : null,
            decoration: InputDecoration(
              hintText: widget.placeholder,
              errorText: widget.error,
              isDense: true,
              contentPadding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              suffixIcon: widget.isPassword
                  ? IconButton(
                      icon: Icon(
                        _obscured ? Icons.visibility_off : Icons.visibility,
                        size: 18,
                      ),
                      // Disable eye toggle when value is masked from backend
                      onPressed: _ctrl.text == '***'
                          ? null
                          : () => setState(() => _obscured = !_obscured),
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
