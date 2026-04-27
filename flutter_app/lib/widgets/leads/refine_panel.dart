import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class RefinePanel extends StatefulWidget {
  const RefinePanel({
    super.key,
    required this.leadId,
    required this.currentDraft,
    required this.onAccept,
  });

  final String leadId;
  final String currentDraft;
  final ValueChanged<String> onAccept;

  @override
  State<RefinePanel> createState() => _RefinePanelState();
}

class _RefinePanelState extends State<RefinePanel> {
  final _hintCtrl = TextEditingController();
  bool _refining = false;
  String? _refinedText;
  List<Map<String, dynamic>> _variants = [];
  int _selectedVariant = -1;

  @override
  void dispose() {
    _hintCtrl.dispose();
    super.dispose();
  }

  Future<void> _refine() async {
    setState(() {
      _refining = true;
      _refinedText = null;
      _variants = [];
    });
    final result = await context.read<RedditLeadsProvider>().refineLead(
      widget.leadId,
      hint: _hintCtrl.text,
    );
    setState(() {
      _refining = false;
      _refinedText = result['text']?.toString();
    });
  }

  Future<void> _generateVariants() async {
    setState(() {
      _refining = true;
      _variants = [];
      _refinedText = null;
      _selectedVariant = -1;
    });
    final result = await context.read<RedditLeadsProvider>().refineLead(
      widget.leadId,
      variants: 3,
    );
    setState(() {
      _refining = false;
      _variants =
          (result['variants'] as List<dynamic>?)
              ?.cast<Map<String, dynamic>>() ??
          [];
    });
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Hint input
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _hintCtrl,
                decoration: const InputDecoration(
                  hintText: 'e.g. "make it shorter", "more technical"',
                  isDense: true,
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 8,
                  ),
                  border: OutlineInputBorder(),
                ),
                style: const TextStyle(fontSize: 12),
              ),
            ),
            const SizedBox(width: 8),
            ElevatedButton(
              onPressed: _refining ? null : _refine,
              child: _refining
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Text(l.improve),
            ),
            const SizedBox(width: 4),
            OutlinedButton(
              onPressed: _refining ? null : _generateVariants,
              child: Text(l.variants),
            ),
          ],
        ),

        // Refined result
        if (_refinedText != null) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: CognithorTheme.green.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: CognithorTheme.green.withValues(alpha: 0.2),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Improved:',
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: CognithorTheme.green,
                  ),
                ),
                const SizedBox(height: 4),
                Text(_refinedText!, style: theme.textTheme.bodySmall),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerRight,
                  child: ElevatedButton.icon(
                    onPressed: () => widget.onAccept(_refinedText!),
                    icon: const Icon(Icons.check, size: 16),
                    label: const Text('Accept'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: CognithorTheme.green,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],

        // Variants
        if (_variants.isNotEmpty) ...[
          const SizedBox(height: 12),
          ...List.generate(_variants.length, (i) {
            final v = _variants[i];
            final isSelected = _selectedVariant == i;
            return GestureDetector(
              onTap: () => setState(() => _selectedVariant = i),
              child: Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: isSelected
                      ? CognithorTheme.accent.withValues(alpha: 0.1)
                      : null,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: isSelected
                        ? CognithorTheme.accent
                        : CognithorTheme.textSecondary.withValues(alpha: 0.2),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      v['style']?.toString().toUpperCase() ?? '',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: CognithorTheme.accent,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      v['text']?.toString() ?? '',
                      style: theme.textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
            );
          }),
          if (_selectedVariant >= 0)
            Align(
              alignment: Alignment.centerRight,
              child: ElevatedButton.icon(
                onPressed: () => widget.onAccept(
                  _variants[_selectedVariant]['text']?.toString() ?? '',
                ),
                icon: const Icon(Icons.check, size: 16),
                label: const Text('Use this variant'),
              ),
            ),
        ],
      ],
    );
  }
}
