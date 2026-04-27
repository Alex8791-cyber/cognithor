import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class TemplatePicker extends StatefulWidget {
  const TemplatePicker({super.key, required this.subreddit});
  final String subreddit;

  @override
  State<TemplatePicker> createState() => _TemplatePickerState();
}

class _TemplatePickerState extends State<TemplatePicker> {
  List<Map<String, dynamic>> _templates = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final templates = await context.read<RedditLeadsProvider>().getTemplates(
      subreddit: widget.subreddit,
    );
    if (mounted) {
      setState(() {
        _templates = templates;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return DraggableScrollableSheet(
      initialChildSize: 0.5,
      minChildSize: 0.3,
      maxChildSize: 0.8,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
          ),
          child: _loading
              ? const Center(child: CircularProgressIndicator())
              : _templates.isEmpty
              ? Center(child: Text(l.noTemplates))
              : ListView.builder(
                  controller: scrollController,
                  padding: const EdgeInsets.all(16),
                  itemCount: _templates.length,
                  itemBuilder: (context, i) {
                    final t = _templates[i];
                    return ListTile(
                      title: Text(t['name']?.toString() ?? ''),
                      subtitle: Text(
                        (t['template_text']?.toString() ?? '').replaceAll(
                          '\n',
                          ' ',
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      trailing: Text(
                        '${t['use_count'] ?? 0}x',
                        style: TextStyle(
                          color: CognithorTheme.textSecondary,
                          fontSize: 11,
                        ),
                      ),
                      onTap: () => Navigator.of(
                        context,
                      ).pop(t['template_text']?.toString() ?? ''),
                    );
                  },
                ),
        );
      },
    );
  }
}
