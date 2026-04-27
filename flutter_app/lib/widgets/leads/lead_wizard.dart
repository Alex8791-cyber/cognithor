import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/leads/refine_panel.dart';
import 'package:cognithor_ui/widgets/leads/template_picker.dart';
import 'package:cognithor_ui/widgets/cognithor_toast.dart';

class LeadWizard extends StatefulWidget {
  const LeadWizard({super.key, required this.leads});
  final List<RedditLead> leads;

  @override
  State<LeadWizard> createState() => _LeadWizardState();
}

class _LeadWizardState extends State<LeadWizard> {
  int _currentIndex = 0;
  int _repliedCount = 0;
  int _skippedCount = 0;
  int _archivedCount = 0;
  late TextEditingController _replyCtrl;
  final FocusNode _replyFocusNode = FocusNode();
  bool _posting = false;
  bool _showRefine = false;

  RedditLead get _currentLead => widget.leads[_currentIndex];
  bool get _isDone => _currentIndex >= widget.leads.length;
  double get _progress =>
      widget.leads.isEmpty ? 1.0 : (_currentIndex / widget.leads.length);

  @override
  void initState() {
    super.initState();
    _replyCtrl = TextEditingController(
      text: widget.leads.isNotEmpty ? widget.leads[0].effectiveReply : '',
    );
    WidgetsBinding.instance.addPostFrameCallback((_) => _preloadAhead());
  }

  void _preloadAhead() {
    if (!mounted) return;
    final provider = context.read<RedditLeadsProvider>();
    for (var i = 1; i <= 2; i++) {
      final idx = _currentIndex + i;
      if (idx < widget.leads.length) {
        provider.preloadPerformance(widget.leads[idx].id);
      }
    }
  }

  @override
  void dispose() {
    _replyCtrl.dispose();
    _replyFocusNode.dispose();
    super.dispose();
  }

  void _advance() {
    final next = _currentIndex + 1;
    if (next >= widget.leads.length) {
      setState(() => _currentIndex = next);
      return;
    }
    setState(() {
      _currentIndex = next;
      _replyCtrl.text = widget.leads[next].effectiveReply;
      _showRefine = false;
    });
    _preloadAhead();
  }

  Future<void> _reply() async {
    setState(() => _posting = true);
    final provider = context.read<RedditLeadsProvider>();
    if (_replyCtrl.text != _currentLead.effectiveReply) {
      await provider.updateLead(_currentLead.id, replyFinal: _replyCtrl.text);
    }
    final ok = await provider.replyToLead(_currentLead.id);
    setState(() => _posting = false);
    if (ok) {
      _repliedCount++;
      if (mounted)
        CognithorToast.show(context, 'Reply posted', type: ToastType.success);
      _advance();
    }
  }

  void _skip() {
    _skippedCount++;
    _advance();
  }

  Future<void> _archive() async {
    await context.read<RedditLeadsProvider>().updateLead(
      _currentLead.id,
      status: 'archived',
    );
    _archivedCount++;
    _advance();
  }

  Future<void> _pickTemplate() async {
    final text = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      builder: (_) => TemplatePicker(subreddit: _currentLead.subreddit),
    );
    if (text != null && text.isNotEmpty && mounted) {
      setState(() => _replyCtrl.text = text);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    if (_isDone) return _buildSummary(l, theme);

    final lead = _currentLead;

    return Scaffold(
      appBar: AppBar(
        title: Text('Lead ${_currentIndex + 1}/${widget.leads.length}'),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => Navigator.of(context).pop(),
        ),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(4),
          child: LinearProgressIndicator(
            value: _progress,
            color: CognithorTheme.accent,
          ),
        ),
      ),
      body: KeyboardListener(
        focusNode: FocusNode()..requestFocus(),
        onKeyEvent: (event) {
          // Only handle shortcuts when reply TextField is NOT focused
          if (event is KeyDownEvent && !_replyFocusNode.hasFocus) {
            if (event.logicalKey == LogicalKeyboardKey.keyA) _archive();
            if (event.logicalKey == LogicalKeyboardKey.keyS) _skip();
            if (event.logicalKey == LogicalKeyboardKey.keyR && !_posting)
              _reply();
            if (event.logicalKey == LogicalKeyboardKey.keyI)
              setState(() => _showRefine = !_showRefine);
          }
        },
        child: Column(
          children: [
            Expanded(
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 150),
                transitionBuilder: (child, anim) =>
                    FadeTransition(opacity: anim, child: child),
                child: ListView(
                  key: ValueKey<int>(_currentIndex),
                  padding: const EdgeInsets.all(16),
                  children: [
                    // Score + subreddit
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: CognithorTheme.accent.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            '${lead.intentScore}/100',
                            style: TextStyle(
                              color: CognithorTheme.accent,
                              fontWeight: FontWeight.w800,
                              fontSize: 18,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          'r/${lead.subreddit}',
                          style: theme.textTheme.titleSmall,
                        ),
                        const Spacer(),
                        Text(lead.timeAgo, style: theme.textTheme.bodySmall),
                      ],
                    ),
                    const SizedBox(height: 12),

                    // Title + body
                    Text(lead.title, style: theme.textTheme.titleMedium),
                    if (lead.body.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text(lead.body, style: theme.textTheme.bodyMedium),
                    ],
                    const SizedBox(height: 8),
                    if (lead.scoreReason.isNotEmpty)
                      Text(
                        'Reason: ${lead.scoreReason}',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: CognithorTheme.textSecondary,
                        ),
                      ),
                    const Divider(height: 24),

                    // Reply editor
                    TextField(
                      controller: _replyCtrl,
                      focusNode: _replyFocusNode,
                      maxLines: 6,
                      decoration: InputDecoration(
                        labelText: l.editReply,
                        border: const OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 8),

                    // Tool buttons
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        OutlinedButton.icon(
                          onPressed: () =>
                              setState(() => _showRefine = !_showRefine),
                          icon: const Icon(Icons.auto_fix_high, size: 16),
                          label: Text(l.improve),
                        ),
                        OutlinedButton.icon(
                          onPressed: _pickTemplate,
                          icon: const Icon(Icons.description, size: 16),
                          label: Text(l.useTemplate),
                        ),
                      ],
                    ),

                    // Refine panel
                    if (_showRefine) ...[
                      const SizedBox(height: 12),
                      RefinePanel(
                        leadId: lead.id,
                        currentDraft: _replyCtrl.text,
                        onAccept: (text) {
                          setState(() {
                            _replyCtrl.text = text;
                            _showRefine = false;
                          });
                        },
                      ),
                    ],
                    const SizedBox(height: 24),
                  ],
                ),
              ),
            ),
            // Sticky footer with action buttons
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                color: theme.colorScheme.surface,
                border: Border(
                  top: BorderSide(color: theme.dividerColor, width: 0.5),
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      TextButton.icon(
                        onPressed: _archive,
                        icon: Icon(
                          Icons.archive,
                          size: 16,
                          color: CognithorTheme.textSecondary,
                        ),
                        label: Text(
                          l.archiveLead,
                          style: TextStyle(color: CognithorTheme.textSecondary),
                        ),
                      ),
                      const Spacer(),
                      OutlinedButton.icon(
                        onPressed: _skip,
                        icon: const Icon(Icons.skip_next, size: 16),
                        label: Text(l.skipLead),
                      ),
                      const SizedBox(width: 8),
                      ElevatedButton.icon(
                        onPressed: _posting ? null : _reply,
                        icon: _posting
                            ? const SizedBox(
                                width: 14,
                                height: 14,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : const Icon(Icons.reply, size: 16),
                        label: Text(l.postReply),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Shortcuts: A=Archive  S=Skip  I=Improve  R=Reply',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: CognithorTheme.textSecondary,
                      fontSize: 10,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummary(AppLocalizations l, ThemeData theme) {
    return Scaffold(
      appBar: AppBar(title: Text(l.wizardComplete)),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.check_circle, size: 64, color: Colors.green),
            const SizedBox(height: 16),
            Text(l.wizardComplete, style: theme.textTheme.headlineSmall),
            const SizedBox(height: 8),
            Text(
              l.wizardSummary(_repliedCount, _skippedCount, _archivedCount),
              style: theme.textTheme.bodyLarge,
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text(l.confirm),
            ),
          ],
        ),
      ),
    );
  }
}
