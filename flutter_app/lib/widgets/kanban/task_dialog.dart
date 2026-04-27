import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';

/// Dialog for creating or editing a Kanban task.
class TaskDialog extends StatefulWidget {
  final String? initialTitle;
  final String? initialDescription;
  final String? initialPriority;
  final String? initialAgent;
  final List<String>? initialLabels;
  final List<String> availableAgents;

  const TaskDialog({
    super.key,
    this.initialTitle,
    this.initialDescription,
    this.initialPriority,
    this.initialAgent,
    this.initialLabels,
    this.availableAgents = const [],
  });

  @override
  State<TaskDialog> createState() => _TaskDialogState();
}

class _TaskDialogState extends State<TaskDialog> {
  late final TextEditingController _titleCtrl;
  late final TextEditingController _descCtrl;
  late final TextEditingController _labelsCtrl;
  String _priority = 'medium';
  String _agent = '';

  List<String> get _agents => ['', ...widget.availableAgents];
  static const _priorities = ['low', 'medium', 'high', 'urgent'];

  @override
  void initState() {
    super.initState();
    _titleCtrl = TextEditingController(text: widget.initialTitle ?? '');
    _descCtrl = TextEditingController(text: widget.initialDescription ?? '');
    _labelsCtrl = TextEditingController(
      text: (widget.initialLabels ?? []).join(', '),
    );
    _priority = widget.initialPriority ?? 'medium';
    _agent = widget.initialAgent ?? '';
  }

  @override
  void dispose() {
    _titleCtrl.dispose();
    _descCtrl.dispose();
    _labelsCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return AlertDialog(
      title: Text(
        widget.initialTitle != null ? l.kanbanEditTask : l.kanbanNewTask,
      ),
      content: SizedBox(
        width: 400,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: _titleCtrl,
                decoration: const InputDecoration(
                  labelText: 'Title',
                  border: OutlineInputBorder(),
                ),
                autofocus: true,
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _descCtrl,
                decoration: const InputDecoration(
                  labelText: 'Description',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: DropdownButtonFormField<String>(
                      initialValue: _priority,
                      decoration: const InputDecoration(
                        labelText: 'Priority',
                        border: OutlineInputBorder(),
                      ),
                      items: _priorities
                          .map(
                            (p) => DropdownMenuItem(value: p, child: Text(p)),
                          )
                          .toList(),
                      onChanged: (v) =>
                          setState(() => _priority = v ?? 'medium'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: DropdownButtonFormField<String>(
                      initialValue: _agent,
                      decoration: const InputDecoration(
                        labelText: 'Agent',
                        border: OutlineInputBorder(),
                      ),
                      items: _agents
                          .map(
                            (a) => DropdownMenuItem(
                              value: a,
                              child: Text(a.isEmpty ? '(none)' : a),
                            ),
                          )
                          .toList(),
                      onChanged: (v) => setState(() => _agent = v ?? ''),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _labelsCtrl,
                decoration: const InputDecoration(
                  labelText: 'Labels (comma separated)',
                  border: OutlineInputBorder(),
                  hintText: 'bug, research, urgent',
                ),
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(l.cancel),
        ),
        FilledButton(
          onPressed: () {
            if (_titleCtrl.text.trim().isEmpty) return;
            final labels = _labelsCtrl.text
                .split(',')
                .map((s) => s.trim())
                .where((s) => s.isNotEmpty)
                .toList();
            Navigator.of(context).pop({
              'title': _titleCtrl.text.trim(),
              'description': _descCtrl.text.trim(),
              'priority': _priority,
              'assigned_agent': _agent,
              'labels': labels,
            });
          },
          child: Text(
            widget.initialTitle != null ? l.kanbanSave : l.kanbanCreate,
          ),
        ),
      ],
    );
  }
}
