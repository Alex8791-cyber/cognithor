import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_toast.dart';

/// Full-screen skill editor for creating and editing skills.
class SkillEditorScreen extends StatefulWidget {
  const SkillEditorScreen({super.key, this.slug});

  /// If null, creating a new skill. If set, editing existing.
  final String? slug;

  @override
  State<SkillEditorScreen> createState() => _SkillEditorScreenState();
}

class _SkillEditorScreenState extends State<SkillEditorScreen> {
  final _formKey = GlobalKey<FormState>();

  // Controllers
  final _nameCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _keywordsCtrl = TextEditingController();
  final _toolsCtrl = TextEditingController();
  final _priorityCtrl = TextEditingController(text: '5');
  final _modelCtrl = TextEditingController();
  final _bodyCtrl = TextEditingController();

  // State
  String _category = 'general';
  bool _enabled = true;
  bool _isLoading = true;
  bool _isSaving = false;
  bool _isDirty = false;
  bool _isBuiltIn = false;

  // Stats (read-only, for existing skills)
  int _totalUses = 0;
  double _successRate = 0.0;
  String? _lastUsed;

  // Original data for dirty tracking
  Map<String, dynamic>? _originalData;

  static const _categories = [
    'general',
    'productivity',
    'research',
    'analysis',
    'development',
    'automation',
  ];

  bool get _isEditing => widget.slug != null;

  @override
  void initState() {
    super.initState();
    if (_isEditing) {
      _loadSkill();
    } else {
      _isLoading = false;
    }

    // Track changes for dirty state
    _nameCtrl.addListener(_markDirty);
    _descCtrl.addListener(_markDirty);
    _keywordsCtrl.addListener(_markDirty);
    _toolsCtrl.addListener(_markDirty);
    _priorityCtrl.addListener(_markDirty);
    _modelCtrl.addListener(_markDirty);
    _bodyCtrl.addListener(_markDirty);
  }

  void _markDirty() {
    if (!_isDirty) {
      setState(() => _isDirty = true);
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _descCtrl.dispose();
    _keywordsCtrl.dispose();
    _toolsCtrl.dispose();
    _priorityCtrl.dispose();
    _modelCtrl.dispose();
    _bodyCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadSkill() async {
    final provider = context.read<SkillsProvider>();
    final api = context.read<ConnectionProvider>().api;
    provider.setApi(api);

    final data = await provider.getSkillDetail(widget.slug!);
    if (data == null || !mounted) return;

    _originalData = data;
    _nameCtrl.text = data['name']?.toString() ?? '';
    _descCtrl.text = data['description']?.toString() ?? '';
    _category = data['category']?.toString() ?? 'general';
    if (!_categories.contains(_category)) _category = 'general';
    _keywordsCtrl.text = _listToCommaString(
      data['trigger_keywords'] ?? data['keywords'],
    );
    _toolsCtrl.text = _listToCommaString(
      data['required_tools'] ?? data['tools'],
    );
    _priorityCtrl.text = (data['priority'] ?? 5).toString();
    _modelCtrl.text = data['model_preference']?.toString() ?? '';
    _enabled = data['enabled'] as bool? ?? true;
    _bodyCtrl.text = data['body']?.toString() ?? '';
    _isBuiltIn = data['source']?.toString() == 'builtin';

    // Stats
    _totalUses = (data['total_uses'] as num?)?.toInt() ?? 0;
    _successRate = (data['success_rate'] as num?)?.toDouble() ?? 0.0;
    _lastUsed = data['last_used']?.toString();

    setState(() {
      _isLoading = false;
      _isDirty = false;
    });
  }

  String _listToCommaString(dynamic value) {
    if (value == null) return '';
    if (value is List) return value.map((e) => e.toString()).join(', ');
    return value.toString();
  }

  List<String> _commaStringToList(String value) {
    if (value.trim().isEmpty) return [];
    return value
        .split(',')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();
  }

  Map<String, dynamic> _buildPayload() {
    return {
      'name': _nameCtrl.text.trim(),
      'description': _descCtrl.text.trim(),
      'category': _category,
      'trigger_keywords': _commaStringToList(_keywordsCtrl.text),
      'required_tools': _commaStringToList(_toolsCtrl.text),
      'priority': int.tryParse(_priorityCtrl.text) ?? 5,
      'model_preference': _modelCtrl.text.trim().isEmpty
          ? null
          : _modelCtrl.text.trim(),
      'enabled': _enabled,
      'body': _bodyCtrl.text,
    };
  }

  Future<void> _save() async {
    if (!(_formKey.currentState?.validate() ?? false)) return;

    setState(() => _isSaving = true);

    final provider = context.read<SkillsProvider>();
    final api = context.read<ConnectionProvider>().api;
    provider.setApi(api);

    final payload = _buildPayload();
    final l = AppLocalizations.of(context);
    bool success;

    if (_isEditing) {
      success = await provider.updateSkill(widget.slug!, payload);
    } else {
      success = await provider.createSkill(payload);
    }

    if (!mounted) return;
    setState(() => _isSaving = false);

    if (success) {
      _isDirty = false;
      CognithorToast.show(
        context,
        _isEditing ? l.skillSaved : l.skillCreated,
        type: ToastType.success,
      );
      Navigator.of(context).pop(true);
    } else {
      CognithorToast.show(
        context,
        provider.error ?? 'Error',
        type: ToastType.error,
      );
    }
  }

  Future<void> _delete() async {
    final l = AppLocalizations.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.deleteSkill),
        content: Text(l.confirmDeleteSkill),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: ElevatedButton.styleFrom(
              backgroundColor: CognithorTheme.red,
            ),
            child: Text(l.delete),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;

    final provider = context.read<SkillsProvider>();
    final success = await provider.deleteSkill(widget.slug!);

    if (!mounted) return;

    if (success) {
      _isDirty = false;
      CognithorToast.show(context, l.skillDeleted, type: ToastType.success);
      Navigator.of(context).pop(true);
    } else {
      CognithorToast.show(
        context,
        provider.error ?? 'Error',
        type: ToastType.error,
      );
    }
  }

  Future<void> _export() async {
    final provider = context.read<SkillsProvider>();
    final md = await provider.exportSkill(widget.slug!);
    if (!mounted) return;
    final l = AppLocalizations.of(context);

    if (md != null) {
      await Clipboard.setData(ClipboardData(text: md));
      if (!mounted) return;
      CognithorToast.show(context, l.skillExported, type: ToastType.success);
    } else {
      CognithorToast.show(context, 'Export failed', type: ToastType.error);
    }
  }

  Future<bool> _onWillPop() async {
    if (!_isDirty) return true;
    final l = AppLocalizations.of(context);
    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.discardChanges),
        content: Text(l.discardChangesBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: Text(l.discard),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    return PopScope(
      canPop: !_isDirty,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        final nav = Navigator.of(context);
        final shouldPop = await _onWillPop();
        if (shouldPop && mounted) {
          nav.pop();
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text(
            _isEditing
                ? (_nameCtrl.text.isNotEmpty ? _nameCtrl.text : l.editSkill)
                : l.newSkill,
          ),
          actions: [
            if (_isEditing)
              IconButton(
                onPressed: _export,
                icon: const Icon(Icons.file_download_outlined),
                tooltip: l.exportSkillMd,
              ),
            const SizedBox(width: 4),
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: FilledButton.icon(
                onPressed: _isSaving ? null : _save,
                icon: _isSaving
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.save_outlined, size: 18),
                label: Text(l.save),
              ),
            ),
          ],
        ),
        body: _isLoading
            ? const Center(child: CircularProgressIndicator())
            : _buildForm(l, theme),
      ),
    );
  }

  Widget _buildForm(AppLocalizations l, ThemeData theme) {
    return Form(
      key: _formKey,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Built-in notice
          if (_isBuiltIn) ...[
            NeonCard(
              tint: CognithorTheme.orange,
              child: Row(
                children: [
                  Icon(
                    Icons.lock_outline,
                    color: CognithorTheme.orange,
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      l.builtInSkill,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: CognithorTheme.orange,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
          ],

          // ── Metadata Section ──────────────────────────────────
          _SectionHeader(title: l.metadata, icon: Icons.info_outline),
          const SizedBox(height: 8),
          NeonCard(
            tint: CognithorTheme.sectionSkills,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Name
                TextFormField(
                  controller: _nameCtrl,
                  decoration: InputDecoration(
                    labelText: l.skillName,
                    prefixIcon: const Icon(Icons.label_outline),
                  ),
                  readOnly: _isBuiltIn,
                  validator: (v) =>
                      (v == null || v.trim().isEmpty) ? 'Required' : null,
                ),
                const SizedBox(height: 16),

                // Description
                TextFormField(
                  controller: _descCtrl,
                  decoration: InputDecoration(
                    labelText: l.description,
                    prefixIcon: const Icon(Icons.description_outlined),
                  ),
                  readOnly: _isBuiltIn,
                  maxLines: 3,
                  minLines: 2,
                ),
                const SizedBox(height: 16),

                // Category
                DropdownButtonFormField<String>(
                  initialValue: _category,
                  decoration: const InputDecoration(
                    labelText: 'Category',
                    prefixIcon: Icon(Icons.category_outlined),
                  ),
                  items: _categories.map((c) {
                    return DropdownMenuItem(
                      value: c,
                      child: Text(_categoryLabel(l, c)),
                    );
                  }).toList(),
                  onChanged: _isBuiltIn
                      ? null
                      : (v) {
                          setState(() {
                            _category = v ?? 'general';
                            _isDirty = true;
                          });
                        },
                ),
                const SizedBox(height: 16),

                // Trigger Keywords
                TextFormField(
                  controller: _keywordsCtrl,
                  decoration: InputDecoration(
                    labelText: l.triggerKeywords,
                    hintText: l.commaSeparated,
                    prefixIcon: const Icon(Icons.tag),
                  ),
                  readOnly: _isBuiltIn,
                ),
                if (_keywordsCtrl.text.trim().isNotEmpty) ...[
                  const SizedBox(height: 8),
                  _ChipsPreview(
                    values: _commaStringToList(_keywordsCtrl.text),
                    color: CognithorTheme.sectionSkills,
                  ),
                ],
                const SizedBox(height: 16),

                // Required Tools
                TextFormField(
                  controller: _toolsCtrl,
                  decoration: InputDecoration(
                    labelText: l.requiredTools,
                    hintText: l.commaSeparated,
                    prefixIcon: const Icon(Icons.build_outlined),
                  ),
                  readOnly: _isBuiltIn,
                ),
                if (_toolsCtrl.text.trim().isNotEmpty) ...[
                  const SizedBox(height: 8),
                  _ChipsPreview(
                    values: _commaStringToList(_toolsCtrl.text),
                    color: CognithorTheme.accent,
                  ),
                ],
                const SizedBox(height: 16),

                // Priority + Model Preference row
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(
                      width: 120,
                      child: TextFormField(
                        controller: _priorityCtrl,
                        decoration: InputDecoration(
                          labelText: l.priority,
                          prefixIcon: const Icon(Icons.low_priority, size: 20),
                        ),
                        readOnly: _isBuiltIn,
                        keyboardType: TextInputType.number,
                        inputFormatters: [
                          FilteringTextInputFormatter.digitsOnly,
                        ],
                        validator: (v) {
                          final n = int.tryParse(v ?? '');
                          if (n == null || n < 0 || n > 10) {
                            return '0-10';
                          }
                          return null;
                        },
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: TextFormField(
                        controller: _modelCtrl,
                        decoration: InputDecoration(
                          labelText: l.modelPreference,
                          prefixIcon: const Icon(Icons.memory_outlined),
                        ),
                        readOnly: _isBuiltIn,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // Enabled switch
                SwitchListTile(
                  title: Text(l.enabled),
                  value: _enabled,
                  activeThumbColor: CognithorTheme.sectionSkills,
                  contentPadding: EdgeInsets.zero,
                  onChanged: _isBuiltIn
                      ? null
                      : (v) {
                          setState(() {
                            _enabled = v;
                            _isDirty = true;
                          });
                        },
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // ── Body Section ──────────────────────────────────────
          _SectionHeader(title: l.skillBody, icon: Icons.code),
          const SizedBox(height: 8),
          NeonCard(
            tint: CognithorTheme.sectionSkills,
            child: TextFormField(
              controller: _bodyCtrl,
              decoration: InputDecoration(
                hintText: l.skillBodyHint,
                border: InputBorder.none,
                enabledBorder: InputBorder.none,
                focusedBorder: InputBorder.none,
              ),
              readOnly: _isBuiltIn,
              maxLines: null,
              minLines: 15,
              style: CognithorTheme.monoTextTheme.bodyMedium?.copyWith(
                fontSize: 13,
                height: 1.6,
              ),
            ),
          ),

          // ── Stats Section (existing skills only) ──────────────
          if (_isEditing && _originalData != null) ...[
            const SizedBox(height: 24),
            _SectionHeader(title: l.statistics, icon: Icons.analytics_outlined),
            const SizedBox(height: 8),
            NeonCard(
              tint: CognithorTheme.sectionSkills,
              child: Row(
                children: [
                  _StatTile(
                    label: l.totalUses,
                    value: _totalUses.toString(),
                    icon: Icons.touch_app_outlined,
                  ),
                  const SizedBox(width: 24),
                  _StatTile(
                    label: l.successRate,
                    value: '${(_successRate * 100).toStringAsFixed(1)}%',
                    icon: Icons.check_circle_outline,
                  ),
                  const SizedBox(width: 24),
                  _StatTile(
                    label: l.lastUsed,
                    value: _lastUsed ?? '-',
                    icon: Icons.schedule,
                  ),
                ],
              ),
            ),
          ],

          // ── Delete Button (user-created only) ─────────────────
          if (_isEditing && !_isBuiltIn) ...[
            const SizedBox(height: 32),
            Center(
              child: OutlinedButton.icon(
                onPressed: _delete,
                icon: Icon(Icons.delete_outline, color: CognithorTheme.red),
                label: Text(l.deleteSkill),
                style: OutlinedButton.styleFrom(
                  foregroundColor: CognithorTheme.red,
                  side: BorderSide(
                    color: CognithorTheme.red.withValues(alpha: 0.5),
                  ),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                ),
              ),
            ),
          ],

          const SizedBox(height: 48),
        ],
      ),
    );
  }

  String _categoryLabel(AppLocalizations l, String cat) {
    return switch (cat) {
      'general' => l.general,
      'productivity' => l.productivity,
      'research' => l.research,
      'analysis' => l.analysis,
      'development' => l.development,
      'automation' => l.automation,
      _ => cat,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private Widgets
// ─────────────────────────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.title, required this.icon});

  final String title;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 18, color: CognithorTheme.sectionSkills),
        const SizedBox(width: 8),
        Text(
          title,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
            color: CognithorTheme.sectionSkills,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

class _ChipsPreview extends StatelessWidget {
  const _ChipsPreview({required this.values, required this.color});

  final List<String> values;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 6,
      runSpacing: 4,
      children: values.map((v) {
        return Chip(
          label: Text(v, style: const TextStyle(fontSize: 11)),
          backgroundColor: color.withValues(alpha: 0.15),
          side: BorderSide(color: color.withValues(alpha: 0.3)),
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          padding: const EdgeInsets.symmetric(horizontal: 4),
          visualDensity: VisualDensity.compact,
        );
      }).toList(),
    );
  }
}

class _StatTile extends StatelessWidget {
  const _StatTile({
    required this.label,
    required this.value,
    required this.icon,
  });

  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 14, color: CognithorTheme.textSecondary),
              const SizedBox(width: 4),
              Text(label, style: theme.textTheme.bodySmall),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: theme.textTheme.bodyLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
