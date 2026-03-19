import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/admin_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
import 'package:jarvis_ui/widgets/jarvis_toast.dart';

/// Full-screen agent editor for creating and editing agent profiles.
class AgentEditorScreen extends StatefulWidget {
  const AgentEditorScreen({super.key, this.agentName});

  /// If null, creating new. If set, editing existing.
  final String? agentName;

  @override
  State<AgentEditorScreen> createState() => _AgentEditorScreenState();
}

class _AgentEditorScreenState extends State<AgentEditorScreen> {
  final _formKey = GlobalKey<FormState>();

  // Controllers
  final _nameCtrl = TextEditingController();
  final _displayNameCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _modelCtrl = TextEditingController();
  final _priorityCtrl = TextEditingController(text: '0');
  final _systemPromptCtrl = TextEditingController();
  final _allowedToolsCtrl = TextEditingController();
  final _blockedToolsCtrl = TextEditingController();
  final _sandboxTimeoutCtrl = TextEditingController(text: '30');

  // State
  String _language = 'en';
  double _temperature = 0.7;
  bool _enabled = true;
  String _sandboxNetwork = 'allow';
  bool _isLoading = true;
  bool _isSaving = false;
  bool _isDirty = false;

  static const _languages = ['en', 'de', 'zh', 'ar'];
  static const _networkOptions = ['allow', 'deny', 'restricted'];

  bool get _isEditing => widget.agentName != null;

  @override
  void initState() {
    super.initState();
    if (_isEditing) {
      _loadAgent();
    } else {
      _isLoading = false;
    }

    _nameCtrl.addListener(_markDirty);
    _displayNameCtrl.addListener(_markDirty);
    _descCtrl.addListener(_markDirty);
    _modelCtrl.addListener(_markDirty);
    _priorityCtrl.addListener(_markDirty);
    _systemPromptCtrl.addListener(_markDirty);
    _allowedToolsCtrl.addListener(_markDirty);
    _blockedToolsCtrl.addListener(_markDirty);
    _sandboxTimeoutCtrl.addListener(_markDirty);
  }

  Future<void> _showModelPicker(BuildContext context) async {
    // Fetch available models from backend
    List<String> models = [];
    try {
      final api = context.read<ConnectionProvider>().api;
      final data = await api.get('models/available');
      final raw = data['models'];
      if (raw is List) {
        models = raw.map((m) => m.toString()).toList()..sort();
      }
    } catch (_) {}

    if (!mounted || models.isEmpty) return;

    final selected = await showDialog<String>(
      context: context,
      builder: (ctx) {
        String search = '';
        return StatefulBuilder(
          builder: (ctx, setState) {
            final filtered = search.isEmpty
                ? models
                : models.where((m) => m.toLowerCase().contains(search.toLowerCase())).toList();
            return AlertDialog(
              title: const Text('Select Model'),
              content: SizedBox(
                width: 400,
                height: 500,
                child: Column(
                  children: [
                    TextField(
                      decoration: const InputDecoration(
                        hintText: 'Search models...',
                        prefixIcon: Icon(Icons.search, size: 20),
                        isDense: true,
                      ),
                      onChanged: (v) => setState(() => search = v),
                    ),
                    const SizedBox(height: 12),
                    Expanded(
                      child: ListView.builder(
                        itemCount: filtered.length,
                        itemBuilder: (ctx, i) {
                          final name = filtered[i];
                          final isCurrent = name == _modelCtrl.text;
                          return ListTile(
                            dense: true,
                            selected: isCurrent,
                            selectedColor: JarvisTheme.sectionAdmin,
                            leading: Icon(
                              isCurrent ? Icons.check_circle : Icons.circle_outlined,
                              size: 18,
                              color: isCurrent ? JarvisTheme.sectionAdmin : null,
                            ),
                            title: Text(name, style: const TextStyle(fontSize: 13)),
                            onTap: () => Navigator.pop(ctx, name),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Cancel'),
                ),
              ],
            );
          },
        );
      },
    );

    if (selected != null) {
      setState(() {
        _modelCtrl.text = selected;
        _isDirty = true;
      });
    }
  }

  void _markDirty() {
    if (!_isDirty) {
      setState(() => _isDirty = true);
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _displayNameCtrl.dispose();
    _descCtrl.dispose();
    _modelCtrl.dispose();
    _priorityCtrl.dispose();
    _systemPromptCtrl.dispose();
    _allowedToolsCtrl.dispose();
    _blockedToolsCtrl.dispose();
    _sandboxTimeoutCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadAgent() async {
    final admin = context.read<AdminProvider>();
    admin.setApi(context.read<ConnectionProvider>().api);

    final data = await admin.getAgent(widget.agentName!);
    if (data == null || !mounted) return;

    _nameCtrl.text = data['name']?.toString() ?? '';
    _displayNameCtrl.text = data['display_name']?.toString() ?? '';
    _descCtrl.text = data['description']?.toString() ?? '';
    _modelCtrl.text = data['preferred_model']?.toString() ?? '';
    _priorityCtrl.text = (data['priority'] ?? 0).toString();
    _systemPromptCtrl.text = data['system_prompt']?.toString() ?? '';
    _allowedToolsCtrl.text =
        _listToCommaString(data['allowed_tools']);
    _blockedToolsCtrl.text =
        _listToCommaString(data['blocked_tools']);
    _sandboxTimeoutCtrl.text = (data['sandbox_timeout'] ?? 30).toString();

    final lang = data['language']?.toString() ?? 'en';
    _language = _languages.contains(lang) ? lang : 'en';
    _temperature = (data['temperature'] as num?)?.toDouble() ?? 0.7;
    _enabled = data['enabled'] as bool? ?? true;
    final net = data['sandbox_network']?.toString() ?? 'allow';
    _sandboxNetwork = _networkOptions.contains(net) ? net : 'allow';

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
      'name': _nameCtrl.text.trim().toLowerCase().replaceAll(' ', '-'),
      'display_name': _displayNameCtrl.text.trim(),
      'description': _descCtrl.text.trim(),
      'system_prompt': _systemPromptCtrl.text,
      'language': _language,
      'preferred_model': _modelCtrl.text.trim(),
      'temperature': _temperature,
      'priority': int.tryParse(_priorityCtrl.text) ?? 0,
      'enabled': _enabled,
      'allowed_tools': _commaStringToList(_allowedToolsCtrl.text),
      'blocked_tools': _commaStringToList(_blockedToolsCtrl.text),
      'sandbox_timeout': int.tryParse(_sandboxTimeoutCtrl.text) ?? 30,
      'sandbox_network': _sandboxNetwork,
    };
  }

  Future<void> _save() async {
    if (!(_formKey.currentState?.validate() ?? false)) return;

    setState(() => _isSaving = true);

    final admin = context.read<AdminProvider>();
    admin.setApi(context.read<ConnectionProvider>().api);

    final payload = _buildPayload();
    final l = AppLocalizations.of(context);
    bool success;

    if (_isEditing) {
      success = await admin.updateAgent(widget.agentName!, payload);
    } else {
      success = await admin.createAgent(payload);
    }

    if (!mounted) return;
    setState(() => _isSaving = false);

    if (success) {
      _isDirty = false;
      JarvisToast.show(
        context,
        _isEditing ? l.agentSaved : l.agentCreated,
        type: ToastType.success,
      );
      Navigator.of(context).pop(true);
    } else {
      JarvisToast.show(
        context,
        admin.error ?? 'Error',
        type: ToastType.error,
      );
    }
  }

  Future<void> _delete() async {
    final l = AppLocalizations.of(context);

    if (widget.agentName == 'jarvis') {
      JarvisToast.show(context, l.cannotDeleteDefault, type: ToastType.error);
      return;
    }

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.deleteAgent),
        content: Text(l.confirmDeleteAgent),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l.cancel),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: ElevatedButton.styleFrom(
              backgroundColor: JarvisTheme.red,
            ),
            child: Text(l.delete),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;

    final admin = context.read<AdminProvider>();
    final success = await admin.deleteAgent(widget.agentName!);

    if (!mounted) return;

    if (success) {
      _isDirty = false;
      JarvisToast.show(context, l.agentDeleted, type: ToastType.success);
      Navigator.of(context).pop(true);
    } else {
      JarvisToast.show(
        context,
        admin.error ?? 'Error',
        type: ToastType.error,
      );
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
                ? (_displayNameCtrl.text.isNotEmpty
                    ? _displayNameCtrl.text
                    : l.editAgent)
                : l.newAgent,
          ),
          actions: [
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
          // -- Identity Section --
          _SectionHeader(title: l.metadata, icon: Icons.info_outline),
          const SizedBox(height: 8),
          NeonCard(
            tint: JarvisTheme.sectionAdmin,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Name
                TextFormField(
                  controller: _nameCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Name',
                    prefixIcon: Icon(Icons.label_outline),
                    hintText: 'my-agent',
                  ),
                  readOnly: _isEditing,
                  enabled: !_isEditing,
                  validator: (v) =>
                      (v == null || v.trim().isEmpty) ? 'Required' : null,
                ),
                const SizedBox(height: 16),

                // Display Name
                TextFormField(
                  controller: _displayNameCtrl,
                  decoration: InputDecoration(
                    labelText: l.displayName,
                    prefixIcon: const Icon(Icons.badge_outlined),
                  ),
                ),
                const SizedBox(height: 16),

                // Description
                TextFormField(
                  controller: _descCtrl,
                  decoration: InputDecoration(
                    labelText: l.description,
                    prefixIcon: const Icon(Icons.description_outlined),
                  ),
                  maxLines: 3,
                  minLines: 2,
                ),
                const SizedBox(height: 16),

                // Language
                DropdownButtonFormField<String>(
                  initialValue: _language,
                  decoration: InputDecoration(
                    labelText: l.language,
                    prefixIcon: const Icon(Icons.language),
                  ),
                  items: _languages.map((lang) {
                    return DropdownMenuItem(
                      value: lang,
                      child: Text(_languageLabel(lang)),
                    );
                  }).toList(),
                  onChanged: (v) {
                    setState(() {
                      _language = v ?? 'en';
                      _isDirty = true;
                    });
                  },
                ),
                const SizedBox(height: 16),

                // Preferred Model (tap to pick from available)
                GestureDetector(
                  onTap: () => _showModelPicker(context),
                  child: AbsorbPointer(
                    child: TextFormField(
                      controller: _modelCtrl,
                      decoration: InputDecoration(
                        labelText: l.preferredModel,
                        prefixIcon: const Icon(Icons.memory_outlined),
                        suffixIcon: const Icon(Icons.arrow_drop_down),
                        hintText: 'Tap to select...',
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // Temperature slider
                Row(
                  children: [
                    Text(
                      '${l.temperature}: ${_temperature.toStringAsFixed(1)}',
                      style: theme.textTheme.bodyMedium,
                    ),
                  ],
                ),
                Slider(
                  value: _temperature,
                  min: 0.0,
                  max: 2.0,
                  divisions: 20,
                  activeColor: JarvisTheme.sectionAdmin,
                  label: _temperature.toStringAsFixed(1),
                  onChanged: (v) {
                    setState(() {
                      _temperature = v;
                      _isDirty = true;
                    });
                  },
                ),
                const SizedBox(height: 8),

                // Priority + Enabled row
                Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    SizedBox(
                      width: 120,
                      child: TextFormField(
                        controller: _priorityCtrl,
                        decoration: InputDecoration(
                          labelText: l.priority,
                          prefixIcon:
                              const Icon(Icons.low_priority, size: 20),
                        ),
                        keyboardType: TextInputType.number,
                        inputFormatters: [
                          FilteringTextInputFormatter.digitsOnly,
                        ],
                        validator: (v) {
                          final n = int.tryParse(v ?? '');
                          if (n == null || n < 0 || n > 100) {
                            return '0-100';
                          }
                          return null;
                        },
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: SwitchListTile(
                        title: Text(l.enabled),
                        value: _enabled,
                        activeThumbColor: JarvisTheme.sectionAdmin,
                        contentPadding: EdgeInsets.zero,
                        onChanged: (v) {
                          setState(() {
                            _enabled = v;
                            _isDirty = true;
                          });
                        },
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // -- System Prompt Section --
          _SectionHeader(
            title: l.systemPrompt,
            icon: Icons.terminal,
          ),
          const SizedBox(height: 8),
          NeonCard(
            tint: JarvisTheme.sectionAdmin,
            child: TextFormField(
              controller: _systemPromptCtrl,
              decoration: InputDecoration(
                hintText: l.systemPrompt,
                border: InputBorder.none,
                enabledBorder: InputBorder.none,
                focusedBorder: InputBorder.none,
              ),
              maxLines: null,
              minLines: 10,
              style: JarvisTheme.monoTextTheme.bodyMedium?.copyWith(
                fontSize: 13,
                height: 1.6,
              ),
            ),
          ),

          const SizedBox(height: 24),

          // -- Tools Section --
          _SectionHeader(
            title: l.allowedTools,
            icon: Icons.build_outlined,
          ),
          const SizedBox(height: 8),
          NeonCard(
            tint: JarvisTheme.sectionAdmin,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextFormField(
                  controller: _allowedToolsCtrl,
                  decoration: InputDecoration(
                    labelText: l.allowedTools,
                    hintText: l.commaSeparated,
                    prefixIcon: const Icon(Icons.check_circle_outline),
                  ),
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _blockedToolsCtrl,
                  decoration: InputDecoration(
                    labelText: l.blockedTools,
                    hintText: l.commaSeparated,
                    prefixIcon: const Icon(Icons.block),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // -- Sandbox Section --
          _SectionHeader(
            title: l.sandboxTimeout,
            icon: Icons.security,
          ),
          const SizedBox(height: 8),
          NeonCard(
            tint: JarvisTheme.sectionAdmin,
            child: Row(
              children: [
                SizedBox(
                  width: 160,
                  child: TextFormField(
                    controller: _sandboxTimeoutCtrl,
                    decoration: InputDecoration(
                      labelText: l.sandboxTimeout,
                      prefixIcon: const Icon(Icons.timer, size: 20),
                    ),
                    keyboardType: TextInputType.number,
                    inputFormatters: [
                      FilteringTextInputFormatter.digitsOnly,
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: DropdownButtonFormField<String>(
                    initialValue: _sandboxNetwork,
                    decoration: InputDecoration(
                      labelText: l.sandboxNetwork,
                      prefixIcon: const Icon(Icons.wifi),
                    ),
                    items: _networkOptions.map((opt) {
                      return DropdownMenuItem(
                        value: opt,
                        child: Text(opt),
                      );
                    }).toList(),
                    onChanged: (v) {
                      setState(() {
                        _sandboxNetwork = v ?? 'allow';
                        _isDirty = true;
                      });
                    },
                  ),
                ),
              ],
            ),
          ),

          // -- Delete Button (non-jarvis only) --
          if (_isEditing && widget.agentName != 'jarvis') ...[
            const SizedBox(height: 32),
            Center(
              child: OutlinedButton.icon(
                onPressed: _delete,
                icon: Icon(Icons.delete_outline, color: JarvisTheme.red),
                label: Text(l.deleteAgent),
                style: OutlinedButton.styleFrom(
                  foregroundColor: JarvisTheme.red,
                  side: BorderSide(
                      color: JarvisTheme.red.withValues(alpha: 0.5)),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 24, vertical: 12),
                ),
              ),
            ),
          ],

          const SizedBox(height: 48),
        ],
      ),
    );
  }

  String _languageLabel(String code) {
    return switch (code) {
      'en' => 'English',
      'de' => 'Deutsch',
      'zh' => 'Chinese',
      'ar' => 'Arabic',
      _ => code,
    };
  }
}

// ---------------------------------------------------------------------------
// Private Widgets
// ---------------------------------------------------------------------------

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.title, required this.icon});

  final String title;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 18, color: JarvisTheme.sectionAdmin),
        const SizedBox(width: 8),
        Text(
          title,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: JarvisTheme.sectionAdmin,
                fontWeight: FontWeight.w600,
              ),
        ),
      ],
    );
  }
}
