import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/agent_editor_screen.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_chip.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';

class AgentsScreen extends StatefulWidget {
  const AgentsScreen({super.key});

  @override
  State<AgentsScreen> createState() => _AgentsScreenState();
}

class _AgentsScreenState extends State<AgentsScreen> {
  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _loadData();
    }
  }

  void _loadData() {
    final admin = context.read<AdminProvider>();
    admin.setApi(context.read<ConnectionProvider>().api);
    admin.loadAgents();
  }

  Future<void> _openEditor(String? agentName) async {
    final result = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => AgentEditorScreen(agentName: agentName),
      ),
    );
    if (result == true && mounted) {
      context.read<AdminProvider>().loadAgents();
    }
  }

  Future<void> _toggleAgent(Map<String, dynamic> agent, bool enabled) async {
    final admin = context.read<AdminProvider>();
    final name = agent['name']?.toString();
    if (name == null) return;
    await admin.updateAgent(name, {'enabled': enabled});
  }

  Future<void> _deleteAgent(Map<String, dynamic> agent) async {
    final name = agent['name']?.toString();
    if (name == null) return;
    final l = AppLocalizations.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l.delete),
        content: Text('${l.confirmDeleteAgent}\n\nAgent: $name'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l.cancel),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: TextButton.styleFrom(foregroundColor: CognithorTheme.red),
            child: Text(l.delete),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      await context.read<AdminProvider>().deleteAgent(name);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final admin = context.watch<AdminProvider>();

    if (admin.isLoading && admin.agents.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(l.loading),
          ],
        ),
      );
    }

    if (admin.error != null && admin.agents.isEmpty) {
      return CognithorEmptyState(
        icon: Icons.smart_toy_outlined,
        title: l.agentsTitle,
        subtitle: admin.error,
        action: ElevatedButton.icon(
          onPressed: () => context.read<AdminProvider>().loadAgents(),
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    if (admin.agents.isEmpty) {
      return CognithorEmptyState(
        icon: Icons.smart_toy_outlined,
        title: l.noAgents,
        action: ElevatedButton.icon(
          onPressed: () => context.read<AdminProvider>().loadAgents(),
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    return Scaffold(
      body: RefreshIndicator(
        onRefresh: () => context.read<AdminProvider>().loadAgents(),
        color: CognithorTheme.accent,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Summary
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                CognithorStat(
                  label: l.agentsTitle,
                  value: admin.agents.length.toString(),
                  icon: Icons.smart_toy,
                  color: CognithorTheme.accent,
                ),
              ],
            ),
            const SizedBox(height: 16),

            CognithorSection(title: l.agentsTitle),
            ...admin.agents.map<Widget>((agent) {
              final a = agent as Map<String, dynamic>? ?? {};
              return _AgentCard(
                agent: a,
                onEdit: () => _openEditor(a['name']?.toString()),
                onToggle: (enabled) => _toggleAgent(a, enabled),
                onDelete: () => _deleteAgent(a),
              );
            }),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _openEditor(null),
        icon: const Icon(Icons.add),
        label: Text(l.newAgent),
        backgroundColor: CognithorTheme.sectionAdmin,
      ),
    );
  }
}

class _AgentCard extends StatelessWidget {
  const _AgentCard({
    required this.agent,
    this.onEdit,
    this.onToggle,
    this.onDelete,
  });

  final Map<String, dynamic> agent;
  final VoidCallback? onEdit;
  final ValueChanged<bool>? onToggle;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    final name = agent['name']?.toString() ?? '';
    final displayName = agent['display_name']?.toString() ?? name;
    final description = agent['description']?.toString() ?? '';
    final model = agent['preferred_model']?.toString() ?? '';
    final temperature = agent['temperature']?.toString() ?? '';
    final enabled = agent['enabled'] as bool? ?? true;
    final priority = agent['priority']?.toString() ?? '';
    final allowed = agent['allowed_tools'] as List<dynamic>? ?? [];
    final blocked = agent['blocked_tools'] as List<dynamic>? ?? [];

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: NeonCard(
        tint: CognithorTheme.sectionAdmin,
        glowOnHover: true,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(
                  Icons.smart_toy,
                  size: 18,
                  color: CognithorTheme.sectionAdmin,
                ),
                const SizedBox(width: 8),
                Expanded(child: Text(name, style: theme.textTheme.titleMedium)),
                Switch(
                  value: enabled,
                  activeTrackColor: CognithorTheme.sectionAdmin,
                  onChanged: onToggle,
                ),
                const SizedBox(width: 4),
                IconButton(
                  icon: const Icon(Icons.edit_outlined, size: 18),
                  onPressed: onEdit,
                  tooltip: l.editAgent,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(
                    minWidth: 32,
                    minHeight: 32,
                  ),
                ),
                IconButton(
                  icon: Icon(
                    Icons.delete_outline,
                    size: 18,
                    color: CognithorTheme.red,
                  ),
                  onPressed: onDelete,
                  tooltip: l.delete,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(
                    minWidth: 32,
                    minHeight: 32,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (displayName != name)
              Text(displayName, style: theme.textTheme.bodyMedium),
            if (description.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(description, style: theme.textTheme.bodySmall),
            ],
            const SizedBox(height: 8),
            Wrap(
              spacing: 12,
              runSpacing: 4,
              children: [
                if (model.isNotEmpty)
                  _infoChip(l.model, model, Icons.psychology),
                if (temperature.isNotEmpty)
                  _infoChip(l.temperature, temperature, Icons.thermostat),
                if (priority.isNotEmpty)
                  _infoChip(l.priority, priority, Icons.low_priority),
              ],
            ),
            if (allowed.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(l.allowedTools, style: theme.textTheme.bodySmall),
              const SizedBox(height: 4),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: allowed
                    .map<Widget>(
                      (t) => CognithorChip(
                        label: t.toString(),
                        color: CognithorTheme.green,
                      ),
                    )
                    .toList(),
              ),
            ],
            if (blocked.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(l.blockedTools, style: theme.textTheme.bodySmall),
              const SizedBox(height: 4),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: blocked
                    .map<Widget>(
                      (t) => CognithorChip(
                        label: t.toString(),
                        color: CognithorTheme.red,
                      ),
                    )
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _infoChip(String label, String value, IconData icon) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: CognithorTheme.textSecondary),
        const SizedBox(width: 4),
        Text(
          '$label: $value',
          style: TextStyle(color: CognithorTheme.textSecondary, fontSize: 12),
        ),
      ],
    );
  }
}
