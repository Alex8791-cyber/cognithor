import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/admin_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
import 'package:jarvis_ui/widgets/jarvis_chip.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';
import 'package:jarvis_ui/widgets/jarvis_stat.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';

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
      return JarvisEmptyState(
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
      return JarvisEmptyState(
        icon: Icons.smart_toy_outlined,
        title: l.noAgents,
        action: ElevatedButton.icon(
          onPressed: () => context.read<AdminProvider>().loadAgents(),
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: () => context.read<AdminProvider>().loadAgents(),
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Summary
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              JarvisStat(
                label: l.agentsTitle,
                value: admin.agents.length.toString(),
                icon: Icons.smart_toy,
                color: JarvisTheme.accent,
              ),
            ],
          ),
          const SizedBox(height: 16),

          JarvisSection(title: l.agentsTitle),
          ...admin.agents.map<Widget>((agent) {
            final a = agent as Map<String, dynamic>? ?? {};
            return _AgentCard(agent: a);
          }),
        ],
      ),
    );
  }
}

class _AgentCard extends StatelessWidget {
  const _AgentCard({required this.agent});

  final Map<String, dynamic> agent;

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
        tint: JarvisTheme.sectionAdmin,
        glowOnHover: true,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.smart_toy, size: 18, color: JarvisTheme.sectionAdmin),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(name, style: theme.textTheme.titleMedium),
                ),
                JarvisStatusBadge(
                  label: enabled ? l.enabled : l.disabled,
                  color: enabled ? JarvisTheme.green : JarvisTheme.red,
                  icon: enabled ? Icons.check_circle : Icons.cancel,
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
                      (t) => JarvisChip(
                        label: t.toString(),
                        color: JarvisTheme.green,
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
                      (t) => JarvisChip(
                        label: t.toString(),
                        color: JarvisTheme.red,
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
        Icon(icon, size: 14, color: JarvisTheme.textSecondary),
        const SizedBox(width: 4),
        Text(
          '$label: $value',
          style: TextStyle(
            color: JarvisTheme.textSecondary,
            fontSize: 12,
          ),
        ),
      ],
    );
  }
}
