import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/admin_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/jarvis_card.dart';
import 'package:jarvis_ui/widgets/jarvis_chip.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';
import 'package:jarvis_ui/widgets/jarvis_stat.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';

class ModelsScreen extends StatefulWidget {
  const ModelsScreen({super.key});

  @override
  State<ModelsScreen> createState() => _ModelsScreenState();
}

class _ModelsScreenState extends State<ModelsScreen> {
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
    admin.loadModels();
    admin.loadModelStats();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final admin = context.watch<AdminProvider>();

    if (admin.isLoading && admin.models == null) {
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

    if (admin.error != null && admin.models == null) {
      return JarvisEmptyState(
        icon: Icons.model_training,
        title: l.modelsTitle,
        subtitle: admin.error,
        action: ElevatedButton.icon(
          onPressed: () {
            context.read<AdminProvider>().loadModels();
            context.read<AdminProvider>().loadModelStats();
          },
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    final modelsData = admin.models ?? {};
    final configured = modelsData['configured'] as Map<String, dynamic>? ?? {};
    final available = modelsData['available'] as List<dynamic>? ?? [];
    final warnings = modelsData['warnings'] as List<dynamic>? ?? [];

    final stats = admin.modelStats ?? {};
    final totalModels = stats['total']?.toString() ?? available.length.toString();
    final providerCount = stats['providers']?.toString() ?? '-';
    final capCount = stats['capabilities']?.toString() ?? '-';

    final planner = configured['planner'] as Map<String, dynamic>? ?? {};
    final executor = configured['executor'] as Map<String, dynamic>? ?? {};
    final coder = configured['coder'] as Map<String, dynamic>? ?? {};
    final embedding = configured['embedding'] as Map<String, dynamic>? ?? {};

    return RefreshIndicator(
      onRefresh: () async {
        final a = context.read<AdminProvider>();
        await Future.wait([a.loadModels(), a.loadModelStats()]);
      },
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Configured models
          JarvisSection(title: l.configured),
          _ConfiguredModelCard(
            label: l.plannerModel,
            icon: Icons.architecture,
            model: planner,
          ),
          _ConfiguredModelCard(
            label: l.executorModel,
            icon: Icons.play_arrow,
            model: executor,
          ),
          _ConfiguredModelCard(
            label: l.coderModel,
            icon: Icons.code,
            model: coder,
          ),
          _ConfiguredModelCard(
            label: l.embeddingModel,
            icon: Icons.text_fields,
            model: embedding,
          ),

          // Available models
          const SizedBox(height: 8),
          JarvisSection(title: l.availableModels),
          if (available.isEmpty)
            JarvisEmptyState(
              icon: Icons.model_training,
              title: l.noModels,
            ),
          ...available.map<Widget>((m) {
            final model = m as Map<String, dynamic>? ?? {};
            final name = model['name']?.toString() ?? '';
            final prov = model['provider']?.toString() ?? '';
            final caps = model['capabilities'] as List<dynamic>? ?? [];

            return JarvisCard(
              title: name,
              icon: Icons.model_training,
              trailing: prov.isNotEmpty
                  ? JarvisStatusBadge(
                      label: prov,
                      color: JarvisTheme.accent,
                    )
                  : null,
              child: caps.isNotEmpty
                  ? Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: caps
                          .map<Widget>(
                            (c) => JarvisChip(label: c.toString()),
                          )
                          .toList(),
                    )
                  : const SizedBox.shrink(),
            );
          }),

          // Stats
          const SizedBox(height: 8),
          JarvisSection(title: l.modelStats),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              JarvisStat(
                label: l.total,
                value: totalModels,
                icon: Icons.model_training,
                color: JarvisTheme.accent,
              ),
              JarvisStat(
                label: l.providers,
                value: providerCount,
                icon: Icons.cloud,
                color: JarvisTheme.green,
              ),
              JarvisStat(
                label: l.capabilities,
                value: capCount,
                icon: Icons.star,
                color: JarvisTheme.orange,
              ),
            ],
          ),

          // Warnings
          if (warnings.isNotEmpty) ...[
            const SizedBox(height: 16),
            JarvisSection(title: l.modelWarnings),
            ...warnings.map<Widget>((w) {
              return JarvisCard(
                child: Row(
                  children: [
                    Icon(Icons.warning_amber, color: JarvisTheme.orange, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        w.toString(),
                        style: TextStyle(color: JarvisTheme.orange, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ],
      ),
    );
  }
}

class _ConfiguredModelCard extends StatelessWidget {
  const _ConfiguredModelCard({
    required this.label,
    required this.icon,
    required this.model,
  });

  final String label;
  final IconData icon;
  final Map<String, dynamic> model;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final name = model['name']?.toString() ?? l.notConfigured;
    final provider = model['provider']?.toString() ?? '';

    return JarvisCard(
      title: label,
      icon: icon,
      trailing: provider.isNotEmpty
          ? JarvisStatusBadge(label: provider, color: JarvisTheme.accent)
          : null,
      child: Text(
        name,
        style: Theme.of(context).textTheme.bodyMedium,
      ),
    );
  }
}
