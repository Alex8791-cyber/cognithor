import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';

/// Safely convert a model config value to Map.
/// API may return either a Map or a plain String (just the model name).
Map<String, dynamic> _asModelMap(dynamic value) {
  if (value is Map<String, dynamic>) return value;
  if (value is String) return {'model': value};
  return {};
}

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
    final conn = context.read<ConnectionProvider>();
    admin.setApi(conn.api);
    admin.loadModels();
    admin.loadModelStats();
    // Also load config for configured model details
    final cfg = context.read<ConfigProvider>();
    cfg.setApi(conn.api);
    if (cfg.cfg.isEmpty) cfg.loadAll();
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
      return CognithorEmptyState(
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
    // Available models come as a flat list of strings from /models/available
    final rawAvailable = modelsData['models'];
    final available = rawAvailable is List ? rawAvailable : [];

    final stats = admin.modelStats ?? {};
    final totalModels =
        stats['total_models']?.toString() ?? available.length.toString();
    final providersList = stats['providers'];
    final providerCount = providersList is List
        ? providersList.length.toString()
        : '-';
    final capsList = stats['capabilities'];
    final capCount = capsList is List ? capsList.length.toString() : '-';

    // Configured models come from the config API (ConfigProvider)
    final cfg = context.watch<ConfigProvider>();
    final configModels = cfg.cfg['models'] as Map<String, dynamic>? ?? {};
    final planner = _asModelMap(configModels['planner']);
    final executor = _asModelMap(configModels['executor']);
    final coder = _asModelMap(configModels['coder']);
    final embedding = _asModelMap(configModels['embedding']);

    return RefreshIndicator(
      onRefresh: () async {
        final a = context.read<AdminProvider>();
        await Future.wait([a.loadModels(), a.loadModelStats()]);
      },
      color: CognithorTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Configured models
          CognithorSection(title: l.configured),
          _ConfiguredModelCard(
            label: l.plannerModel,
            icon: Icons.architecture,
            model: planner,
            configKey: 'planner',
            availableModels: available,
          ),
          _ConfiguredModelCard(
            label: l.executorModel,
            icon: Icons.play_arrow,
            model: executor,
            configKey: 'executor',
            availableModels: available,
          ),
          _ConfiguredModelCard(
            label: l.coderModel,
            icon: Icons.code,
            model: coder,
            configKey: 'coder',
            availableModels: available,
          ),
          _ConfiguredModelCard(
            label: l.embeddingModel,
            icon: Icons.text_fields,
            model: embedding,
            configKey: 'embedding',
            availableModels: available,
          ),

          // Available models
          const SizedBox(height: 8),
          CognithorSection(title: l.availableModels),
          if (available.isEmpty)
            CognithorEmptyState(icon: Icons.model_training, title: l.noModels),
          ...available.map<Widget>((m) {
            // Available models may be strings or Maps
            final name = m is String
                ? m
                : (m is Map ? m['name']?.toString() ?? '' : m.toString());

            return Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: NeonCard(
                tint: CognithorTheme.sectionAdmin,
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 10,
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.model_training,
                      size: 16,
                      color: CognithorTheme.sectionAdmin,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        name,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }),
          // Remove the old caps/prov section — replace with a simple note
          if (available.length > 20) ...[
            const SizedBox(height: 8),
            Text(
              '${available.length} models available from ${cfg.cfg['llm_backend_type'] ?? 'backend'}',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: CognithorTheme.textSecondary,
              ),
            ),
          ],

          // Stats
          const SizedBox(height: 8),
          CognithorSection(title: l.modelStats),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              CognithorStat(
                label: l.total,
                value: totalModels,
                icon: Icons.model_training,
                color: CognithorTheme.accent,
              ),
              CognithorStat(
                label: l.providers,
                value: providerCount,
                icon: Icons.cloud,
                color: CognithorTheme.green,
              ),
              CognithorStat(
                label: l.capabilities,
                value: capCount,
                icon: Icons.star,
                color: CognithorTheme.orange,
              ),
            ],
          ),

          // Warnings (from model stats)
          if (stats['warnings'] is List &&
              (stats['warnings'] as List).isNotEmpty) ...[
            const SizedBox(height: 16),
            CognithorSection(title: l.modelWarnings),
            ...(stats['warnings'] as List).map<Widget>((w) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: NeonCard(
                  tint: CognithorTheme.orange,
                  child: Row(
                    children: [
                      Icon(
                        Icons.warning_amber,
                        color: CognithorTheme.orange,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          w.toString(),
                          style: TextStyle(
                            color: CognithorTheme.orange,
                            fontSize: 13,
                          ),
                        ),
                      ),
                    ],
                  ),
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
    required this.configKey,
    required this.availableModels,
  });

  final String label;
  final IconData icon;
  final Map<String, dynamic> model;
  final String configKey; // "planner", "executor", "coder", "embedding"
  final List<dynamic> availableModels;

  void _showModelPicker(BuildContext context) {
    final currentName = model['name']?.toString() ?? '';
    final models =
        availableModels.map((m) => m is String ? m : m.toString()).toList()
          ..sort();

    showDialog<String>(
      context: context,
      builder: (ctx) {
        String? search;
        return StatefulBuilder(
          builder: (ctx, setState) {
            final filtered = search == null || search!.isEmpty
                ? models
                : models
                      .where(
                        (m) => m.toLowerCase().contains(search!.toLowerCase()),
                      )
                      .toList();

            final ml = AppLocalizations.of(ctx);
            return AlertDialog(
              title: Text('${ml.selectTemplate}: $label'),
              content: SizedBox(
                width: 400,
                height: 500,
                child: Column(
                  children: [
                    TextField(
                      decoration: InputDecoration(
                        hintText: ml.search,
                        prefixIcon: const Icon(Icons.search, size: 20),
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
                          final isSelected = name == currentName;
                          return ListTile(
                            dense: true,
                            selected: isSelected,
                            selectedColor: CognithorTheme.sectionAdmin,
                            leading: Icon(
                              isSelected
                                  ? Icons.check_circle
                                  : Icons.circle_outlined,
                              size: 18,
                              color: isSelected
                                  ? CognithorTheme.sectionAdmin
                                  : null,
                            ),
                            title: Text(
                              name,
                              style: const TextStyle(fontSize: 13),
                            ),
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
                  child: Text(ml.cancel),
                ),
              ],
            );
          },
        );
      },
    ).then((selected) {
      if (selected != null && selected != currentName && context.mounted) {
        _saveModelSelection(context, selected);
      }
    });
  }

  void _saveModelSelection(BuildContext context, String modelName) {
    final cfg = context.read<ConfigProvider>();
    cfg.set('models.$configKey.name', modelName);
    cfg.save().then((ok) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              ok
                  ? '$label: $modelName'
                  : AppLocalizations.of(context).saveFailed,
            ),
            backgroundColor: ok ? CognithorTheme.green : CognithorTheme.red,
          ),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final name = model['name']?.toString() ?? l.notConfigured;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: NeonCard(
        tint: CognithorTheme.sectionAdmin,
        glowOnHover: true,
        onTap: availableModels.isNotEmpty
            ? () => _showModelPicker(context)
            : null,
        child: Row(
          children: [
            Icon(icon, size: 18, color: CognithorTheme.sectionAdmin),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label, style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 2),
                  Text(
                    name,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: CognithorTheme.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.swap_horiz,
              size: 20,
              color: CognithorTheme.textSecondary,
            ),
          ],
        ),
      ),
    );
  }
}
