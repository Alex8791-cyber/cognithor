import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class EvolutionConfigPage extends StatelessWidget {
  const EvolutionConfigPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final evo = cfg.cfg['evolution'] as Map<String, dynamic>? ?? {};
        final enabled = evo['enabled'] == true;
        final idleMinutes = ((evo['idle_minutes'] as num?) ?? 5).toInt();
        final maxCycles = ((evo['max_cycles_per_day'] as num?) ?? 10).toInt();
        final deepLearning = evo['deep_learning_enabled'] == true;

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            SwitchListTile(
              title: const Text('Evolution Engine'),
              subtitle: const Text(
                'Enable autonomous learning during idle time',
              ),
              value: enabled,
              activeThumbColor: CognithorTheme.accent,
              onChanged: (v) => cfg.set('evolution.enabled', v),
            ),
            const Divider(height: 24),
            ListTile(
              title: const Text('Idle Threshold'),
              subtitle: Text(
                'Start learning after $idleMinutes minutes of inactivity',
              ),
              trailing: SizedBox(
                width: 120,
                child: DropdownButtonFormField<int>(
                  initialValue: [1, 2, 5, 10, 15, 30].contains(idleMinutes)
                      ? idleMinutes
                      : 5,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                  ),
                  items: const [
                    DropdownMenuItem(value: 1, child: Text('1 min')),
                    DropdownMenuItem(value: 2, child: Text('2 min')),
                    DropdownMenuItem(value: 5, child: Text('5 min')),
                    DropdownMenuItem(value: 10, child: Text('10 min')),
                    DropdownMenuItem(value: 15, child: Text('15 min')),
                    DropdownMenuItem(value: 30, child: Text('30 min')),
                  ],
                  onChanged: (v) => cfg.set('evolution.idle_minutes', v),
                ),
              ),
            ),
            const SizedBox(height: 12),
            ListTile(
              title: const Text('Max Cycles / Day'),
              subtitle: Text('$maxCycles cycles allowed per day'),
              trailing: SizedBox(
                width: 120,
                child: DropdownButtonFormField<int>(
                  initialValue: [1, 3, 5, 10, 20, 50, 100].contains(maxCycles)
                      ? maxCycles
                      : 10,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                  ),
                  items: const [
                    DropdownMenuItem(value: 1, child: Text('1')),
                    DropdownMenuItem(value: 3, child: Text('3')),
                    DropdownMenuItem(value: 5, child: Text('5')),
                    DropdownMenuItem(value: 10, child: Text('10')),
                    DropdownMenuItem(value: 20, child: Text('20')),
                    DropdownMenuItem(value: 50, child: Text('50')),
                    DropdownMenuItem(value: 100, child: Text('100')),
                  ],
                  onChanged: (v) => cfg.set('evolution.max_cycles_per_day', v),
                ),
              ),
            ),
            const SizedBox(height: 12),
            SwitchListTile(
              title: const Text('Deep Learning Plans'),
              subtitle: const Text(
                'Auto-promote complex goals to structured learning plans',
              ),
              value: deepLearning,
              activeThumbColor: CognithorTheme.accent,
              onChanged: (v) => cfg.set('evolution.deep_learning_enabled', v),
            ),
          ],
        );
      },
    );
  }
}
