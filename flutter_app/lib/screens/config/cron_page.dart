import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class CronPage extends StatelessWidget {
  const CronPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final hb = cfg.cfg['heartbeat'] as Map<String, dynamic>? ?? {};
        final plugins = cfg.cfg['plugins'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorCollapsibleCard(
              title: 'Heartbeat',
              icon: Icons.favorite,
              initiallyExpanded: true,
              children: [
                CognithorToggleField(
                  label: 'Enabled',
                  value: hb['enabled'] == true,
                  onChanged: (v) => cfg.set('heartbeat.enabled', v),
                ),
                CognithorNumberField(
                  label: 'Interval (minutes)',
                  value: (hb['interval_minutes'] as num?) ?? 60,
                  onChanged: (v) => cfg.set('heartbeat.interval_minutes', v),
                  min: 1,
                ),
                CognithorTextField(
                  label: 'Checklist File',
                  value: (hb['checklist_file'] ?? '').toString(),
                  onChanged: (v) => cfg.set('heartbeat.checklist_file', v),
                ),
                CognithorTextField(
                  label: 'Channel',
                  value: (hb['channel'] ?? '').toString(),
                  onChanged: (v) => cfg.set('heartbeat.channel', v),
                ),
                CognithorTextField(
                  label: 'Model',
                  value: (hb['model'] ?? '').toString(),
                  onChanged: (v) => cfg.set('heartbeat.model', v),
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'Plugins',
              icon: Icons.extension,
              children: [
                CognithorTextField(
                  label: 'Skills Directory',
                  value: (plugins['skills_dir'] ?? '').toString(),
                  onChanged: (v) => cfg.set('plugins.skills_dir', v),
                ),
                CognithorToggleField(
                  label: 'Auto Update',
                  value: plugins['auto_update'] == true,
                  onChanged: (v) => cfg.set('plugins.auto_update', v),
                ),
              ],
            ),
            const Divider(height: 32),
            Row(
              children: [
                Text(
                  AppLocalizations.of(context).cronJobs,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontSize: 16),
                ),
                const Spacer(),
                IconButton(
                  icon: Icon(Icons.add, color: CognithorTheme.accent),
                  onPressed: () => cfg.addCronJob({
                    'name': 'new-job',
                    'schedule': '0 * * * *',
                    'command': '',
                    'enabled': true,
                  }),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ...List.generate(cfg.cronJobs.length, (i) {
              final job = cfg.cronJobs[i];
              return CognithorCollapsibleCard(
                title: (job['name'] ?? 'Job $i').toString(),
                icon: Icons.schedule,
                badge: _humanCron(job['schedule']?.toString() ?? ''),
                children: [
                  CognithorTextField(
                    label: 'Name',
                    value: (job['name'] ?? '').toString(),
                    onChanged: (v) => cfg.updateCronJob(i, {...job, 'name': v}),
                  ),
                  CognithorTextField(
                    label: 'Schedule (cron)',
                    value: (job['schedule'] ?? '').toString(),
                    onChanged: (v) =>
                        cfg.updateCronJob(i, {...job, 'schedule': v}),
                    mono: true,
                  ),
                  CognithorTextField(
                    label: 'Command',
                    value: (job['command'] ?? '').toString(),
                    onChanged: (v) =>
                        cfg.updateCronJob(i, {...job, 'command': v}),
                  ),
                  CognithorToggleField(
                    label: 'Enabled',
                    value: job['enabled'] == true,
                    onChanged: (v) =>
                        cfg.updateCronJob(i, {...job, 'enabled': v}),
                  ),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed: () => cfg.removeCronJob(i),
                      icon: Icon(
                        Icons.delete,
                        size: 16,
                        color: CognithorTheme.red,
                      ),
                      label: Text(
                        AppLocalizations.of(context).remove,
                        style: TextStyle(color: CognithorTheme.red),
                      ),
                    ),
                  ),
                ],
              );
            }),
          ],
        );
      },
    );
  }

  static String _humanCron(String cron) {
    if (cron.isEmpty) return '';
    final parts = cron.split(' ');
    if (parts.length < 5) return cron;
    final min = parts[0];
    final hour = parts[1];
    if (min != '*' && hour != '*') {
      return 'at ${hour.padLeft(2, '0')}:${min.padLeft(2, '0')}';
    }
    if (min == '0' && hour == '*') return 'every hour';
    if (min == '*') return 'every minute';
    return cron;
  }
}
