import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

/// ATL (Autonomous Thinking Loop) configuration page.
///
/// Allows the user to enable/disable ATL, set the thinking interval,
/// configure quiet hours, risk ceiling, and max actions per cycle.
class AtlPage extends StatefulWidget {
  const AtlPage({super.key});

  @override
  State<AtlPage> createState() => _AtlPageState();
}

class _AtlPageState extends State<AtlPage> {
  Map<String, dynamic> _atl(ConfigProvider cfg) {
    final raw = cfg.cfg['atl'];
    if (raw is Map<String, dynamic>) return raw;
    return {};
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final atl = _atl(cfg);
        final enabled = atl['enabled'] == true;
        final quietStart = (atl['quiet_hours_start'] ?? '').toString();
        final quietEnd = (atl['quiet_hours_end'] ?? '').toString();
        final hasQuietHours = quietStart.isNotEmpty && quietEnd.isNotEmpty;

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // ── Master Toggle ──
            CognithorToggleField(
              label: 'ATL Enabled',
              description:
                  'Autonomous Thinking Loop — Cognithor thinks proactively '
                  'during idle time, evaluates goals, and dispatches research.',
              value: enabled,
              onChanged: (v) => cfg.set('atl.enabled', v),
            ),

            if (enabled) ...[
              const SizedBox(height: 8),

              // ── Interval ──
              CognithorSliderField(
                label: 'Thinking Interval (minutes)',
                description:
                    'How often ATL runs a thinking cycle when the system is idle.',
                value: ((atl['interval_minutes'] as num?) ?? 15).toDouble(),
                min: 5,
                max: 60,
                step: 5,
                onChanged: (v) => cfg.set('atl.interval_minutes', v.round()),
              ),

              // ── Max Actions ──
              CognithorNumberField(
                label: 'Max Actions per Cycle',
                value: (atl['max_actions_per_cycle'] as num?) ?? 3,
                min: 1,
                max: 10,
                onChanged: (v) =>
                    cfg.set('atl.max_actions_per_cycle', v.toInt()),
              ),

              // ── Risk Ceiling ──
              CognithorSelectField.fromStrings(
                label: 'Risk Ceiling',
                value: (atl['risk_ceiling'] ?? 'YELLOW').toString(),
                options: const ['GREEN', 'YELLOW'],
                onChanged: (v) => cfg.set('atl.risk_ceiling', v),
              ),

              // ── Token Budget ──
              CognithorNumberField(
                label: 'Max Tokens per Cycle',
                value: (atl['max_tokens_per_cycle'] as num?) ?? 4000,
                min: 1000,
                max: 16000,
                onChanged: (v) =>
                    cfg.set('atl.max_tokens_per_cycle', v.toInt()),
              ),

              const Divider(height: 32),

              // ── Quiet Hours Section ──
              Text(
                'Quiet Hours',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 4),
              Text(
                hasQuietHours
                    ? 'ATL pauses between $quietStart and $quietEnd.'
                    : 'No quiet hours configured — ATL runs 24/7.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 12),

              Row(
                children: [
                  Expanded(
                    child: _TimePickerField(
                      label: 'Start',
                      value: quietStart,
                      onChanged: (v) => cfg.set('atl.quiet_hours_start', v),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: _TimePickerField(
                      label: 'End',
                      value: quietEnd,
                      onChanged: (v) => cfg.set('atl.quiet_hours_end', v),
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (hasQuietHours)
                    IconButton(
                      icon: const Icon(Icons.clear),
                      tooltip: 'Disable Quiet Hours',
                      onPressed: () {
                        cfg.set('atl.quiet_hours_start', '');
                        cfg.set('atl.quiet_hours_end', '');
                      },
                    ),
                ],
              ),

              const Divider(height: 32),

              // ── Notification ──
              CognithorTextField(
                label: 'Notification Channel',
                value: (atl['notification_channel'] ?? '').toString(),
                onChanged: (v) => cfg.set('atl.notification_channel', v),
              ),
              CognithorSelectField.fromStrings(
                label: 'Notification Level',
                value: (atl['notification_level'] ?? 'important').toString(),
                options: const ['all', 'important', 'critical'],
                onChanged: (v) => cfg.set('atl.notification_level', v),
              ),
            ],
          ],
        );
      },
    );
  }
}

/// A time picker field that shows HH:MM and opens a TimePicker dialog.
class _TimePickerField extends StatelessWidget {
  const _TimePickerField({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final String value; // "HH:MM" or empty
  final ValueChanged<String> onChanged;

  TimeOfDay? _parse(String v) {
    if (v.isEmpty) return null;
    final parts = v.split(':');
    if (parts.length != 2) return null;
    final h = int.tryParse(parts[0]);
    final m = int.tryParse(parts[1]);
    if (h == null || m == null) return null;
    return TimeOfDay(hour: h, minute: m);
  }

  String _format(TimeOfDay t) {
    return '${t.hour.toString().padLeft(2, '0')}:'
        '${t.minute.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final parsed = _parse(value);
    return InkWell(
      onTap: () async {
        final picked = await showTimePicker(
          context: context,
          initialTime: parsed ?? const TimeOfDay(hour: 23, minute: 0),
          builder: (context, child) {
            return MediaQuery(
              data: MediaQuery.of(
                context,
              ).copyWith(alwaysUse24HourFormat: true),
              child: child!,
            );
          },
        );
        if (picked != null) {
          onChanged(_format(picked));
        }
      },
      borderRadius: BorderRadius.circular(8),
      child: InputDecorator(
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          suffixIcon: const Icon(Icons.access_time),
        ),
        child: Text(
          parsed != null ? _format(parsed) : 'Not set',
          style: Theme.of(context).textTheme.bodyLarge,
        ),
      ),
    );
  }
}
