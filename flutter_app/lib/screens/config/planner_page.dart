import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class PlannerPage extends StatelessWidget {
  const PlannerPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final planner = cfg.cfg['planner'] as Map<String, dynamic>? ?? {};
        final gk = cfg.cfg['gatekeeper'] as Map<String, dynamic>? ?? {};
        final sb = cfg.cfg['sandbox'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorCollapsibleCard(
              title: 'Planner (PGE)',
              icon: Icons.architecture,
              initiallyExpanded: true,
              children: [
                CognithorNumberField(
                  label: 'Max Iterations',
                  value: (planner['max_iterations'] as num?) ?? 25,
                  onChanged: (v) => cfg.set('planner.max_iterations', v),
                  min: 1,
                  max: 50,
                ),
                CognithorNumberField(
                  label: 'Escalation After',
                  value: (planner['escalation_after'] as num?) ?? 3,
                  onChanged: (v) => cfg.set('planner.escalation_after', v),
                  min: 1,
                ),
                CognithorSliderField(
                  label: 'Temperature',
                  value: (planner['temperature'] as num?)?.toDouble() ?? 0.7,
                  onChanged: (v) => cfg.set('planner.temperature', v),
                  max: 2.0,
                  step: 0.05,
                ),
                CognithorNumberField(
                  label: 'Response Token Budget',
                  value: (planner['response_token_budget'] as num?) ?? 4000,
                  onChanged: (v) => cfg.set('planner.response_token_budget', v),
                  min: 256,
                  max: 128000,
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'Gatekeeper',
              icon: Icons.shield,
              children: [
                CognithorTextField(
                  label: 'Policies Directory',
                  value: (gk['policies_dir'] ?? '').toString(),
                  onChanged: (v) => cfg.set('gatekeeper.policies_dir', v),
                ),
                CognithorSelectField.fromStrings(
                  label: 'Default Risk Level',
                  value: (gk['default_risk_level'] ?? 'orange').toString(),
                  options: const ['green', 'yellow', 'orange', 'red'],
                  onChanged: (v) => cfg.set('gatekeeper.default_risk_level', v),
                ),
                CognithorNumberField(
                  label: 'Max Blocked Retries',
                  value: (gk['max_blocked_retries'] as num?) ?? 3,
                  onChanged: (v) =>
                      cfg.set('gatekeeper.max_blocked_retries', v),
                  min: 0,
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'Sandbox',
              icon: Icons.security,
              children: [
                CognithorSelectField.fromStrings(
                  label: 'Level',
                  value: (sb['level'] ?? 'process').toString(),
                  options: const [
                    'process',
                    'namespace',
                    'container',
                    'jobobject',
                  ],
                  onChanged: (v) => cfg.set('sandbox.level', v),
                ),
                CognithorNumberField(
                  label: 'Timeout (seconds)',
                  value: (sb['timeout_seconds'] as num?) ?? 30,
                  onChanged: (v) => cfg.set('sandbox.timeout_seconds', v),
                  min: 1,
                ),
                CognithorNumberField(
                  label: 'Max Memory (MB)',
                  value: (sb['max_memory_mb'] as num?) ?? 512,
                  onChanged: (v) => cfg.set('sandbox.max_memory_mb', v),
                  min: 64,
                ),
                CognithorNumberField(
                  label: 'Max CPU Seconds',
                  value: (sb['max_cpu_seconds'] as num?) ?? 30,
                  onChanged: (v) => cfg.set('sandbox.max_cpu_seconds', v),
                  min: 1,
                ),
                CognithorListField(
                  label: 'Allowed Paths',
                  value: _toStringList(sb['allowed_paths']),
                  onChanged: (v) => cfg.set('sandbox.allowed_paths', v),
                ),
                CognithorToggleField(
                  label: 'Network Access',
                  value: sb['network_access'] == true,
                  onChanged: (v) => cfg.set('sandbox.network_access', v),
                ),
                CognithorJsonEditor(
                  label: 'Environment Variables',
                  value: sb['env_vars'] ?? {},
                  onChanged: (v) => cfg.set('sandbox.env_vars', v),
                  rows: 4,
                ),
              ],
            ),
          ],
        );
      },
    );
  }

  static List<String> _toStringList(dynamic v) {
    if (v is List) return v.map((e) => e.toString()).toList();
    return [];
  }
}
