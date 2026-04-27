import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class LoggingPage extends StatelessWidget {
  const LoggingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final log = cfg.cfg['logging'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorSelectField.fromStrings(
              label: 'Log Level',
              value: (log['level'] ?? 'INFO').toString(),
              options: const ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
              onChanged: (v) => cfg.set('logging.level', v),
            ),
            CognithorToggleField(
              label: 'JSON Logs',
              value: log['json_logs'] == true,
              onChanged: (v) => cfg.set('logging.json_logs', v),
              description: 'Structured JSON output for log aggregation',
            ),
            CognithorToggleField(
              label: 'Console Output',
              value: log['console'] != false,
              onChanged: (v) => cfg.set('logging.console', v),
            ),
          ],
        );
      },
    );
  }
}
