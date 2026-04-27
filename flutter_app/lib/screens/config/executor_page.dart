import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class ExecutorPage extends StatelessWidget {
  const ExecutorPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final ex = cfg.cfg['executor'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorNumberField(
              label: 'Default Timeout (seconds)',
              value: (ex['default_timeout_seconds'] as num?) ?? 30,
              onChanged: (v) => cfg.set('executor.default_timeout_seconds', v),
              min: 5,
              max: 3600,
            ),
            CognithorNumberField(
              label: 'Max Output Chars',
              value: (ex['max_output_chars'] as num?) ?? 10000,
              onChanged: (v) => cfg.set('executor.max_output_chars', v),
              min: 1000,
              max: 500000,
            ),
            CognithorNumberField(
              label: 'Max Retries',
              value: (ex['max_retries'] as num?) ?? 3,
              onChanged: (v) => cfg.set('executor.max_retries', v),
              min: 0,
              max: 10,
            ),
            CognithorNumberField(
              label: 'Backoff Base Delay (seconds)',
              value: (ex['backoff_base_delay_seconds'] as num?) ?? 1,
              onChanged: (v) =>
                  cfg.set('executor.backoff_base_delay_seconds', v),
              min: 0,
              decimal: true,
            ),
            CognithorNumberField(
              label: 'Max Parallel Tools',
              value: (ex['max_parallel_tools'] as num?) ?? 4,
              onChanged: (v) => cfg.set('executor.max_parallel_tools', v),
              min: 1,
              max: 20,
            ),
            const Divider(height: 32),
            Text(
              AppLocalizations.of(context).toolSpecificTimeouts,
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontSize: 16),
            ),
            const SizedBox(height: 12),
            CognithorNumberField(
              label: 'Image Analysis Timeout',
              value: (ex['media_analyze_image_timeout'] as num?) ?? 180,
              onChanged: (v) =>
                  cfg.set('executor.media_analyze_image_timeout', v),
              min: 5,
            ),
            CognithorNumberField(
              label: 'Audio Transcription Timeout',
              value: (ex['media_transcribe_audio_timeout'] as num?) ?? 120,
              onChanged: (v) =>
                  cfg.set('executor.media_transcribe_audio_timeout', v),
              min: 5,
            ),
            CognithorNumberField(
              label: 'Text Extraction Timeout',
              value: (ex['media_extract_text_timeout'] as num?) ?? 120,
              onChanged: (v) =>
                  cfg.set('executor.media_extract_text_timeout', v),
              min: 5,
            ),
            CognithorNumberField(
              label: 'TTS Timeout',
              value: (ex['media_tts_timeout'] as num?) ?? 120,
              onChanged: (v) => cfg.set('executor.media_tts_timeout', v),
              min: 5,
            ),
            CognithorNumberField(
              label: 'Run Python Timeout',
              value: (ex['run_python_timeout'] as num?) ?? 120,
              onChanged: (v) => cfg.set('executor.run_python_timeout', v),
              min: 5,
            ),
          ],
        );
      },
    );
  }
}
