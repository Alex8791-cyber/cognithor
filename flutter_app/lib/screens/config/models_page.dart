import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/providers/config_provider.dart';
import 'package:jarvis_ui/widgets/form/form_widgets.dart';

class ModelsPage extends StatelessWidget {
  const ModelsPage({super.key});

  static const _roles = [
    ('planner', 'Planner', Icons.architecture),
    ('executor', 'Executor', Icons.play_arrow),
    ('coder', 'Coder', Icons.code),
    ('coder_fast', 'Coder (Fast)', Icons.flash_on),
    ('embedding', 'Embedding', Icons.scatter_plot),
  ];

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final models =
            cfg.cfg['models'] as Map<String, dynamic>? ?? {};
        final defaultBackend =
            (cfg.cfg['llm_backend_type'] ?? 'ollama').toString();

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            JarvisTextField(
              label: 'Vision Model',
              value: (cfg.cfg['vision_model'] ?? '').toString(),
              onChanged: (v) => cfg.set('vision_model', v),
            ),
            JarvisTextField(
              label: 'Vision Detail Model',
              value: (cfg.cfg['vision_model_detail'] ?? 'qwen3-vl:32b').toString(),
              onChanged: (v) => cfg.set('vision_model_detail', v),
            ),
            const SizedBox(height: 8),
            ..._roles.map((r) {
              final (key, label, icon) = r;
              final model =
                  models[key] as Map<String, dynamic>? ?? {};
              final modelBackend = (model['backend'] ?? '').toString();
              final effectiveBackend =
                  modelBackend.isNotEmpty ? modelBackend : defaultBackend;
              return JarvisCollapsibleCard(
                title: label,
                icon: icon,
                badge: (model['name'] ?? '').toString().isNotEmpty
                    ? (model['name'] ?? '').toString()
                    : null,
                children: [
                  JarvisTextField(
                    label: 'Model Name',
                    value: (model['name'] ?? '').toString(),
                    onChanged: (v) => cfg.set('models.$key.name', v),
                  ),
                  JarvisNumberField(
                    label: 'Context Window',
                    value: (model['context_window'] as num?) ?? 8192,
                    onChanged: (v) => cfg.set('models.$key.context_window', v),
                    min: 512,
                  ),
                  JarvisNumberField(
                    label: 'VRAM (GB)',
                    value: (model['vram_gb'] as num?) ?? 0,
                    onChanged: (v) => cfg.set('models.$key.vram_gb', v),
                    min: 0,
                    decimal: true,
                  ),
                  JarvisSliderField(
                    label: 'Temperature',
                    value: (model['temperature'] as num?)?.toDouble() ?? 0.7,
                    onChanged: (v) => cfg.set('models.$key.temperature', v),
                    min: 0.0,
                    max: 2.0,
                    step: 0.05,
                  ),
                  JarvisSliderField(
                    label: 'Top P',
                    value: (model['top_p'] as num?)?.toDouble() ?? 0.9,
                    onChanged: (v) => cfg.set('models.$key.top_p', v),
                  ),
                  JarvisSelectField.fromStrings(
                    label: 'Backend',
                    value: effectiveBackend,
                    options: const [
                      'ollama', 'openai', 'anthropic', 'gemini', 'groq',
                      'deepseek', 'mistral', 'together', 'openrouter',
                      'xai', 'cerebras', 'github', 'bedrock', 'huggingface',
                      'moonshot', 'vllm', 'llama_cpp',
                    ],
                    onChanged: (v) => cfg.set('models.$key.backend', v),
                    description: modelBackend.isEmpty
                        ? 'Inherited from global: $defaultBackend'
                        : null,
                  ),
                ],
              );
            }),
          ],
        );
      },
    );
  }
}
