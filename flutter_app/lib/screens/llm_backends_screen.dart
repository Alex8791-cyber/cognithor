import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/vllm_setup_screen.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class LlmBackendsScreen extends StatefulWidget {
  const LlmBackendsScreen({super.key});

  @override
  State<LlmBackendsScreen> createState() => _LlmBackendsScreenState();
}

class _LlmBackendsScreenState extends State<LlmBackendsScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LlmBackendProvider>().refreshList();
    });
  }

  @override
  Widget build(BuildContext context) {
    final p = context.watch<LlmBackendProvider>();
    return Scaffold(
      appBar: AppBar(title: const Text('LLM Backends')),
      body: ListView.builder(
        itemCount: p.backends.length,
        itemBuilder: (ctx, i) {
          final b = p.backends[i];
          final isActive = p.active == b.name;
          return ListTile(
            leading: Icon(
              b.status == 'ready' ? Icons.circle : Icons.circle_outlined,
              color: b.status == 'ready' ? Colors.green : Colors.grey,
              size: 14,
            ),
            title: Text(_displayName(b.name)),
            subtitle: Text(_statusLine(b)),
            trailing: isActive
                ? Container(
                    key: ValueKey('backend-${b.name}-active'),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Theme.of(
                        ctx,
                      ).colorScheme.primary.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Text('Active', style: TextStyle(fontSize: 11)),
                  )
                : const Icon(Icons.chevron_right),
            onTap: () {
              if (b.name == 'vllm') {
                Navigator.of(ctx).push(
                  MaterialPageRoute(builder: (_) => const VllmSetupScreen()),
                );
              }
            },
          );
        },
      ),
    );
  }

  static String _displayName(String name) {
    switch (name) {
      case 'ollama':
        return 'Ollama';
      case 'vllm':
        return 'vLLM';
      case 'openai':
        return 'OpenAI';
      case 'anthropic':
        return 'Anthropic';
      default:
        return name;
    }
  }

  static String _statusLine(BackendEntry b) {
    if (!b.enabled) return 'Disabled';
    return b.status;
  }
}
