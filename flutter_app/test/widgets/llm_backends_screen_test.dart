import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/llm_backends_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

LlmBackendProvider _mkProvider(List<BackendEntry> backends, String active) {
  final p = LlmBackendProvider(apiBaseUrl: 'http://test');
  p.backends = backends;
  p.active = active;
  return p;
}

void main() {
  testWidgets('renders all backends with status dots', (tester) async {
    final provider = _mkProvider([
      BackendEntry(name: 'ollama', enabled: true, status: 'ready'),
      BackendEntry(name: 'vllm', enabled: false, status: 'disabled'),
    ], 'ollama');

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider<LlmBackendProvider>.value(
          value: provider,
          child: const LlmBackendsScreen(),
        ),
      ),
    );

    expect(find.text('Ollama'), findsOneWidget);
    expect(find.text('vLLM'), findsOneWidget);
  });

  testWidgets('active backend has a visual marker', (tester) async {
    final provider = _mkProvider([
      BackendEntry(name: 'ollama', enabled: true, status: 'ready'),
      BackendEntry(name: 'vllm', enabled: true, status: 'ready'),
    ], 'vllm');

    await tester.pumpWidget(
      MaterialApp(
        home: ChangeNotifierProvider<LlmBackendProvider>.value(
          value: provider,
          child: const LlmBackendsScreen(),
        ),
      ),
    );

    expect(find.byKey(const ValueKey('backend-vllm-active')), findsOneWidget);
  });
}
