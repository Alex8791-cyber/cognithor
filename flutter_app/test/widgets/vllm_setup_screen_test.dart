import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/vllm_setup_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

LlmBackendProvider _mkProvider(VLLMStatus? s) {
  final p = LlmBackendProvider(apiBaseUrl: 'http://test');
  p.vllmStatus = s;
  return p;
}

void main() {
  testWidgets('all four status cards are rendered', (tester) async {
    final provider = _mkProvider(VLLMStatus(
      hardwareOk: false,
      hardwareInfo: null,
      dockerOk: false,
      imagePulled: false,
      containerRunning: false,
      currentModel: null,
      lastError: null,
    ));

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const VllmSetupScreen(),
      ),
    ));

    expect(find.byKey(const ValueKey('card-hardware')), findsOneWidget);
    expect(find.byKey(const ValueKey('card-docker')), findsOneWidget);
    expect(find.byKey(const ValueKey('card-image')), findsOneWidget);
    expect(find.byKey(const ValueKey('card-model')), findsOneWidget);
  });

  testWidgets('hardware card shows GPU name when detected', (tester) async {
    final provider = _mkProvider(VLLMStatus(
      hardwareOk: true,
      hardwareInfo: HardwareInfo(
        gpuName: 'RTX 5090',
        vramGb: 32,
        computeCapability: '12.0',
      ),
      dockerOk: true,
      imagePulled: false,
      containerRunning: false,
      currentModel: null,
      lastError: null,
    ));

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const VllmSetupScreen(),
      ),
    ));

    expect(find.textContaining('RTX 5090'), findsOneWidget);
    expect(find.textContaining('32 GB'), findsOneWidget);
  });

  testWidgets('image card shows pull button when pending', (tester) async {
    final provider = _mkProvider(VLLMStatus(
      hardwareOk: true,
      hardwareInfo: HardwareInfo(
        gpuName: 'RTX 5090',
        vramGb: 32,
        computeCapability: '12.0',
      ),
      dockerOk: true,
      imagePulled: false,
      containerRunning: false,
      currentModel: null,
      lastError: null,
    ));

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const VllmSetupScreen(),
      ),
    ));

    expect(find.text('Pull image'), findsOneWidget);
  });
}
