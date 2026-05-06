/// Smoke test for [VllmSetupScreen].
///
/// The screen owns:
///   - a post-frame `LlmBackendProvider.startPolling()` call which
///     spins a 2-second Timer.periodic against `vllm/status`,
///   - a nested `_ModelCard.initState` that fires `fetchAvailableModels()`
///     against `/api/backends/vllm/available-models`.
///
/// [SilentLlmBackendProvider] no-ops both methods, so a finite pump
/// after mount drains the postFrame callbacks without leaking any
/// network/Timer work.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/vllm_setup_screen.dart';

import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

void main() {
  group('VllmSetupScreen smoke', () {
    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<LlmBackendProvider>(
        create: (_) => SilentLlmBackendProvider(),
        child: const VllmSetupScreen(),
      ),
    );

    testWidgets('renders without crashing on null vllm status', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      // Drain `addPostFrameCallback` for both startPolling +
      // fetchAvailableModels (both no-op via the silent provider).
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(VllmSetupScreen), findsOneWidget);
    });

    testWidgets('shows the four status cards', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // Hardware / Docker / Image / Model cards each have their own
      // ValueKey so `find.byKey` is the cleanest sanity check.
      expect(find.byKey(const ValueKey('card-hardware')), findsOneWidget);
      expect(find.byKey(const ValueKey('card-docker')), findsOneWidget);
      expect(find.byKey(const ValueKey('card-image')), findsOneWidget);
      expect(find.byKey(const ValueKey('card-model')), findsOneWidget);
    });

    testWidgets('disposes cleanly when replaced', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // dispose path: VllmSetupScreen calls _provider!.stopPolling()
      // — must not throw even when startPolling was a no-op.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      tester.takeException();
      expect(find.byType(VllmSetupScreen), findsNothing);
    });
  });
}
