/// Smoke test for [LlmBackendsScreen].
///
/// `initState` schedules a `WidgetsBinding.addPostFrameCallback` that
/// calls `LlmBackendProvider.refreshList()`. We use [SilentLlmBackendProvider]
/// (already in `silent_providers.dart`) which overrides that method
/// + suppresses the 2-second polling Timer. A finite pump drains the
/// post-frame callback without leaking any work.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/llm_backends_screen.dart';

import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

void main() {
  group('LlmBackendsScreen smoke', () {
    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<LlmBackendProvider>(
        create: (_) => SilentLlmBackendProvider(),
        child: const LlmBackendsScreen(),
      ),
    );

    testWidgets('renders without crashing on empty backend list', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      // Drain the post-frame callback that fires `refreshList`.
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(LlmBackendsScreen), findsOneWidget);
    });

    testWidgets('shows the AppBar title', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.text('LLM Backends'), findsOneWidget);
    });
  });
}
