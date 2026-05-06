/// Smoke test for [ArcScreen].
///
/// Stateless placeholder screen: a Center column with a psychology
/// icon, a title and three info paragraphs. No providers, no Timer,
/// no animations — a single `pump` is enough.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/screens/arc_screen.dart';

import '../helpers/test_app.dart';

void main() {
  group('ArcScreen smoke', () {
    testWidgets('renders without crashing', (tester) async {
      await tester.pumpWidget(localizedTestApp(child: const ArcScreen()));
      await tester.pump();
      tester.takeException();
      expect(find.byType(ArcScreen), findsOneWidget);
    });

    testWidgets('shows the AppBar title and the placeholder icon', (
      tester,
    ) async {
      await tester.pumpWidget(localizedTestApp(child: const ArcScreen()));
      await tester.pump();
      tester.takeException();
      expect(find.text('ARC-AGI-3 Benchmark'), findsWidgets);
      expect(find.byIcon(Icons.psychology_outlined), findsOneWidget);
    });
  });
}
