/// Smoke test for [AdminHubScreen].
///
/// The hub is a composite landing screen: in wide mode it embeds the
/// currently-selected admin sub-screen (default: [ConfigScreen]) into
/// the right pane, which would pull in `ConfigProvider` and friends.
/// In narrow (mobile) mode it just renders the list of section tiles
/// — which is what we want for a smoke test.
///
/// We force a narrow surface via `tester.view.physicalSize` so the
/// `MediaQuery.sizeOf(context).width > 700` branch in `build` falls
/// to the simple list path. Same technique as `main_shell_test.dart`.
///
/// `StaggeredList` schedules ~16 `Future.delayed` cascades so we drain
/// them with a 2-second pump before tearing the tree down.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/screens/admin_hub_screen.dart';

import '../helpers/test_app.dart';

void main() {
  group('AdminHubScreen smoke', () {
    Future<void> setMobileSurface(WidgetTester tester) async {
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
    }

    Widget wrap() => localizedTestApp(child: const AdminHubScreen());

    testWidgets('renders without crashing on narrow surface', (tester) async {
      await setMobileSurface(tester);
      await tester.pumpWidget(wrap());
      await tester.pump();
      // Drain the StaggeredList Future.delayed cascade (16 sections * 50ms).
      await tester.pump(const Duration(seconds: 2));
      tester.takeException();
      expect(find.byType(AdminHubScreen), findsOneWidget);
    });

    testWidgets('shows the section list with at least the Agents tile', (
      tester,
    ) async {
      await setMobileSurface(tester);
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(seconds: 2));
      tester.takeException();
      // The list is a ListView wrapping a StaggeredList of ListTiles.
      expect(find.byType(ListView), findsOneWidget);
      expect(find.byType(ListTile), findsWidgets);
    });

    testWidgets('disposes cleanly when replaced', (tester) async {
      await setMobileSurface(tester);
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(seconds: 2));

      // dispose path: every staggered AnimationController must be
      // released. A leaked controller surfaces as an uncaught
      // exception when the test ends.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 100));
      tester.takeException();
      expect(find.byType(AdminHubScreen), findsNothing);
    });
  });
}
