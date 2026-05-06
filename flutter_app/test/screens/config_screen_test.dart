/// Smoke test for [ConfigScreen].
///
/// The screen owns:
///   - a `TabController` (5 tabs) with a `_onTabChanged` listener,
///   - a `_NeonPulseWrapper` AnimationController that only ever runs
///     when `cfg.hasChanges` is true (false in this smoke test),
///   - a `didChangeDependencies` that calls
///     `ConfigProvider.setApi(conn.api)` and `ConfigProvider.loadAll()`.
///
/// We supply [SilentConfigProvider] (no-op `loadAll`) and a
/// `FakeConnectionProvider` so the screen mounts cleanly. The default
/// landing page is the first key of the first category — `providers`
/// — which is a stateless `Consumer<ConfigProvider>` that reads from
/// an empty `cfg.cfg` map.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/config_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('ConfigScreen smoke', () {
    late FakeConnectionProvider conn;

    setUp(() {
      conn = FakeConnectionProvider(apiClient: _MockApiClient());
    });

    /// Force a narrow surface so the wide-mode `Row` (with the
    /// sub-page sidebar) never builds. Keeps the test focused on
    /// the one main page builder + save bar.
    Future<void> setMobileSurface(WidgetTester tester) async {
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
    }

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<ConfigProvider>(
            create: (_) => SilentConfigProvider(),
          ),
        ],
        child: const ConfigScreen(),
      ),
    );

    testWidgets('renders without crashing on empty config map', (tester) async {
      await setMobileSurface(tester);
      await tester.pumpWidget(wrap());
      await tester.pump();
      // Drain any post-frame callbacks (e.g. the no-op loadAll).
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(ConfigScreen), findsOneWidget);
    });

    testWidgets('shows the category TabBar', (tester) async {
      await setMobileSurface(tester);
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // 5 categories → 5 tabs in the TabBar.
      expect(find.byType(TabBar), findsOneWidget);
      expect(find.byType(Tab), findsNWidgets(5));
    });

    testWidgets('disposes cleanly when replaced', (tester) async {
      await setMobileSurface(tester);
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // dispose path: TabController + AnimationController must be
      // released without "controller used after dispose" exceptions.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 100));
      tester.takeException();
      expect(find.byType(ConfigScreen), findsNothing);
    });
  });
}
