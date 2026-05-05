import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/splash_screen.dart';

import '../helpers/test_app.dart';

/// SplashScreen renders a state-machine on top of ConnectionProvider:
/// disconnected (waiting), connecting (spinner), error (cloud-off icon
/// + retry/settings buttons), connected (auto-navigate — out of scope
/// for these smoke tests because the auto-navigation calls
/// `prefs.SharedPreferences` + `conn.ws` which is hard to stub
/// determinately). We exercise the three non-navigating branches.
class _StateConn extends ConnectionProvider {
  _StateConn(
    CognithorConnectionState s, {
    this.versionMismatchOverride = false,
  }) {
    state = s;
  }
  final bool versionMismatchOverride;

  @override
  bool get versionMismatch => versionMismatchOverride;
}

void main() {
  group('SplashScreen smoke', () {
    Widget wrap(ConnectionProvider conn) => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const SplashScreen(),
      ),
    );

    testWidgets('disconnected state shows the connecting label', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(_StateConn(CognithorConnectionState.disconnected)),
      );
      expect(find.byType(SplashScreen), findsOneWidget);
      expect(find.byType(Scaffold), findsOneWidget);
      // No retry buttons in disconnected state.
      expect(find.byIcon(Icons.refresh), findsNothing);
    });

    testWidgets('connecting state renders a CircularProgressIndicator', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(_StateConn(CognithorConnectionState.connecting)),
      );
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('error state renders cloud-off icon + retry/settings buttons', (
      tester,
    ) async {
      await tester.pumpWidget(wrap(_StateConn(CognithorConnectionState.error)));
      expect(find.byIcon(Icons.cloud_off), findsOneWidget);
      expect(find.byIcon(Icons.refresh), findsOneWidget);
      expect(find.byIcon(Icons.settings), findsOneWidget);
    });

    testWidgets('version-mismatch error renders update-icon + version labels', (
      tester,
    ) async {
      final conn = _StateConn(
        CognithorConnectionState.error,
        versionMismatchOverride: true,
      )..backendVersion = '0.50.0';

      await tester.pumpWidget(wrap(conn));
      expect(find.byIcon(Icons.system_update_alt), findsOneWidget);
      expect(find.text('Version Mismatch'), findsOneWidget);
      // Frontend version label should mention the kFrontendVersion.
      expect(find.textContaining('Frontend version:'), findsOneWidget);
      expect(find.textContaining('Backend version: 0.50.0'), findsOneWidget);
    });

    testWidgets('app title is rendered', (tester) async {
      await tester.pumpWidget(
        wrap(_StateConn(CognithorConnectionState.disconnected)),
      );
      // App title comes from AppLocalizations.appTitle ("Cognithor"
      // for the default en locale).
      expect(find.text('Cognithor'), findsOneWidget);
    });
  });
}
