/// Deep interaction tests for [SettingsScreen].
///
/// Builds on the smoke tests in `settings_screen_test.dart` (PR #489) by
/// driving real user flows on the small settings surface: typing into
/// the URL field, tapping Save invokes [ConnectionProvider.setServerUrl],
/// the version label appears once a backendVersion is supplied, the
/// AppBar title is localized, and the same TextField round-trips the
/// edited URL back into the controller.
///
/// We use a [_RecordingConnectionProvider] subclass to observe
/// [setServerUrl] calls without spinning up a real HTTP/WS stack.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/settings_screen.dart';

import '../helpers/test_app.dart';

/// ConnectionProvider that records [setServerUrl] calls without going
/// near `connect()` (which would touch HTTP). We override the entry
/// point and store what was asked of us; the parent class still owns
/// the `serverUrl` field which we update directly so the provider's
/// `notifyListeners()` semantics stay realistic.
class _RecordingConnectionProvider extends ConnectionProvider {
  _RecordingConnectionProvider({String initialUrl = 'http://localhost:8741'}) {
    serverUrl = initialUrl;
  }

  int setServerUrlCalls = 0;
  String? lastUrlSet;

  @override
  Future<void> setServerUrl(String url) async {
    setServerUrlCalls++;
    lastUrlSet = url;
    final clean = url.trimRight().replaceAll(RegExp(r'/+$'), '');
    if (clean == serverUrl) return;
    serverUrl = clean;
    notifyListeners();
  }
}

void main() {
  group('SettingsScreen interactions', () {
    late _RecordingConnectionProvider conn;

    setUp(() {
      // Some downstream initializers under test indirectly hit
      // SharedPreferences.
      SharedPreferences.setMockInitialValues(<String, Object>{});
      conn = _RecordingConnectionProvider(initialUrl: 'http://localhost:8741');
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const SettingsScreen(),
      ),
    );

    Future<void> teardown(WidgetTester tester) async {
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 50));
    }

    testWidgets(
      'editing the URL field and tapping Save dispatches setServerUrl',
      (tester) async {
        await tester.pumpWidget(wrap());
        await tester.pump();

        // The single TextField on the screen is the URL editor.
        await tester.enterText(
          find.byType(TextField),
          'http://example.local:9000',
        );
        await tester.pump();

        // Tap the localized "Save" button (en locale → "Save").
        await tester.tap(find.text('Save'));
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 50));

        expect(conn.setServerUrlCalls, 1);
        expect(conn.lastUrlSet, 'http://example.local:9000');

        await teardown(tester);
      },
    );

    testWidgets(
      'submitting the URL field via onSubmitted dispatches setServerUrl',
      (tester) async {
        await tester.pumpWidget(wrap());
        await tester.pump();

        // The onSubmitted callback wires straight into setServerUrl.
        await tester.enterText(
          find.byType(TextField),
          'http://other.host:8000',
        );
        await tester.testTextInput.receiveAction(TextInputAction.done);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 50));

        expect(conn.setServerUrlCalls, 1);
        expect(conn.lastUrlSet, 'http://other.host:8000');

        await teardown(tester);
      },
    );

    testWidgets('AppBar title is the localized "Settings" string', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();

      // AppBar title text comes from AppLocalizations.of(context).settings
      // — for the en locale the test app forces, that is "Settings".
      expect(find.widgetWithText(AppBar, 'Settings'), findsOneWidget);

      await teardown(tester);
    });

    testWidgets(
      'version label appears when ConnectionProvider exposes a backendVersion',
      (tester) async {
        // Promote the provider to a connected-with-version state.
        conn.backendVersion = '0.97.0';
        // Also make sure rebuild happens — context.watch picks up changes.

        await tester.pumpWidget(wrap());
        await tester.pump();

        // The version copy is interpolated via AppLocalizations.version(...)
        // which embeds the supplied version. We only need to verify that
        // "0.97.0" is present somewhere in the rendered tree — the exact
        // localized template differs across locales.
        expect(find.textContaining('0.97.0'), findsOneWidget);

        await teardown(tester);
      },
    );

    testWidgets('editing the URL field updates the TextField controller text', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();

      await tester.enterText(
        find.byType(TextField),
        'http://updated.host:1234',
      );
      await tester.pump();

      // The controller text reflects the typed value (round-trip).
      final tf = tester.widget<TextField>(find.byType(TextField));
      expect(tf.controller?.text, 'http://updated.host:1234');
      // Also visible in the rendered tree.
      expect(find.text('http://updated.host:1234'), findsOneWidget);

      await teardown(tester);
    });
  });
}
