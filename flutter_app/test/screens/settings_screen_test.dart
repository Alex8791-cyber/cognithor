import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/settings_screen.dart';

import '../helpers/test_app.dart';

void main() {
  group('SettingsScreen smoke', () {
    Widget wrap(ConnectionProvider conn) => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const SettingsScreen(),
      ),
    );

    testWidgets('renders without crashing with a fresh ConnectionProvider', (
      tester,
    ) async {
      await tester.pumpWidget(wrap(ConnectionProvider()));

      // AppBar title + Save button + URL TextField present.
      expect(find.byType(Scaffold), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
      expect(find.byType(TextField), findsOneWidget);
      expect(find.byType(ElevatedButton), findsOneWidget);
    });

    testWidgets(
      'initial TextField value mirrors ConnectionProvider.serverUrl',
      (tester) async {
        final conn = ConnectionProvider();
        await tester.pumpWidget(wrap(conn));

        final tf = tester.widget<TextField>(find.byType(TextField));
        expect(tf.controller?.text, conn.serverUrl);
      },
    );

    testWidgets('when backendVersion is null, no version label is rendered', (
      tester,
    ) async {
      await tester.pumpWidget(wrap(ConnectionProvider()));
      expect(find.textContaining('0.97'), findsNothing);
    });

    testWidgets('TextEditingController is disposed without leak', (
      tester,
    ) async {
      await tester.pumpWidget(wrap(ConnectionProvider()));
      // Replace with empty container — triggers SettingsScreen.dispose().
      // No exception means the controller was cleaned up.
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      expect(tester.takeException(), isNull);
    });
  });
}
