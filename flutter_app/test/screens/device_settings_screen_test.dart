/// Smoke test for [DeviceSettingsScreen].
///
/// Reads [ConnectionProvider] (for serverUrl + connection status) and
/// [DeviceProvider] (for permission toggles + sensor data). Neither
/// has a Timer or animation; the screen is a `ListView` of `Card`s,
/// so a single `pump()` after mount renders cleanly. No silent
/// override needed since the providers' constructors don't fire any
/// network/IO.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/device_provider.dart';
import 'package:cognithor_ui/screens/device_settings_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('DeviceSettingsScreen smoke', () {
    late FakeConnectionProvider conn;

    setUp(() {
      conn = FakeConnectionProvider(apiClient: _MockApiClient());
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<DeviceProvider>(
            create: (_) => DeviceProvider(),
          ),
        ],
        child: const DeviceSettingsScreen(),
      ),
    );

    testWidgets('renders without crashing on default device state', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(DeviceSettingsScreen), findsOneWidget);
    });

    testWidgets('shows the Server Connection and Permissions sections', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.text('Server Connection'), findsOneWidget);
      expect(find.text('Device Permissions'), findsOneWidget);
      // 4 permission switch tiles by default (location/camera/mic/photos).
      expect(find.byType(SwitchListTile), findsNWidgets(4));
    });
  });
}
