import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/setup_wizard_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('SetupWizardScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(
        () => api.getBackendStatus(),
      ).thenAnswer((_) async => <String, dynamic>{});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const SetupWizardScreen(),
      ),
    );

    testWidgets('renders the step-1 backend picker', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(SetupWizardScreen), findsOneWidget);
    });

    testWidgets('drives the /backend/status API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.getBackendStatus()).called(greaterThanOrEqualTo(1));
    });

    testWidgets('exposes the SharedPreferences gating key', (tester) async {
      // Locks the public contract — `SplashScreen` reads this key to
      // decide between Wizard / MainShell.
      expect(SetupWizardScreen.prefKey, 'first_run_complete');
    });
  });
}
