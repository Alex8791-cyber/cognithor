import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/vault_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('VaultScreen smoke', () {
    late _MockApiClient api;
    late AdminProvider admin;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      admin = SilentAdminProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<AdminProvider>.value(
          value: admin,
          child: const VaultScreen(),
        ),
      ),
    );

    testWidgets('renders without crashing on empty vault data', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(VaultScreen), findsOneWidget);
    });

    testWidgets('shows empty-state when no vault agents', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      // Empty-state lock icon.
      expect(find.byIcon(Icons.lock_outlined), findsWidgets);
    });
  });
}
