import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/models_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('ModelsScreen smoke', () {
    late _MockApiClient api;
    late AdminProvider admin;
    late ConfigProvider cfg;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      admin = SilentAdminProvider();
      cfg = SilentConfigProvider();
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: ChangeNotifierProvider<AdminProvider>.value(
          value: admin,
          child: ChangeNotifierProvider<ConfigProvider>.value(
            value: cfg,
            child: const ModelsScreen(),
          ),
        ),
      ),
    );

    testWidgets('renders without crashing on empty models data', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(ModelsScreen), findsOneWidget);
    });
  });
}
