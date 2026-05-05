/// Smoke tests for the 8 stateful config sub-pages that were deferred
/// from `config_pages_smoke_test.dart`.
///
/// These pages are stateful and either:
///   1. Drive `ConnectionProvider.api` directly in `initState` (audit,
///      budget, context_profile, providers, system_profile), or
///   2. Use `Consumer<ConfigProvider>` but rely on populated cfg keys
///      (atl, language, system).
///
/// Pattern matches `system_screen_test.dart` / `vault_screen_test.dart`
/// (multi-provider wrap with mocktail-stubbed `ApiClient`).
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/locale_provider.dart';
import 'package:cognithor_ui/screens/config/atl_page.dart';
import 'package:cognithor_ui/screens/config/audit_page.dart';
import 'package:cognithor_ui/screens/config/budget_page.dart';
import 'package:cognithor_ui/screens/config/context_profile_page.dart';
import 'package:cognithor_ui/screens/config/language_page.dart';
import 'package:cognithor_ui/screens/config/providers_page.dart';
import 'package:cognithor_ui/screens/config/system_page.dart';
import 'package:cognithor_ui/screens/config/system_profile_page.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../../helpers/fakes.dart';
import '../../helpers/silent_providers.dart';
import '../../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  setUpAll(() {
    // mocktail wants any() registered for non-primitive types.
    registerFallbackValue(<String, dynamic>{});
  });

  group('AuditPage smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // initState fires `audit/timestamps` via post-frame callback.
      when(() => api.get('audit/timestamps')).thenAnswer(
        (_) async => {'tsa_enabled': false, 'count': 0, 'timestamps': []},
      );
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const Scaffold(body: AuditPage()),
      ),
    );

    testWidgets('renders without crashing on empty audit payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(AuditPage), findsOneWidget);
    });
  });

  group('BudgetPage smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // BudgetPage hits 3 endpoints in parallel from initState.
      when(() => api.get('budget/agents')).thenAnswer(
        (_) async => {
          'agents_today': <String, dynamic>{},
          'agents_week': <String, dynamic>{},
          'agents_month': <String, dynamic>{},
          'budgets': <String, dynamic>{},
        },
      );
      when(() => api.get('system/resources')).thenAnswer(
        (_) async => {
          'cpu_percent': 0,
          'ram_used_gb': 0,
          'ram_total_gb': 1,
          'ram_percent': 0,
        },
      );
      when(() => api.get('evolution/stats')).thenAnswer(
        (_) async => {
          'running': false,
          'is_idle': true,
          'cycles_today': 0,
          'total_cycles': 0,
          'total_skills_created': 0,
          'recent_results': <dynamic>[],
        },
      );
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const Scaffold(body: BudgetPage()),
      ),
    );

    testWidgets('renders without crashing on empty budget payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      // Wait for the parallel Future.wait to resolve.
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(BudgetPage), findsOneWidget);
    });
  });

  group('ContextProfilePage smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // _buildProfileTile() reads `num_ctx`, `temperature`, `top_p`,
      // `description` as required keys — so the available map must be
      // populated with at least one well-formed entry, even though the
      // smoke test doesn't tap a tile.
      when(() => api.get('v1/context_profile')).thenAnswer(
        (_) async => {
          'active': null,
          'available': {
            'default': {
              'num_ctx': 8192,
              'temperature': 0.7,
              'top_p': 0.9,
              'description': 'Default profile',
            },
          },
        },
      );
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const Scaffold(body: ContextProfilePage()),
      ),
    );

    testWidgets('renders without crashing with one populated profile', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(ContextProfilePage), findsOneWidget);
    });
  });

  group('LanguagePage smoke', () {
    // LanguagePage uses ConfigProvider + LocaleProvider + ConnectionProvider.
    // Initial render only reads `cfg.cfg['language']` — empty cfg ok.
    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConfigProvider>.value(
            value: SilentConfigProvider(),
          ),
          ChangeNotifierProvider<LocaleProvider>(
            create: (_) => LocaleProvider(),
          ),
        ],
        child: const Scaffold(body: LanguagePage()),
      ),
    );

    testWidgets('renders without crashing on empty cfg', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(LanguagePage), findsOneWidget);
    });
  });

  group('ProvidersPage smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // _CurrentBackendCard.initState calls api.getBackendStatus().
      when(
        () => api.getBackendStatus(),
      ).thenAnswer((_) async => {'backends': <String, dynamic>{}});
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConfigProvider>.value(
            value: SilentConfigProvider(),
          ),
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
        ],
        child: const Scaffold(body: ProvidersPage()),
      ),
    );

    testWidgets('renders without crashing on empty cfg + empty backends', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(ProvidersPage), findsOneWidget);
    });
  });

  group('SystemConfigPage smoke', () {
    // SystemConfigPage only reads `cfg.cfg[...]` defaults on initial
    // render — ConnectionProvider is only used inside button-tap
    // callbacks, which the smoke test never triggers.
    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConfigProvider>.value(
        value: SilentConfigProvider(),
        child: const Scaffold(body: SystemConfigPage()),
      ),
    );

    testWidgets('renders without crashing on empty cfg', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(SystemConfigPage), findsOneWidget);
    });
  });

  group('SystemProfilePage smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      // initState calls api.get('system/profile').
      when(() => api.get('system/profile')).thenAnswer(
        (_) async => {
          'tier': 'standard',
          'recommended_mode': 'auto',
          'results': <String, dynamic>{},
        },
      );
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const Scaffold(body: SystemProfilePage()),
      ),
    );

    testWidgets('renders without crashing on minimal profile payload', (
      tester,
    ) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(SystemProfilePage), findsOneWidget);
    });
  });

  group('AtlPage smoke', () {
    // AtlPage reads `cfg.cfg['atl']` with fallback to {} — empty cfg ok.
    // The `enabled` toggle is false by default, so the conditional
    // sub-fields don't render in the empty case.
    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConfigProvider>.value(
        value: SilentConfigProvider(),
        child: const Scaffold(body: AtlPage()),
      ),
    );

    testWidgets('renders without crashing on empty cfg', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      tester.takeException();
      expect(find.byType(AtlPage), findsOneWidget);
    });
  });
}
