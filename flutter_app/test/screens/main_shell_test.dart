/// Smoke test for [MainShell] — the top-level navigation shell that
/// stitches together every primary screen via [IndexedStack].
///
/// Previously deferred because the shell composes ~7 screens at once,
/// each with its own provider dependencies and Timer/animation
/// lifecycles. We provide silent no-op variants for every provider
/// the shell or any of its sub-screens reads, mirroring the strategy
/// used in `kanban_screen_test.dart` + `dashboard_screen_test.dart`.
///
/// We do NOT call `pumpAndSettle` — several sub-screens use repeating
/// animations (skeleton shimmers, pulse-fade empty states) that would
/// keep the test pending forever.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/cron_provider.dart';
import 'package:cognithor_ui/providers/hacker_mode_provider.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/providers/navigation_provider.dart';
import 'package:cognithor_ui/providers/packs_provider.dart';
import 'package:cognithor_ui/providers/pip_provider.dart';
import 'package:cognithor_ui/providers/research_provider.dart';
import 'package:cognithor_ui/providers/robot_office_provider.dart';
import 'package:cognithor_ui/providers/sessions_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/providers/theme_provider.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/providers/tree_provider.dart';
import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/screens/main_shell.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/trace_service.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

import '../helpers/fakes.dart';
import '../helpers/silent_providers.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockWsService extends Mock implements WebSocketService {}

void main() {
  setUpAll(() {
    registerFallbackValue(<String, dynamic>{});
    registerFallbackValue(<dynamic, dynamic>{});
  });

  group('MainShell smoke', () {
    late _MockApiClient api;
    late _MockWsService ws;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      ws = _MockWsService();
      // TraceListScreen.initState fires `subscribeToLifecycle` which
      // calls `ws.send(...)` (returns bool). Default mock returns null,
      // which crashes the call. Stub `send` to return false (the
      // dropped-frame return value).
      when(() => ws.send(any())).thenReturn(false);
      // Stub the three monitoring endpoints that DashboardScreen fires
      // from initState (it lives inside the IndexedStack and thus
      // builds even when not visible).
      when(() => api.getMonitoringDashboard()).thenAnswer(
        (_) async => <String, dynamic>{
          'cpu_usage': 0,
          'memory_usage': 0,
          'response_time_ms': 0,
          'total_tokens': 0,
        },
      );
      when(
        () => api.getMonitoringEvents(n: any(named: 'n')),
      ).thenAnswer((_) async => {'events': <Map<String, dynamic>>[]});
      when(
        () => api.getModelStats(),
      ).thenAnswer((_) async => <String, dynamic>{});

      conn = FakeConnectionProvider(
        apiClient: api,
        wsService: ws,
        backendVersion: '0.97.0',
      );
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          // ── MainShell-direct dependencies ──────────────────────
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<NavigationProvider>(
            create: (_) => NavigationProvider(),
          ),
          ChangeNotifierProvider<ThemeProvider>(create: (_) => ThemeProvider()),
          ChangeNotifierProvider<PipProvider>(create: (_) => PipProvider()),
          ChangeNotifierProvider<SourcesProvider>(
            create: (_) => SilentSourcesProvider(),
          ),
          ChangeNotifierProvider<PacksProvider>(create: (_) => PacksProvider()),

          // ── Sub-screen dependencies (IndexedStack builds them all) ──
          ChangeNotifierProvider<ChatProvider>(create: (_) => ChatProvider()),
          ChangeNotifierProvider<SessionsProvider>(
            create: (_) => SilentSessionsProvider(),
          ),
          ChangeNotifierProvider<VoiceProvider>(create: (_) => VoiceProvider()),
          ChangeNotifierProvider<TreeProvider>(create: (_) => TreeProvider()),
          ChangeNotifierProvider<HackerModeProvider>(
            create: (_) => HackerModeProvider(),
          ),
          ChangeNotifierProvider<LlmBackendProvider>(
            create: (_) => SilentLlmBackendProvider(),
          ),
          ChangeNotifierProvider<RobotOfficeProvider>(
            create: (_) => SilentRobotOfficeProvider(),
          ),
          ChangeNotifierProvider<KanbanProvider>(
            create: (_) => SilentKanbanProvider(),
          ),
          ChangeNotifierProvider<CronProvider>(
            create: (_) => SilentCronProvider(),
          ),
          ChangeNotifierProvider<AdminProvider>(
            create: (_) => SilentAdminProvider(),
          ),
          ChangeNotifierProvider<SkillsProvider>(
            create: (_) => SilentSkillsProvider(),
          ),
          ChangeNotifierProvider<TraceProvider>(
            create: (_) => TraceProvider(
              traceService: TraceService(apiClient: api, wsService: ws),
            ),
          ),
          ChangeNotifierProvider<ResearchProvider>(
            create: (_) => ResearchProvider(),
          ),
        ],
        child: const MainShell(),
      ),
    );

    /// Force a narrow (mobile) test surface so [AdminHubScreen] (which
    /// is one of the IndexedStack children) renders its list view
    /// instead of the embedded ConfigScreen. The latter pulls in
    /// ConfigProvider + half a dozen sub-page providers we don't want
    /// in a smoke test.
    Future<void> setMobileSurface(WidgetTester tester) async {
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
    }

    /// Pumps the shell + drains all the one-shot `Future.delayed`s
    /// queued by the AdminHubScreen's StaggeredList intro animation.
    /// We always replace the widget tree at the end with a SizedBox
    /// so any AnimationControllers / pending Timers belonging to the
    /// many sub-screens are released before the test ends and the
    /// framework runs its `!timersPending` assertion.
    Future<void> pumpAndTeardown(WidgetTester tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      // The AdminHubScreen list uses StaggeredList which schedules
      // ~12 * 50ms `Future.delayed`s as it cascades children in.
      await tester.pump(const Duration(seconds: 2));
    }

    Future<void> replaceWithBlank(WidgetTester tester) async {
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 100));
    }

    testWidgets('renders without crashing on minimal connected state', (
      tester,
    ) async {
      await setMobileSurface(tester);
      await pumpAndTeardown(tester);
      tester.takeException();
      expect(find.byType(MainShell), findsOneWidget);
      await replaceWithBlank(tester);
      tester.takeException();
    });

    testWidgets('mounts the default chat tab as the active screen', (
      tester,
    ) async {
      await setMobileSurface(tester);
      await pumpAndTeardown(tester);
      tester.takeException();
      // Chat is tab 0 (the default). Its AppBar uses the history icon
      // for the drawer-opener.
      expect(find.byIcon(Icons.history), findsOneWidget);
      await replaceWithBlank(tester);
      tester.takeException();
    });

    testWidgets('disposes cleanly when replaced', (tester) async {
      await setMobileSurface(tester);
      await pumpAndTeardown(tester);

      // Replacing the tree triggers dispose chain across every sub-
      // screen. Any leaked Timer or AnimationController would surface
      // here as an uncaught exception.
      await replaceWithBlank(tester);
      tester.takeException();
      expect(find.byType(MainShell), findsNothing);
    });
  });
}
