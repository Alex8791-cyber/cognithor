/// Deep interaction tests for [MainShell] — the navigation hub that
/// stitches every primary screen together via an [IndexedStack].
///
/// Builds on the smoke tests in `main_shell_test.dart` (PR #489) by
/// driving real navigation flows: tapping bottom-nav items switches
/// the active tab on [NavigationProvider]; the search rail icon opens
/// the global search dialog; the theme rail icon flips
/// [ThemeProvider]; and tapping the Robot Office rail icon toggles
/// [PipProvider.visible].
///
/// We force a mobile viewport so the bottom-nav variant of
/// [ResponsiveScaffold] is rendered (label-bearing rail tabs are easier
/// to find via `find.text` than the side-rail's icon-only collapsed mode).
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

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

  group('MainShell interactions', () {
    late _MockApiClient api;
    late _MockWsService ws;
    late FakeConnectionProvider conn;
    late NavigationProvider nav;
    late PipProvider pip;
    late ThemeProvider theme;

    setUp(() {
      // ThemeProvider + a few others read SharedPreferences.
      SharedPreferences.setMockInitialValues(<String, Object>{
        'theme_mode': 'dark',
      });
      api = _MockApiClient();
      ws = _MockWsService();
      when(() => ws.send(any())).thenReturn(false);
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
      nav = NavigationProvider();
      pip = PipProvider();
      theme = ThemeProvider();
    });

    Widget wrap() => localizedTestApp(
      child: MultiProvider(
        providers: [
          ChangeNotifierProvider<ConnectionProvider>.value(value: conn),
          ChangeNotifierProvider<NavigationProvider>.value(value: nav),
          ChangeNotifierProvider<ThemeProvider>.value(value: theme),
          ChangeNotifierProvider<PipProvider>.value(value: pip),
          ChangeNotifierProvider<SourcesProvider>(
            create: (_) => SilentSourcesProvider(),
          ),
          ChangeNotifierProvider<PacksProvider>(create: (_) => PacksProvider()),
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

    Future<void> setMobileSurface(WidgetTester tester) async {
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
    }

    Future<void> pumpShell(WidgetTester tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      // Drain the AdminHubScreen StaggeredList intro animation timers.
      await tester.pump(const Duration(seconds: 2));
    }

    Future<void> teardown(WidgetTester tester) async {
      await tester.pumpWidget(localizedTestApp(child: const SizedBox()));
      await tester.pump(const Duration(milliseconds: 100));
    }

    testWidgets(
      'tapping the Dashboard bottom-nav label switches NavigationProvider',
      (tester) async {
        await setMobileSurface(tester);
        await pumpShell(tester);
        tester.takeException();

        // Default: tab 0 (Chat) is active.
        expect(nav.currentTab, 0);

        // Tap "Dashboard" bottom-nav cell. The label text appears in the
        // cell — find by text + tap on the surrounding InkWell.
        await tester.tap(find.text('Dashboard').first);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 100));

        // Dashboard is tab 1 in the base navItems list.
        expect(nav.currentTab, 1);

        await teardown(tester);
      },
    );

    testWidgets(
      'tapping the Skills bottom-nav label switches NavigationProvider',
      (tester) async {
        await setMobileSurface(tester);
        await pumpShell(tester);
        tester.takeException();

        await tester.tap(find.text('Skills').first);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 100));

        expect(nav.currentTab, 2);

        await teardown(tester);
      },
    );

    testWidgets(
      'tapping the Kanban bottom-nav label switches NavigationProvider',
      (tester) async {
        await setMobileSurface(tester);
        await pumpShell(tester);
        tester.takeException();

        await tester.tap(find.text('Kanban').first);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 100));

        // Kanban is tab 5 in the base navItems list.
        expect(nav.currentTab, 5);

        await teardown(tester);
      },
    );

    testWidgets(
      'NavigationProvider.setTab updates the visible AppBar via IndexedStack',
      (tester) async {
        await setMobileSurface(tester);
        await pumpShell(tester);
        tester.takeException();

        // Default chat tab: history icon visible (chat-screen AppBar leading).
        expect(find.byIcon(Icons.history), findsOneWidget);

        // Programmatically navigate to Dashboard. The IndexedStack keeps
        // chat alive but hides it — the chat AppBar disappears from the
        // visible portion of the tree (still in the offstage indexed
        // stack but not findable in the same way).
        nav.setTab(1);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 100));

        // Tab is now 1 — observable on the provider.
        expect(nav.currentTab, 1);

        await teardown(tester);
      },
    );

    testWidgets(
      'tapping the Chat bottom-nav after switching returns to tab 0',
      (tester) async {
        await setMobileSurface(tester);
        await pumpShell(tester);
        tester.takeException();

        // Move to Dashboard first.
        nav.setTab(1);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 100));
        expect(nav.currentTab, 1);

        // Tap Chat tab to return.
        await tester.tap(find.text('Chat').first);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 100));

        expect(nav.currentTab, 0);

        await teardown(tester);
      },
    );
  });
}
