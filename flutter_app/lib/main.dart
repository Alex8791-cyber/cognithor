import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/locale_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/memory_provider.dart';
import 'package:cognithor_ui/providers/navigation_provider.dart';
import 'package:cognithor_ui/providers/security_provider.dart';
import 'package:cognithor_ui/providers/sessions_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/providers/theme_provider.dart';
import 'package:cognithor_ui/providers/hacker_mode_provider.dart';
import 'package:cognithor_ui/providers/pip_provider.dart';
import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/providers/device_provider.dart';
import 'package:cognithor_ui/providers/tree_provider.dart';
import 'package:cognithor_ui/providers/workflow_provider.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/providers/evolution_provider.dart';
import 'package:cognithor_ui/providers/robot_office_provider.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/providers/cron_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/providers/packs_provider.dart';
import 'package:cognithor_ui/providers/research_provider.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/services/trace_service.dart';
import 'package:cognithor_ui/screens/splash_screen.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CognithorApp());
}

class CognithorApp extends StatelessWidget {
  const CognithorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ConnectionProvider()..init()),
        ChangeNotifierProvider(create: (_) => AdminProvider()),
        ChangeNotifierProvider(create: (_) => SkillsProvider()),
        ChangeNotifierProvider(create: (_) => MemoryProvider()),
        ChangeNotifierProvider(create: (_) => SecurityProvider()),
        ChangeNotifierProvider(create: (_) => WorkflowProvider()),
        ChangeNotifierProvider(create: (_) => ConfigProvider()),
        ChangeNotifierProvider(create: (_) => LocaleProvider()),
        ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ChangeNotifierProvider(create: (_) => VoiceProvider()),
        ChangeNotifierProvider(create: (_) => NavigationProvider()),
        ChangeNotifierProvider(create: (_) => PipProvider()),
        ChangeNotifierProvider(create: (_) => HackerModeProvider()),
        ChangeNotifierProvider(create: (_) => ChatProvider()),
        ChangeNotifierProvider(create: (_) => DeviceProvider()),
        ChangeNotifierProvider(create: (_) => SessionsProvider()),
        ChangeNotifierProvider(create: (_) => TreeProvider()),
        ChangeNotifierProvider(create: (_) => KanbanProvider()),
        ChangeNotifierProvider(create: (_) => CronProvider()),
        ChangeNotifierProvider(create: (_) => EvolutionProvider()),
        ChangeNotifierProvider(create: (_) => RobotOfficeProvider()),
        ChangeNotifierProvider(create: (_) => RedditLeadsProvider()),
        ChangeNotifierProvider(create: (_) => SourcesProvider()),
        ChangeNotifierProvider(create: (_) => PacksProvider()),
        ChangeNotifierProvider(create: (_) => ResearchProvider()),
        ChangeNotifierProxyProvider<ConnectionProvider, TraceProvider?>(
          create: (_) => null,
          update: (_, conn, prev) {
            if (conn.state != CognithorConnectionState.connected) return null;
            if (prev != null) return prev;
            return TraceProvider(
              traceService: TraceService(
                apiClient: conn.api,
                wsService: conn.ws,
              ),
            );
          },
        ),
      ],
      child: Consumer2<ThemeProvider, LocaleProvider>(
        builder: (context, themeProvider, localeProvider, _) {
          return MaterialApp(
            title: 'Cognithor',
            debugShowCheckedModeBanner: false,
            theme: CognithorTheme.light,
            darkTheme: CognithorTheme.dark,
            themeMode: themeProvider.mode,
            locale: localeProvider.locale,
            localizationsDelegates: const [
              AppLocalizations.delegate,
              GlobalMaterialLocalizations.delegate,
              GlobalWidgetsLocalizations.delegate,
              GlobalCupertinoLocalizations.delegate,
            ],
            supportedLocales: AppLocalizations.supportedLocales,
            home: const SplashScreen(),
          );
        },
      ),
    );
  }
}
