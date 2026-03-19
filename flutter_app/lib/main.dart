import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/admin_provider.dart';
import 'package:jarvis_ui/providers/config_provider.dart';
import 'package:jarvis_ui/providers/locale_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/memory_provider.dart';
import 'package:jarvis_ui/providers/navigation_provider.dart';
import 'package:jarvis_ui/providers/security_provider.dart';
import 'package:jarvis_ui/providers/skills_provider.dart';
import 'package:jarvis_ui/providers/theme_provider.dart';
import 'package:jarvis_ui/providers/hacker_mode_provider.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/providers/voice_provider.dart';
import 'package:jarvis_ui/providers/workflow_provider.dart';
import 'package:jarvis_ui/screens/splash_screen.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const JarvisApp());
}

class JarvisApp extends StatelessWidget {
  const JarvisApp({super.key});

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
      ],
      child: Consumer2<ThemeProvider, LocaleProvider>(
        builder: (context, themeProvider, localeProvider, _) {
          return MaterialApp(
            title: 'Jarvis',
            debugShowCheckedModeBanner: false,
            theme: JarvisTheme.light,
            darkTheme: JarvisTheme.dark,
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
