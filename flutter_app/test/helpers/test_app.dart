/// Test scaffolding helpers shared by screen smoke tests.
///
/// Most screens call ``AppLocalizations.of(context).<key>``; without the
/// localizationsDelegates wired up the call returns null and the screen
/// crashes during the first build. ``localizedTestApp`` wraps any widget
/// in a MaterialApp with the four standard delegates + en/de locales so
/// every smoke test can render without that boilerplate.
///
/// Callers wrap their own providers around the [child] before passing
/// it in — keeps the provider list type-clean without depending on the
/// transitive `nested` package directly.
library;

import 'package:flutter/material.dart';
// ignore: depend_on_referenced_packages
import 'package:flutter_localizations/flutter_localizations.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';

/// Wraps [child] in a MaterialApp configured for tests:
/// - localizations delegates (AppLocalizations + Material/Widgets/Cupertino)
/// - en + de supported locales (forced to [locale] for determinism)
///
/// Example:
/// ```dart
/// await tester.pumpWidget(localizedTestApp(
///   child: ChangeNotifierProvider<MyProvider>.value(
///     value: mock,
///     child: const MyScreen(),
///   ),
/// ));
/// ```
Widget localizedTestApp({
  required Widget child,
  Locale locale = const Locale('en'),
  ThemeData? theme,
}) {
  return MaterialApp(
    locale: locale,
    localizationsDelegates: const [
      AppLocalizations.delegate,
      GlobalMaterialLocalizations.delegate,
      GlobalWidgetsLocalizations.delegate,
      GlobalCupertinoLocalizations.delegate,
    ],
    supportedLocales: AppLocalizations.supportedLocales,
    theme: theme ?? ThemeData(useMaterial3: true),
    home: child,
  );
}
