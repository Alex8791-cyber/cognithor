import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cognithor_ui/providers/theme_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('ThemeProvider', () {
    setUp(() {
      // Default to empty prefs — first-launch flow.
      SharedPreferences.setMockInitialValues({});
    });

    test('default mode is dark when no preference is stored', () async {
      final provider = ThemeProvider();
      // Allow the async _load() called from constructor to settle.
      await Future<void>.delayed(Duration.zero);

      expect(provider.mode, ThemeMode.dark);
      expect(provider.isDark, true);
    });

    test('persisted "light" preference loads as light mode', () async {
      SharedPreferences.setMockInitialValues({'theme_mode': 'light'});
      final provider = ThemeProvider();
      // Wait for the async _load() triggered by the constructor.
      // Two micro-pumps cover both `getInstance().then(...)` and the
      // notifyListeners() call.
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.mode, ThemeMode.light);
      expect(provider.isDark, false);
    });

    test('toggle flips mode + persists', () async {
      final provider = ThemeProvider();
      await Future<void>.delayed(Duration.zero);

      await provider.toggle();
      expect(provider.mode, ThemeMode.light);

      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getString('theme_mode'), 'light');

      await provider.toggle();
      expect(provider.mode, ThemeMode.dark);
      expect(prefs.getString('theme_mode'), 'dark');
    });

    test('toggle notifies listeners', () async {
      final provider = ThemeProvider();
      await Future<void>.delayed(Duration.zero);

      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      await provider.toggle();
      expect(notifyCount, greaterThanOrEqualTo(1));
    });
  });
}
