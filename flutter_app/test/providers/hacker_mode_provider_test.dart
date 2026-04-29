import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cognithor_ui/providers/hacker_mode_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('HackerModeProvider', () {
    setUp(() {
      SharedPreferences.setMockInitialValues({});
    });

    test('default enabled is false when no preference is stored', () async {
      final provider = HackerModeProvider();
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.enabled, false);
    });

    test('persisted true preference loads as enabled', () async {
      SharedPreferences.setMockInitialValues({'hacker_mode_enabled': true});
      final provider = HackerModeProvider();
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.enabled, true);
    });

    test('toggle flips state + persists', () async {
      final provider = HackerModeProvider();
      await Future<void>.delayed(Duration.zero);

      await provider.toggle();
      expect(provider.enabled, true);

      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getBool('hacker_mode_enabled'), true);

      await provider.toggle();
      expect(provider.enabled, false);
      expect(prefs.getBool('hacker_mode_enabled'), false);
    });

    test('toggle notifies listeners', () async {
      final provider = HackerModeProvider();
      await Future<void>.delayed(Duration.zero);

      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      await provider.toggle();
      expect(notifyCount, greaterThanOrEqualTo(1));
    });
  });
}
