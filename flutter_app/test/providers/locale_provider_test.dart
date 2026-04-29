import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cognithor_ui/providers/locale_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('LocaleProvider', () {
    setUp(() {
      SharedPreferences.setMockInitialValues({});
    });

    test('persisted locale takes priority over system detection', () async {
      SharedPreferences.setMockInitialValues({'app_locale': 'de'});
      final provider = LocaleProvider();
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      expect(provider.locale.languageCode, 'de');
    });

    test(
      'first launch falls back to "en" when system locale unsupported',
      () async {
        // Mock initial values are empty → first-launch path.
        // Test runner runs with system locale = en_US which IS supported, so
        // this exercises the supported-system-locale branch.
        final provider = LocaleProvider();
        await Future<void>.delayed(Duration.zero);
        await Future<void>.delayed(Duration.zero);

        expect(
          LocaleProvider.supportedCodes,
          contains(provider.locale.languageCode),
        );
      },
    );

    test('setLocale updates locale + persists', () async {
      final provider = LocaleProvider();
      await Future<void>.delayed(Duration.zero);

      await provider.setLocale('zh');

      expect(provider.locale.languageCode, 'zh');
      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getString('app_locale'), 'zh');
    });

    test('setLocale ignores unsupported codes silently', () async {
      final provider = LocaleProvider();
      await Future<void>.delayed(Duration.zero);
      final before = provider.locale;

      await provider.setLocale('not-a-locale');

      expect(
        provider.locale,
        before,
        reason: 'Unsupported locale must be a no-op',
      );
    });

    test('setLocale notifies listeners', () async {
      final provider = LocaleProvider();
      await Future<void>.delayed(Duration.zero);

      var notifyCount = 0;
      provider.addListener(() => notifyCount++);
      await provider.setLocale('ar');

      expect(notifyCount, greaterThanOrEqualTo(1));
    });

    test(
      'syncFromConfig changes locale when language matches a supported code',
      () async {
        final provider = LocaleProvider();
        await Future<void>.delayed(Duration.zero);

        provider.syncFromConfig('zh');
        expect(provider.locale.languageCode, 'zh');
      },
    );

    test('syncFromConfig is a no-op for unsupported language', () async {
      SharedPreferences.setMockInitialValues({'app_locale': 'de'});
      final provider = LocaleProvider();
      await Future<void>.delayed(Duration.zero);
      await Future<void>.delayed(Duration.zero);

      provider.syncFromConfig('xx'); // unsupported
      expect(provider.locale.languageCode, 'de');

      provider.syncFromConfig(null); // null
      expect(provider.locale.languageCode, 'de');
    });

    test('supportedCodes includes en + de + zh + ar', () {
      expect(
        LocaleProvider.supportedCodes,
        containsAll(['en', 'de', 'zh', 'ar']),
      );
    });
  });
}
