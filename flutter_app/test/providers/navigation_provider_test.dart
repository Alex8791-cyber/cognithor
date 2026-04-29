import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/providers/navigation_provider.dart';

void main() {
  group('NavigationProvider', () {
    late NavigationProvider provider;

    setUp(() {
      provider = NavigationProvider();
    });

    test('initial tab is 0', () {
      expect(provider.currentTab, 0);
    });

    test('setTab changes currentTab and notifies listeners', () {
      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      provider.setTab(2);

      expect(provider.currentTab, 2);
      expect(notifyCount, 1);
    });

    test('setTab is a no-op when index is unchanged', () {
      provider.setTab(1);
      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      provider.setTab(1);

      expect(provider.currentTab, 1);
      expect(
        notifyCount,
        0,
        reason: 'No listeners should fire if tab is already at requested index',
      );
    });

    test('sidebarWidth is 220 on Admin tab (index 3), 180 elsewhere', () {
      provider.setTab(0);
      expect(provider.sidebarWidth, 180);

      provider.setTab(3);
      expect(provider.sidebarWidth, 220);

      provider.setTab(5);
      expect(provider.sidebarWidth, 180);
    });

    test('sectionColor + sectionName resolve via theme helper', () {
      // We don't assert specific colors (theme tokens are owned by
      // CognithorTheme); we just verify the provider exposes the helper
      // pass-through and returns something non-null for a few tab indices.
      for (final index in [0, 1, 2, 3]) {
        provider.setTab(index);
        expect(provider.sectionColor, isNotNull);
        expect(provider.sectionName, isNotEmpty);
      }
    });
  });
}
