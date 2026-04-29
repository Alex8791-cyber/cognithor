import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/providers/pip_provider.dart';

void main() {
  group('PipProvider', () {
    late PipProvider provider;

    setUp(() {
      provider = PipProvider();
    });

    test('initial state: visible, not busy, not fullscreen', () {
      expect(provider.visible, true);
      expect(provider.busy, false);
      expect(provider.fullscreenOnDashboard, false);
    });

    test('setBusy(true) toggles + notifies', () {
      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      provider.setBusy(true);
      expect(provider.busy, true);
      expect(notifyCount, 1);
    });

    test('setBusy with same value is a no-op (no notify)', () {
      provider.setBusy(true);
      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      provider.setBusy(true);

      expect(notifyCount, 0);
    });

    test('hide / show toggle visibility', () {
      provider.hide();
      expect(provider.visible, false);

      provider.show();
      expect(provider.visible, true);
    });

    test('toggle flips visibility', () {
      expect(provider.visible, true);
      provider.toggle();
      expect(provider.visible, false);
      provider.toggle();
      expect(provider.visible, true);
    });

    test('enterFullscreen hides PiP + sets fullscreen', () {
      provider.enterFullscreen();

      expect(provider.fullscreenOnDashboard, true);
      expect(provider.visible, false);
    });

    test('exitFullscreen restores PiP + clears fullscreen', () {
      provider.enterFullscreen();
      provider.exitFullscreen();

      expect(provider.fullscreenOnDashboard, false);
      expect(provider.visible, true);
    });

    test('every state mutation calls notifyListeners exactly once', () {
      var notifyCount = 0;
      provider.addListener(() => notifyCount++);

      provider.hide();
      provider.show();
      provider.toggle();
      provider.toggle();
      provider.enterFullscreen();
      provider.exitFullscreen();
      provider.setBusy(true);

      expect(notifyCount, 7);
    });
  });
}
