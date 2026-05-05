import 'package:flutter_test/flutter_test.dart';
import 'package:permission_handler/permission_handler.dart';

import 'package:cognithor_ui/providers/device_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('DeviceProvider', () {
    test('initial state — every toggle is off and every datum null', () {
      final p = DeviceProvider();
      expect(p.locationEnabled, isFalse);
      expect(p.cameraEnabled, isFalse);
      expect(p.microphoneEnabled, isFalse);
      expect(p.photosEnabled, isFalse);
      expect(p.lastPosition, isNull);
      expect(p.batteryLevel, isNull);
      expect(p.networkType, isNull);
      expect(p.deviceInfo, isNull);
    });

    test('enablePermission flips the matching toggle and notifies', () {
      final p = DeviceProvider();
      var notified = 0;
      p.addListener(() => notified++);

      p.enablePermission(Permission.location);
      expect(p.locationEnabled, isTrue);

      p.enablePermission(Permission.camera);
      expect(p.cameraEnabled, isTrue);

      p.enablePermission(Permission.microphone);
      expect(p.microphoneEnabled, isTrue);

      p.enablePermission(Permission.photos);
      expect(p.photosEnabled, isTrue);

      // Unrecognized permissions are a no-op (default branch in switch).
      p.enablePermission(Permission.contacts);

      expect(notified, 5);
    });

    test('disablePermission turns toggles off again', () {
      final p = DeviceProvider();
      p
        ..enablePermission(Permission.location)
        ..enablePermission(Permission.camera)
        ..enablePermission(Permission.microphone)
        ..enablePermission(Permission.photos);

      expect(p.locationEnabled, isTrue);

      p
        ..disablePermission(Permission.location)
        ..disablePermission(Permission.camera)
        ..disablePermission(Permission.microphone)
        ..disablePermission(Permission.photos);

      expect(p.locationEnabled, isFalse);
      expect(p.cameraEnabled, isFalse);
      expect(p.microphoneEnabled, isFalse);
      expect(p.photosEnabled, isFalse);
    });

    test('getDeviceContext is empty when nothing is enabled', () {
      final p = DeviceProvider();
      expect(p.getDeviceContext(), isEmpty);
    });

    test('getDeviceContext only includes battery/network when populated', () {
      final p = DeviceProvider();
      // We can't easily set lastPosition (final getter on Position), but we
      // can verify the conditional shape via the other fields:
      // batteryLevel + networkType + deviceInfo are public mutable.
      p.batteryLevel = 88;
      p.networkType = 'wifi';
      p.deviceInfo = {'model': 'pc'};

      final ctx = p.getDeviceContext();
      expect(ctx['battery'], 88);
      expect(ctx['network'], 'wifi');
      expect(ctx['device'], {'model': 'pc'});
      // Location absent because locationEnabled is false.
      expect(ctx.containsKey('location'), isFalse);
    });

    test('recordAudio stub returns null without throwing', () async {
      final p = DeviceProvider();
      final result = await p.recordAudio();
      expect(result, isNull);
    });

    test(
      'getCurrentLocation returns null when permission is disabled',
      () async {
        final p = DeviceProvider();
        // locationEnabled is false by default → fast-path null without
        // touching the Geolocator platform plugin.
        final pos = await p.getCurrentLocation();
        expect(pos, isNull);
      },
    );

    test(
      'capturePhoto / pickPhoto return null when their permissions are off',
      () async {
        final p = DeviceProvider();
        expect(await p.capturePhoto(), isNull);
        expect(await p.pickPhoto(), isNull);
      },
    );
  });
}
