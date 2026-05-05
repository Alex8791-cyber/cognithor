import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';

void main() {
  group('ConnectionProvider', () {
    test('initial state is disconnected with default URL', () {
      final p = ConnectionProvider();
      expect(p.state, CognithorConnectionState.disconnected);
      expect(p.serverUrl.isNotEmpty, isTrue);
      expect(p.errorMessage, isNull);
      expect(p.backendVersion, isNull);
      expect(p.versionMismatch, isFalse);
      expect(p.wasConnected, isFalse);
    });

    test('frontendVersion exposes kFrontendVersion constant', () {
      final p = ConnectionProvider();
      expect(p.frontendVersion, kFrontendVersion);
      // Sanity-check the constant follows X.Y.Z shape — guards against
      // a regression where the bump-script forgets to keep it
      // in semver form (see release flow in CLAUDE.md).
      expect(RegExp(r'^\d+\.\d+\.\d+$').hasMatch(kFrontendVersion), isTrue);
    });

    test('default URL is HTTP-form ending without trailing slash', () {
      final p = ConnectionProvider();
      expect(p.serverUrl.startsWith('http'), isTrue);
      expect(p.serverUrl.endsWith('/'), isFalse);
    });

    test('api / ws getters throw before connect is called', () {
      final p = ConnectionProvider();
      // _api and _ws are null until connect() succeeds. The bang-getters
      // must not silently return null — they should raise.
      expect(() => p.api, throwsA(isA<TypeError>()));
      expect(() => p.ws, throwsA(isA<TypeError>()));
    });

    test('multiple instances are independent', () {
      final a = ConnectionProvider();
      final b = ConnectionProvider();
      expect(identical(a, b), isFalse);
      expect(a.serverUrl, b.serverUrl);
    });

    test('CognithorConnectionState enum has all 4 states', () {
      // Lock the enum surface — adding a state should be a deliberate
      // change because the splash + connection guards switch on it.
      expect(CognithorConnectionState.values.length, 4);
      expect(
        CognithorConnectionState.values,
        containsAll([
          CognithorConnectionState.disconnected,
          CognithorConnectionState.connecting,
          CognithorConnectionState.connected,
          CognithorConnectionState.error,
        ]),
      );
    });
  });
}
