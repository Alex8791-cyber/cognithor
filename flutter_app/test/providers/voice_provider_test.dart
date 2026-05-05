import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/voice_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/voice_service.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('VoiceProvider', () {
    test('constructs with VoiceState.off and inactive', () {
      final p = VoiceProvider();
      expect(p.state, VoiceState.off);
      expect(p.isActive, isFalse);
      expect(p.lastTranscript, isEmpty);
      expect(p.errorMessage, isNull);
    });

    test('constructor accepts an apiClient and a sendToChat callback', () {
      final api = _MockApiClient();
      String? captured;
      final p = VoiceProvider(
        apiClient: api,
        sendToChat: (text) => captured = text,
      );
      expect(p.apiClient, same(api));
      // Callback installed but not called yet.
      expect(captured, isNull);
    });

    test('speakResponse with no apiClient is a silent no-op', () async {
      final p = VoiceProvider();
      // Should not throw; just logs via debugPrint.
      await p.speakResponse('hello');
      // State must remain off (no playback was attempted).
      expect(p.state, VoiceState.off);
    });

    test(
      'speakResponse uses the provider apiClient when none passed',
      () async {
        final api = _MockApiClient();
        when(() => api.synthesizeSpeech(any())).thenAnswer((_) async => null);

        final p = VoiceProvider(apiClient: api);
        await p.speakResponse('hi');

        verify(() => api.synthesizeSpeech('hi')).called(1);
      },
    );

    test('speakResponse prefers explicit api argument over field', () async {
      final fallback = _MockApiClient();
      final preferred = _MockApiClient();
      when(
        () => preferred.synthesizeSpeech(any()),
      ).thenAnswer((_) async => null);

      final p = VoiceProvider(apiClient: fallback);
      await p.speakResponse('hi', preferred);

      verify(() => preferred.synthesizeSpeech('hi')).called(1);
      verifyNever(() => fallback.synthesizeSpeech(any()));
    });

    test('speakResponse swallows synthesizeSpeech exception', () async {
      final api = _MockApiClient();
      when(
        () => api.synthesizeSpeech(any()),
      ).thenThrow(Exception('network down'));

      final p = VoiceProvider(apiClient: api);
      // Must not propagate.
      await p.speakResponse('hi');

      expect(p.state, VoiceState.off);
    });

    test('speakResponse with empty bytes returns without playback', () async {
      final api = _MockApiClient();
      when(() => api.synthesizeSpeech(any())).thenAnswer((_) async => <int>[]);

      final p = VoiceProvider(apiClient: api);
      await p.speakResponse('hi');
      // Empty bytes → playTts is not called → state stays off.
      expect(p.state, VoiceState.off);
    });
  });
}
