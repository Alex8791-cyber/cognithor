// ignore_for_file: experimental_member_use

/// Voice mode service with 5-state machine, wake word detection,
/// German phonetic normalization, real STT via speech_to_text,
/// and TTS playback via just_audio.
library;

import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:just_audio/just_audio.dart';
import 'package:speech_to_text/speech_recognition_error.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';

enum VoiceState { off, listening, conversation, processing, speaking }

class VoiceService extends ChangeNotifier {
  VoiceService();

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  VoiceState _state = VoiceState.off;
  String _lastTranscript = '';
  String _wakeWord = 'jarvis';
  String? _errorMessage;
  final List<String> _endPhrases = ['jarvis ende', 'jarvis stop', 'ende'];

  VoiceState get state => _state;
  String get lastTranscript => _lastTranscript;
  String? get errorMessage => _errorMessage;

  // ---------------------------------------------------------------------------
  // Plugins
  // ---------------------------------------------------------------------------

  final SpeechToText _speech = SpeechToText();
  final AudioPlayer _player = AudioPlayer();
  bool _speechAvailable = false;

  // ---------------------------------------------------------------------------
  // Restart / error tracking
  // ---------------------------------------------------------------------------

  int _consecutiveErrors = 0;
  Timer? _restartTimer;
  Timer? _conversationTimeoutTimer;
  bool _disposed = false;

  /// Conversation timeout in seconds: if no input for this long while in
  /// CONVERSATION state, fall back to LISTENING.
  static const int _conversationTimeoutSec = 45;

  // ---------------------------------------------------------------------------
  // State transitions
  // ---------------------------------------------------------------------------

  void _setState(VoiceState s) {
    if (_disposed) return;
    _state = s;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /// Initialize STT engine and start listening for the wake word.
  Future<void> start() async {
    _errorMessage = null;

    try {
      _speechAvailable = await _speech.initialize(
        onError: _onSpeechError,
        onStatus: _onSpeechStatus,
      );
    } catch (e) {
      _speechAvailable = false;
      _errorMessage = 'Speech recognition init failed: $e';
      notifyListeners();
      return;
    }

    if (!_speechAvailable) {
      _errorMessage =
          'Speech recognition is not available on this device/browser.';
      notifyListeners();
      return;
    }

    _consecutiveErrors = 0;
    _setState(VoiceState.listening);
    _startListening();
  }

  /// Stop everything and go to OFF.
  Future<void> stop() async {
    _cancelTimers();
    await _speech.stop();
    await _player.stop();
    _lastTranscript = '';
    _errorMessage = null;
    _setState(VoiceState.off);
  }

  @override
  void dispose() {
    _disposed = true;
    _cancelTimers();
    _speech.stop();
    _player.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // STT – one-shot listen cycle
  // ---------------------------------------------------------------------------

  void _startListening() {
    if (_disposed) return;
    if (_state == VoiceState.off) return;
    if (!_speechAvailable) return;

    _speech.listen(
      onResult: _onSpeechResult,
      listenOptions: SpeechListenOptions(
        listenMode: ListenMode.confirmation, // one-shot
        cancelOnError: true,
      ),
      localeId: 'de_DE',
    );
  }

  void _onSpeechResult(SpeechRecognitionResult result) {
    if (_disposed) return;
    if (!result.finalResult) return;

    final text = result.recognizedWords.trim();
    if (text.isEmpty) {
      _scheduleRestart();
      return;
    }

    _consecutiveErrors = 0;
    onTranscript(text);

    // Auto-restart listening if still in a listening state.
    if (_state == VoiceState.listening || _state == VoiceState.conversation) {
      _scheduleRestart();
    }
  }

  void _onSpeechError(SpeechRecognitionError error) {
    if (_disposed) return;
    // "error_no_match" is common and benign – just means silence.
    if (error.errorMsg == 'error_no_match') {
      _scheduleRestart();
      return;
    }

    _consecutiveErrors++;
    if (_state == VoiceState.listening || _state == VoiceState.conversation) {
      _scheduleRestart();
    }
  }

  void _onSpeechStatus(String status) {
    // When the recognizer stops on its own (e.g. "notListening" / "done"),
    // schedule a restart if we should still be listening.
    if (_disposed) return;
    if (status == 'notListening' || status == 'done') {
      if (_state == VoiceState.listening || _state == VoiceState.conversation) {
        _scheduleRestart();
      }
    }
  }

  /// Schedule a restart of the STT listener with back-off delays.
  /// 500ms normally, 3s on error, 5s after multiple errors.
  void _scheduleRestart() {
    _restartTimer?.cancel();
    if (_disposed) return;
    if (_state == VoiceState.off) return;

    final Duration delay;
    if (_consecutiveErrors >= 3) {
      delay = const Duration(seconds: 5);
    } else if (_consecutiveErrors >= 1) {
      delay = const Duration(seconds: 3);
    } else {
      delay = const Duration(milliseconds: 500);
    }

    _restartTimer = Timer(delay, () {
      if (!_disposed &&
          (_state == VoiceState.listening ||
              _state == VoiceState.conversation)) {
        _startListening();
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Transcript processing (wake word + end phrase detection)
  // ---------------------------------------------------------------------------

  /// Process a transcript from the speech recognizer.
  void onTranscript(String text) {
    _lastTranscript = text;
    final normalized = _normalizeGerman(text.toLowerCase().trim());

    switch (_state) {
      case VoiceState.listening:
        // Check for wake word in any token
        for (final word in normalized.split(RegExp(r'\s+'))) {
          if (_matchesWakeWord(word)) {
            _setState(VoiceState.conversation);
            _resetConversationTimeout();
            return;
          }
        }
      case VoiceState.conversation:
        // Check for end phrase
        for (final phrase in _endPhrases) {
          if (normalized.contains(phrase)) {
            _cancelConversationTimeout();
            _setState(VoiceState.listening);
            return;
          }
        }
        // Otherwise this is a command — process it
        _cancelConversationTimeout();
        _setState(VoiceState.processing);
      case VoiceState.processing:
      case VoiceState.speaking:
      case VoiceState.off:
        break;
    }
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Conversation timeout
  // ---------------------------------------------------------------------------

  void _resetConversationTimeout() {
    _cancelConversationTimeout();
    _conversationTimeoutTimer = Timer(
      const Duration(seconds: _conversationTimeoutSec),
      () {
        if (!_disposed && _state == VoiceState.conversation) {
          _setState(VoiceState.listening);
          _startListening();
        }
      },
    );
  }

  void _cancelConversationTimeout() {
    _conversationTimeoutTimer?.cancel();
    _conversationTimeoutTimer = null;
  }

  // ---------------------------------------------------------------------------
  // TTS playback via just_audio
  // ---------------------------------------------------------------------------

  /// Play WAV audio bytes through just_audio.
  /// Sets state to SPEAKING, then calls [onSpeakingDone] when complete.
  Future<void> playTts(List<int> audioBytes) async {
    if (_disposed) return;
    _setState(VoiceState.speaking);

    try {
      final source = _WavBytesAudioSource(Uint8List.fromList(audioBytes));
      await _player.setAudioSource(source);
      await _player.play();

      // Wait for playback to finish.
      await _player.playerStateStream.firstWhere(
        (s) => s.processingState == ProcessingState.completed,
      );
    } catch (e) {
      debugPrint('TTS playback error: $e');
    }

    onSpeakingDone();
  }

  /// After TTS playback completes, return to conversation mode and resume STT.
  void onSpeakingDone() {
    if (_disposed) return;
    if (_state == VoiceState.speaking) {
      _setState(VoiceState.conversation);
      _resetConversationTimeout();
      _scheduleRestart();
    }
  }

  // ---------------------------------------------------------------------------
  // German phonetic normalization for wake word matching
  // ---------------------------------------------------------------------------

  static String _normalizeGerman(String input) {
    return input
        .replaceAll('tsch', 'j')
        .replaceAll('dsch', 'j')
        .replaceAll('sch', 'j')
        .replaceAll('sh', 'j')
        .replaceAll('ch', 'k')
        .replaceAll('w', 'v')
        .replaceAll('ph', 'f')
        .replaceAll(RegExp(r'[äae]'), 'a')
        .replaceAll(RegExp(r'[öoe]'), 'o')
        .replaceAll(RegExp(r'[üue]'), 'u');
  }

  bool _matchesWakeWord(String word) {
    final normalizedWord = _normalizeGerman(word);
    final normalizedWake = _normalizeGerman(_wakeWord);
    return _levenshtein(normalizedWord, normalizedWake) <= 3;
  }

  /// Levenshtein distance between two strings.
  static int _levenshtein(String a, String b) {
    if (a.isEmpty) return b.length;
    if (b.isEmpty) return a.length;

    final matrix = List.generate(
      a.length + 1,
      (i) => List.generate(b.length + 1, (j) => i == 0 ? j : (j == 0 ? i : 0)),
    );

    for (var i = 1; i <= a.length; i++) {
      for (var j = 1; j <= b.length; j++) {
        final cost = a[i - 1] == b[j - 1] ? 0 : 1;
        matrix[i][j] = [
          matrix[i - 1][j] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j - 1] + cost,
        ].reduce(min);
      }
    }

    return matrix[a.length][b.length];
  }

  void setWakeWord(String word) {
    _wakeWord = word.toLowerCase();
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  void _cancelTimers() {
    _restartTimer?.cancel();
    _restartTimer = null;
    _cancelConversationTimeout();
  }
}

// -----------------------------------------------------------------------------
// StreamAudioSource for playing raw WAV bytes via just_audio
// -----------------------------------------------------------------------------

class _WavBytesAudioSource extends StreamAudioSource {
  _WavBytesAudioSource(this._bytes);

  final Uint8List _bytes;

  @override
  Future<StreamAudioResponse> request([int? start, int? end]) async {
    final effectiveStart = start ?? 0;
    final effectiveEnd = end ?? _bytes.length;
    return StreamAudioResponse(
      sourceLength: _bytes.length,
      contentLength: effectiveEnd - effectiveStart,
      offset: effectiveStart,
      stream: Stream.value(_bytes.sublist(effectiveStart, effectiveEnd)),
      contentType: 'audio/wav',
    );
  }
}
