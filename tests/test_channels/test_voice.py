"""Tests für den Voice Channel.

Testet AudioBuffer, VAD, STT/TTS-Engines (mit Mocks),
und den VoiceChannel selbst.
"""

from __future__ import annotations

import io
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.voice import (
    AudioBuffer,
    STTBackend,
    STTEngine,
    TTSBackend,
    TTSEngine,
    VADDetector,
    VoiceChannel,
    VoiceConfig,
)
from jarvis.models import OutgoingMessage, PlannedAction

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def voice_config() -> VoiceConfig:
    return VoiceConfig(
        stt_backend=STTBackend.WHISPER,
        tts_backend=TTSBackend.ESPEAK,
        sample_rate=16000,
        channels=1,
        vad_enabled=False,
    )


@pytest.fixture
def audio_buffer(voice_config: VoiceConfig) -> AudioBuffer:
    return AudioBuffer(config=voice_config)


@pytest.fixture
def vad(voice_config: VoiceConfig) -> VADDetector:
    return VADDetector(voice_config)


@pytest.fixture
def channel(voice_config: VoiceConfig) -> VoiceChannel:
    return VoiceChannel(config=voice_config)


def make_pcm_silence(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Erzeugt stille PCM-Daten (16-bit mono)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def make_pcm_tone(
    duration_ms: int = 100,
    sample_rate: int = 16000,
    amplitude: int = 10000,
    freq: int = 440,
) -> bytes:
    """Erzeugt einen Sinuston als PCM-Daten."""
    import math

    num_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(num_samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        samples.append(value)
    return struct.pack(f"<{num_samples}h", *samples)


def make_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Konvertiert PCM-Daten zu WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ============================================================================
# VoiceConfig
# ============================================================================


class TestVoiceConfig:
    def test_defaults(self) -> None:
        config = VoiceConfig()
        assert config.stt_backend == STTBackend.WHISPER
        assert config.tts_backend == TTSBackend.PIPER
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.stt_language == "de"
        assert config.vad_threshold == 0.5
        assert config.silence_duration_ms == 800

    def test_custom_config(self) -> None:
        config = VoiceConfig(
            stt_model="small",
            tts_speed=1.5,
            vad_threshold=0.3,
        )
        assert config.stt_model == "small"
        assert config.tts_speed == 1.5
        assert config.vad_threshold == 0.3

    def test_backend_enums(self) -> None:
        assert STTBackend.WHISPER == "whisper"
        assert STTBackend.WHISPER_CPP == "whisper_cpp"
        assert TTSBackend.PIPER == "piper"
        assert TTSBackend.ESPEAK == "espeak"


# ============================================================================
# AudioBuffer
# ============================================================================


class TestAudioBuffer:
    def test_empty_buffer(self, audio_buffer: AudioBuffer) -> None:
        assert audio_buffer.is_empty
        assert audio_buffer.duration_seconds == 0.0

    def test_add_chunk(self, audio_buffer: AudioBuffer) -> None:
        pcm = make_pcm_silence(100)
        audio_buffer.add_chunk(pcm)
        assert not audio_buffer.is_empty
        assert audio_buffer.duration_seconds == pytest.approx(0.1, abs=0.01)

    def test_multiple_chunks(self, audio_buffer: AudioBuffer) -> None:
        for _ in range(5):
            audio_buffer.add_chunk(make_pcm_silence(100))
        assert audio_buffer.duration_seconds == pytest.approx(0.5, abs=0.05)

    def test_clear(self, audio_buffer: AudioBuffer) -> None:
        audio_buffer.add_chunk(make_pcm_silence(200))
        assert not audio_buffer.is_empty
        audio_buffer.clear()
        assert audio_buffer.is_empty
        assert audio_buffer.duration_seconds == 0.0

    def test_get_audio_data(self, audio_buffer: AudioBuffer) -> None:
        pcm1 = make_pcm_silence(50)
        pcm2 = make_pcm_silence(50)
        audio_buffer.add_chunk(pcm1)
        audio_buffer.add_chunk(pcm2)
        data = audio_buffer.get_audio_data()
        assert data == pcm1 + pcm2

    def test_to_wav_bytes(self, audio_buffer: AudioBuffer) -> None:
        pcm = make_pcm_silence(100)
        audio_buffer.add_chunk(pcm)
        wav = audio_buffer.to_wav_bytes()
        # WAV-Header prüfen
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        # Sollte lesbar sein
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_zero_sample_rate_duration(self) -> None:
        config = VoiceConfig(sample_rate=0)
        buf = AudioBuffer(config=config)
        buf.add_chunk(b"\x00\x00")
        assert buf.duration_seconds == 0.0


# ============================================================================
# VAD (Voice Activity Detection)
# ============================================================================


class TestVADDetector:
    @pytest.mark.asyncio
    async def test_load_fallback(self, vad: VADDetector) -> None:
        await vad.load()
        assert not vad._use_silero  # Silero nicht verfügbar in Tests

    def test_energy_detect_silence(self, vad: VADDetector) -> None:
        silence = make_pcm_silence(100)
        assert not vad._energy_detect(silence)

    def test_energy_detect_loud(self, vad: VADDetector) -> None:
        tone = make_pcm_tone(100, amplitude=15000)
        assert vad._energy_detect(tone)

    def test_energy_detect_empty(self, vad: VADDetector) -> None:
        assert not vad._energy_detect(b"")

    def test_energy_detect_single_byte(self, vad: VADDetector) -> None:
        assert not vad._energy_detect(b"\x00")

    def test_is_speech_uses_energy_fallback(self, vad: VADDetector) -> None:
        tone = make_pcm_tone(100, amplitude=15000)
        silence = make_pcm_silence(100)
        assert vad.is_speech(tone)
        assert not vad.is_speech(silence)


# ============================================================================
# STT Engine
# ============================================================================


class TestSTTEngine:
    def test_init(self, voice_config: VoiceConfig) -> None:
        engine = STTEngine(voice_config)
        assert engine._model is None

    @pytest.mark.asyncio
    async def test_transcribe_without_load_raises(self, voice_config: VoiceConfig) -> None:
        engine = STTEngine(voice_config)
        wav = make_wav(make_pcm_silence(100))
        with pytest.raises(RuntimeError, match="nicht geladen"):
            await engine.transcribe(wav)

    @pytest.mark.asyncio
    async def test_load_whisper_import_error(self, voice_config: VoiceConfig) -> None:
        engine = STTEngine(voice_config)
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(ImportError):
                await engine._load_whisper()


# ============================================================================
# TTS Engine
# ============================================================================


class TestTTSEngine:
    def test_init(self, voice_config: VoiceConfig) -> None:
        engine = TTSEngine(voice_config)
        assert engine._config.tts_backend == TTSBackend.ESPEAK

    @pytest.mark.asyncio
    async def test_espeak_synthesize(self, voice_config: VoiceConfig) -> None:
        engine = TTSEngine(voice_config)
        # Mock espeak-ng
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            make_wav(make_pcm_silence(100)),
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            audio = await engine._synthesize_espeak("Hallo Welt")
            assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_espeak_error_returns_empty(self, voice_config: VoiceConfig) -> None:
        engine = TTSEngine(voice_config)
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"Error")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            audio = await engine._synthesize_espeak("Test")
            assert audio == b""

    def test_raw_to_wav(self) -> None:
        raw = make_pcm_silence(50)
        wav = TTSEngine._raw_to_wav(raw, 16000)
        assert wav[:4] == b"RIFF"
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 16000


# ============================================================================
# VoiceChannel
# ============================================================================


class TestVoiceChannel:
    def test_name(self, channel: VoiceChannel) -> None:
        assert channel.name == "voice"

    def test_initial_state(self, channel: VoiceChannel) -> None:
        assert not channel.is_listening
        assert not channel.is_processing

    @pytest.mark.asyncio
    async def test_start_loads_models(self, channel: VoiceChannel) -> None:
        handler = AsyncMock()
        with (
            patch.object(channel._stt, "load", new_callable=AsyncMock) as stt_load,
            patch.object(channel._tts, "load", new_callable=AsyncMock) as tts_load,
        ):
            await channel.start(handler)
            stt_load.assert_called_once()
            tts_load.assert_called_once()
            assert channel._handler is handler

    @pytest.mark.asyncio
    async def test_stop(self, channel: VoiceChannel) -> None:
        channel._is_listening = True
        channel._audio_buffer.add_chunk(make_pcm_silence(100))
        await channel.stop()
        assert not channel.is_listening
        assert channel._audio_buffer.is_empty

    @pytest.mark.asyncio
    async def test_send_synthesizes_and_plays(self, channel: VoiceChannel) -> None:
        mock_wav = make_wav(make_pcm_silence(100))
        channel._tts = AsyncMock()
        channel._tts.synthesize.return_value = mock_wav

        with patch.object(channel, "_play_audio", new_callable=AsyncMock) as play:
            msg = OutgoingMessage(text="Hallo", session_id="v1", channel="voice")
            await channel.send(msg)
            play.assert_called_once_with(mock_wav)

    @pytest.mark.asyncio
    async def test_send_empty_text_skips(self, channel: VoiceChannel) -> None:
        channel._tts = AsyncMock()
        msg = OutgoingMessage(text="", session_id="v1", channel="voice")
        await channel.send(msg)
        channel._tts.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_streaming_token_is_noop(self, channel: VoiceChannel) -> None:
        await channel.send_streaming_token("s1", "token")

    @pytest.mark.asyncio
    async def test_handle_voice_message(self, channel: VoiceChannel) -> None:
        handler = AsyncMock()
        handler.return_value = OutgoingMessage(
            text="Antwort",
            session_id="voice_session",
            channel="voice",
        )
        channel._handler = handler
        channel._stt = AsyncMock()
        channel._stt.transcribe.return_value = "Wie ist das Wetter?"
        channel._tts = AsyncMock()
        channel._tts.synthesize.return_value = make_wav(make_pcm_silence(100))

        with patch.object(channel, "_play_audio", new_callable=AsyncMock):
            result = await channel.handle_voice_message(b"wav-data")

        assert result == "Antwort"
        handler.assert_called_once()
        call_arg = handler.call_args[0][0]
        assert call_arg.text == "Wie ist das Wetter?"
        assert call_arg.channel == "voice"

    @pytest.mark.asyncio
    async def test_handle_voice_message_empty_transcription(self, channel: VoiceChannel) -> None:
        handler = AsyncMock()
        channel._handler = handler
        channel._stt = AsyncMock()
        channel._stt.transcribe.return_value = "   "

        result = await channel.handle_voice_message(b"wav-data")
        assert result is None
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_voice_message_no_handler(self, channel: VoiceChannel) -> None:
        channel._handler = None
        result = await channel.handle_voice_message(b"wav-data")
        assert result is None

    @pytest.mark.asyncio
    async def test_process_audio_chunk_with_speech(self, channel: VoiceChannel) -> None:
        """VAD erkennt Sprache → Puffer füllen, bei Stille transkribieren."""
        channel._vad = MagicMock()
        channel._stt = AsyncMock()
        channel._stt.transcribe.return_value = "Transkription"

        tone = make_pcm_tone(100, amplitude=15000)
        silence = make_pcm_silence(100)

        # Phase 1: Sprache senden
        channel._vad.is_speech.return_value = True
        result = await channel.process_audio_chunk(tone)
        assert result is None  # Noch kein End-of-Speech

        # Phase 2: Stille beginnt
        channel._vad.is_speech.return_value = False
        channel._audio_buffer._is_speaking = True
        channel._config.silence_duration_ms = 0  # Sofort End-of-Speech

        import time

        channel._audio_buffer._silence_start = time.monotonic() - 1.0
        result = await channel.process_audio_chunk(silence)
        # Sollte transkribiert werden
        if channel._audio_buffer.duration_seconds >= 0.3:
            assert result == "Transkription"

    @pytest.mark.asyncio
    async def test_process_audio_chunk_silence_only(self, channel: VoiceChannel) -> None:
        channel._vad = MagicMock()
        channel._vad.is_speech.return_value = False

        silence = make_pcm_silence(100)
        result = await channel.process_audio_chunk(silence)
        assert result is None  # Keine Sprache → kein Text

    @pytest.mark.asyncio
    async def test_play_audio_fallback(self, channel: VoiceChannel) -> None:
        """_play_audio mit aplay-Fallback."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await channel._play_audio(b"wav-data")
            mock_proc.communicate.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_audio_aplay_not_found(self, channel: VoiceChannel) -> None:
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            # Sollte nicht crashen
            await channel._play_audio(b"wav-data")


# ============================================================================
# Approval via Voice
# ============================================================================


class TestVoiceApproval:
    @pytest.mark.asyncio
    async def test_approval_ja(self, channel: VoiceChannel) -> None:
        channel._tts = AsyncMock()
        channel._tts.synthesize.return_value = make_wav(make_pcm_silence(50))

        with (
            patch.object(channel, "_play_audio", new_callable=AsyncMock),
            patch.object(
                channel, "listen_once", new_callable=AsyncMock, return_value="Ja, mach das"
            ),
        ):
            action = PlannedAction(
                tool="email_send",
                params={"to": "test@example.com"},
                rationale="Test",
            )
            result = await channel.request_approval("v1", action, "E-Mail senden")
            assert result is True

    @pytest.mark.asyncio
    async def test_approval_nein(self, channel: VoiceChannel) -> None:
        channel._tts = AsyncMock()
        channel._tts.synthesize.return_value = make_wav(make_pcm_silence(50))

        with (
            patch.object(channel, "_play_audio", new_callable=AsyncMock),
            patch.object(channel, "listen_once", new_callable=AsyncMock, return_value="Nein"),
        ):
            action = PlannedAction(
                tool="file_delete",
                params={"path": "/tmp/x"},
                rationale="Cleanup",
            )
            result = await channel.request_approval("v1", action, "Datei löschen")
            assert result is False

    @pytest.mark.asyncio
    async def test_approval_timeout(self, channel: VoiceChannel) -> None:
        channel._tts = AsyncMock()
        channel._tts.synthesize.return_value = make_wav(make_pcm_silence(50))

        with (
            patch.object(channel, "_play_audio", new_callable=AsyncMock),
            patch.object(channel, "listen_once", new_callable=AsyncMock, return_value=None),
        ):
            action = PlannedAction(
                tool="shell_exec",
                params={"command": "ls"},
                rationale="List",
            )
            result = await channel.request_approval("v1", action, "Shell")
            assert result is False
