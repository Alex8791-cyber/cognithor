"""Enhanced tests fuer den Voice Channel -- zusaetzliche Coverage.

Deckt: STT load paths, TTS load/synthesize paths, VAD silero,
VoiceChannel listen_once, process_audio_chunk edge cases,
ElevenLabs TTS backend, Piper TTS backend.
"""

from __future__ import annotations

import asyncio
import io
import math
import struct
import tempfile
import wave
from pathlib import Path
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


def _make_pcm_silence(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def _make_pcm_tone(
    duration_ms: int = 100, sample_rate: int = 16000, amplitude: int = 10000
) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = [
        int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


def _make_wav(pcm: bytes, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


# ============================================================================
# STT Engine advanced
# ============================================================================


class TestSTTEngineAdvanced:
    @pytest.mark.asyncio
    async def test_load_whisper_with_torch_cuda(self) -> None:
        """Test whisper load with torch available and cuda."""
        config = VoiceConfig(stt_device="auto")
        engine = STTEngine(config)

        mock_whisper_mod = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_mod, "torch": mock_torch}):
            await engine._load_whisper()
        mock_whisper_mod.WhisperModel.assert_called_once()
        call_kw = mock_whisper_mod.WhisperModel.call_args
        assert call_kw[1]["device"] == "cuda"
        assert call_kw[1]["compute_type"] == "float16"

    @pytest.mark.asyncio
    async def test_load_whisper_with_torch_cpu(self) -> None:
        config = VoiceConfig(stt_device="auto")
        engine = STTEngine(config)

        mock_whisper_mod = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_mod, "torch": mock_torch}):
            await engine._load_whisper()
        call_kw = mock_whisper_mod.WhisperModel.call_args
        assert call_kw[1]["device"] == "cpu"
        assert call_kw[1]["compute_type"] == "int8"

    @pytest.mark.asyncio
    async def test_load_whisper_no_torch(self) -> None:
        config = VoiceConfig(stt_device="auto")
        engine = STTEngine(config)

        mock_whisper_mod = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_mod, "torch": None}):
            await engine._load_whisper()
        call_kw = mock_whisper_mod.WhisperModel.call_args
        assert call_kw[1]["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_load_whisper_explicit_device(self) -> None:
        config = VoiceConfig(stt_device="cpu")
        engine = STTEngine(config)

        mock_whisper_mod = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_mod}):
            await engine._load_whisper()
        call_kw = mock_whisper_mod.WhisperModel.call_args
        assert call_kw[1]["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_load_unsupported_backend(self) -> None:
        config = VoiceConfig(stt_backend=STTBackend.WHISPER_CPP)
        engine = STTEngine(config)
        await engine.load()
        assert engine._model is None

    @pytest.mark.asyncio
    async def test_transcribe_sync(self) -> None:
        config = VoiceConfig()
        engine = STTEngine(config)
        mock_model = MagicMock()
        seg1 = MagicMock()
        seg1.text = " Hello World "
        seg2 = MagicMock()
        seg2.text = " Goodbye "
        mock_info = MagicMock()
        mock_info.language = "de"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)
        engine._model = mock_model

        result = engine._transcribe_sync(str(Path(tempfile.gettempdir()) / "test.wav"))
        assert result == "Hello World Goodbye"

    @pytest.mark.asyncio
    async def test_transcribe_with_loaded_model(self) -> None:
        config = VoiceConfig()
        engine = STTEngine(config)
        mock_model = MagicMock()
        seg = MagicMock()
        seg.text = " Test "
        mock_info = MagicMock()
        mock_info.language = "de"
        mock_info.language_probability = 0.9
        mock_model.transcribe.return_value = ([seg], mock_info)
        engine._model = mock_model

        wav_data = _make_wav(_make_pcm_silence(100))
        text = await engine.transcribe(wav_data)
        assert text == "Test"


# ============================================================================
# TTS Engine advanced
# ============================================================================


class TestTTSEngineAdvanced:
    @pytest.mark.asyncio
    async def test_load_piper_available(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.PIPER)
        engine = TTSEngine(config)

        mock_piper = MagicMock()
        with patch.dict("sys.modules", {"piper": mock_piper}):
            await engine._load_piper()
        assert config.tts_backend == TTSBackend.PIPER

    @pytest.mark.asyncio
    async def test_load_piper_not_available(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.PIPER)
        engine = TTSEngine(config)

        with patch.dict("sys.modules", {"piper": None}):
            await engine._load_piper()
        assert config.tts_backend == TTSBackend.ESPEAK

    @pytest.mark.asyncio
    async def test_load_elevenlabs(self) -> None:
        config = VoiceConfig(
            tts_backend=TTSBackend.ELEVENLABS,
            elevenlabs_api_key="test-key",
            elevenlabs_voice_id="test-voice",
        )
        engine = TTSEngine(config)

        with patch("jarvis.channels.voice.TTSEngine._load_elevenlabs", new_callable=AsyncMock):
            await engine.load()

    @pytest.mark.asyncio
    async def test_load_elevenlabs_no_key(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.ELEVENLABS, elevenlabs_api_key="")
        engine = TTSEngine(config)

        with pytest.raises(ValueError, match="API-Key"):
            await engine._load_elevenlabs()

    @pytest.mark.asyncio
    async def test_load_elevenlabs_no_voice_id(self) -> None:
        config = VoiceConfig(
            tts_backend=TTSBackend.ELEVENLABS,
            elevenlabs_api_key="test-key",
            elevenlabs_voice_id="",
        )
        engine = TTSEngine(config)

        with pytest.raises(ValueError, match="Voice-ID"):
            await engine._load_elevenlabs()

    @pytest.mark.asyncio
    async def test_synthesize_routes_to_piper(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.PIPER)
        engine = TTSEngine(config)

        with patch.object(
            engine, "_synthesize_piper", new_callable=AsyncMock, return_value=b"audio"
        ):
            result = await engine.synthesize("test")
        assert result == b"audio"

    @pytest.mark.asyncio
    async def test_synthesize_routes_to_elevenlabs(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.ELEVENLABS)
        engine = TTSEngine(config)

        with patch.object(
            engine, "_synthesize_elevenlabs", new_callable=AsyncMock, return_value=b"audio"
        ):
            result = await engine.synthesize("test")
        assert result == b"audio"

    @pytest.mark.asyncio
    async def test_synthesize_routes_to_espeak_default(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.ESPEAK)
        engine = TTSEngine(config)

        with patch.object(
            engine, "_synthesize_espeak", new_callable=AsyncMock, return_value=b"audio"
        ):
            result = await engine.synthesize("test")
        assert result == b"audio"

    @pytest.mark.asyncio
    async def test_synthesize_elevenlabs_calls_tts(self) -> None:
        config = VoiceConfig(
            tts_backend=TTSBackend.ELEVENLABS,
            elevenlabs_api_key="key",
            elevenlabs_voice_id="voice",
        )
        engine = TTSEngine(config)
        mock_el = AsyncMock()
        mock_el.synthesize.return_value = b"el-audio"
        engine._elevenlabs_tts = mock_el

        result = await engine._synthesize_elevenlabs("Hello")
        assert result == b"el-audio"

    @pytest.mark.asyncio
    async def test_load_espeak_backend(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.ESPEAK)
        engine = TTSEngine(config)
        await engine.load()  # espeak just logs, no error

    @pytest.mark.asyncio
    async def test_synthesize_piper_success(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.PIPER)
        engine = TTSEngine(config)

        raw_pcm = _make_pcm_silence(50)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = raw_pcm
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result):
            result = await engine._synthesize_piper("Hallo")
        # Should be WAV
        assert result[:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_synthesize_piper_error(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.PIPER)
        engine = TTSEngine(config)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"error"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Piper error"):
                await engine._synthesize_piper("Test")

    @pytest.mark.asyncio
    async def test_synthesize_piper_not_found(self) -> None:
        config = VoiceConfig(tts_backend=TTSBackend.PIPER)
        engine = TTSEngine(config)

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="nicht gefunden"):
                await engine._synthesize_piper("Test")


# ============================================================================
# VAD advanced
# ============================================================================


class TestVADAdvanced:
    @pytest.mark.asyncio
    async def test_load_silero_success(self) -> None:
        vad = VADDetector(VoiceConfig())

        mock_torch = MagicMock()
        mock_model = MagicMock()
        mock_torch.hub.load.return_value = (mock_model, None)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            await vad.load()
        assert vad._use_silero is True
        assert vad._model is mock_model

    @pytest.mark.asyncio
    async def test_load_silero_failure(self) -> None:
        vad = VADDetector(VoiceConfig())

        with patch.dict("sys.modules", {"torch": None}):
            await vad.load()
        assert vad._use_silero is False

    def test_silero_detect(self) -> None:
        vad = VADDetector(VoiceConfig())
        vad._use_silero = True
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.9
        vad._model = mock_model

        mock_torch = MagicMock()
        chunk = _make_pcm_tone(50, amplitude=5000)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = vad._silero_detect(chunk)
        assert result is True

    def test_silero_detect_below_threshold(self) -> None:
        vad = VADDetector(VoiceConfig(vad_threshold=0.5))
        vad._use_silero = True
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.1
        vad._model = mock_model

        mock_torch = MagicMock()
        chunk = _make_pcm_silence(50)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = vad._silero_detect(chunk)
        assert result is False

    def test_silero_detect_exception_falls_back(self) -> None:
        vad = VADDetector(VoiceConfig())
        vad._use_silero = True
        vad._model = MagicMock(side_effect=Exception("broken"))

        mock_torch = MagicMock()
        mock_torch.FloatTensor.side_effect = Exception("broken")
        chunk = _make_pcm_silence(50)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = vad._silero_detect(chunk)
        # Falls back to energy detect on silence
        assert result is False

    def test_is_speech_routes_to_silero(self) -> None:
        vad = VADDetector(VoiceConfig())
        vad._use_silero = True
        vad._model = MagicMock()

        with patch.object(vad, "_silero_detect", return_value=True) as mock:
            result = vad.is_speech(b"\x00\x00")
        mock.assert_called_once()
        assert result is True


# ============================================================================
# VoiceChannel advanced
# ============================================================================


class TestVoiceChannelAdvanced:
    @pytest.mark.asyncio
    async def test_start_with_vad_enabled(self) -> None:
        config = VoiceConfig(vad_enabled=True)
        ch = VoiceChannel(config=config)
        handler = AsyncMock()

        with (
            patch.object(ch._stt, "load", new_callable=AsyncMock),
            patch.object(ch._tts, "load", new_callable=AsyncMock),
            patch.object(ch._vad, "load", new_callable=AsyncMock) as vad_load,
        ):
            await ch.start(handler)
        vad_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_vad_disabled(self) -> None:
        config = VoiceConfig(vad_enabled=False)
        ch = VoiceChannel(config=config)
        handler = AsyncMock()

        with (
            patch.object(ch._stt, "load", new_callable=AsyncMock),
            patch.object(ch._tts, "load", new_callable=AsyncMock),
            patch.object(ch._vad, "load", new_callable=AsyncMock) as vad_load,
        ):
            await ch.start(handler)
        vad_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_listen_once_with_audio(self) -> None:
        config = VoiceConfig(vad_enabled=False)
        ch = VoiceChannel(config=config)
        ch._stt = AsyncMock()
        ch._stt.transcribe.return_value = "Hello World"

        # listen_once clears the buffer first, then loops with asyncio.sleep(0.1)
        # We patch sleep to inject audio into the buffer during the loop.
        call_count = 0
        _original_sleep = asyncio.sleep

        async def _fake_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Inject enough audio (>0.5s) on first sleep
                ch._audio_buffer.add_chunk(_make_pcm_tone(700))
            await _original_sleep(0)  # yield control

        with patch("jarvis.channels.voice.asyncio.sleep", side_effect=_fake_sleep):
            result = await ch.listen_once(timeout=5.0)
        assert result == "Hello World"
        assert not ch.is_listening

    @pytest.mark.asyncio
    async def test_listen_once_timeout(self) -> None:
        config = VoiceConfig(vad_enabled=False)
        ch = VoiceChannel(config=config)
        # Empty buffer => timeout
        result = await ch.listen_once(timeout=0.3)
        assert result is None
        assert not ch.is_listening

    @pytest.mark.asyncio
    async def test_process_audio_chunk_speech_then_silence(self) -> None:
        config = VoiceConfig(silence_duration_ms=0)
        ch = VoiceChannel(config=config)
        ch._vad = MagicMock()
        ch._stt = AsyncMock()
        ch._stt.transcribe.return_value = "Text"

        # Add enough speech data first
        ch._vad.is_speech.return_value = True
        for _ in range(10):
            await ch.process_audio_chunk(_make_pcm_tone(100))

        # Now silence
        import time

        ch._vad.is_speech.return_value = False
        ch._audio_buffer._silence_start = time.monotonic() - 1.0
        result = await ch.process_audio_chunk(_make_pcm_silence(100))
        # Buffer had enough data (>0.3s), so should transcribe
        assert result == "Text"

    @pytest.mark.asyncio
    async def test_process_audio_chunk_short_speech_cleared(self) -> None:
        """Short speech (<0.3s) is cleared without transcription."""
        config = VoiceConfig(silence_duration_ms=0)
        ch = VoiceChannel(config=config)
        ch._vad = MagicMock()
        ch._stt = AsyncMock()

        # Add very little speech data (< 0.3s)
        ch._vad.is_speech.return_value = True
        await ch.process_audio_chunk(_make_pcm_tone(50))  # 0.05s

        # Silence starts
        ch._vad.is_speech.return_value = False
        import time

        ch._audio_buffer._silence_start = time.monotonic() - 1.0
        result = await ch.process_audio_chunk(_make_pcm_silence(50))
        assert result is None  # Too short, just cleared

    @pytest.mark.asyncio
    async def test_process_audio_chunk_empty_transcription(self) -> None:
        config = VoiceConfig(silence_duration_ms=0)
        ch = VoiceChannel(config=config)
        ch._vad = MagicMock()
        ch._stt = AsyncMock()
        ch._stt.transcribe.return_value = "   "  # empty after strip

        # Add enough speech
        ch._vad.is_speech.return_value = True
        for _ in range(10):
            await ch.process_audio_chunk(_make_pcm_tone(100))

        ch._vad.is_speech.return_value = False
        import time

        ch._audio_buffer._silence_start = time.monotonic() - 1.0
        result = await ch.process_audio_chunk(_make_pcm_silence(100))
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_voice_message_processing_flag(self) -> None:
        config = VoiceConfig()
        ch = VoiceChannel(config=config)
        handler = AsyncMock()
        handler.return_value = OutgoingMessage(text="Answer", session_id="s", channel="voice")
        ch._handler = handler
        ch._stt = AsyncMock()
        ch._stt.transcribe.return_value = "Question"
        ch._tts = AsyncMock()
        ch._tts.synthesize.return_value = b""  # no audio

        assert not ch.is_processing
        with patch.object(ch, "_play_audio", new_callable=AsyncMock):
            result = await ch.handle_voice_message(b"data")
        assert result == "Answer"
        assert not ch.is_processing  # reset after processing

    @pytest.mark.asyncio
    async def test_send_no_audio_from_tts(self) -> None:
        config = VoiceConfig()
        ch = VoiceChannel(config=config)
        ch._tts = AsyncMock()
        ch._tts.synthesize.return_value = b""  # empty

        with patch.object(ch, "_play_audio", new_callable=AsyncMock) as play:
            msg = OutgoingMessage(text="test", session_id="s", channel="voice")
            await ch.send(msg)
        play.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_ok_response(self) -> None:
        config = VoiceConfig()
        ch = VoiceChannel(config=config)
        ch._tts = AsyncMock()
        ch._tts.synthesize.return_value = b"audio"

        with (
            patch.object(ch, "_play_audio", new_callable=AsyncMock),
            patch.object(ch, "listen_once", new_callable=AsyncMock, return_value="ok klar"),
        ):
            action = PlannedAction(tool="test", params={})
            result = await ch.request_approval("s1", action, "reason")
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_no_audio(self) -> None:
        config = VoiceConfig()
        ch = VoiceChannel(config=config)
        ch._tts = AsyncMock()
        ch._tts.synthesize.return_value = b""  # no audio, still works

        with patch.object(ch, "listen_once", new_callable=AsyncMock, return_value="nein"):
            action = PlannedAction(tool="test", params={})
            result = await ch.request_approval("s1", action, "reason")
        assert result is False

    def test_default_config(self) -> None:
        ch = VoiceChannel()
        assert ch._config.stt_backend == STTBackend.WHISPER
        assert ch._config.tts_backend == TTSBackend.PIPER

    @pytest.mark.asyncio
    async def test_process_audio_chunk_first_silence_starts_timer(self) -> None:
        config = VoiceConfig(silence_duration_ms=5000)
        ch = VoiceChannel(config=config)
        ch._vad = MagicMock()
        ch._vad.is_speech.return_value = False

        # Mark as speaking
        ch._audio_buffer._is_speaking = True
        ch._audio_buffer._silence_start = 0.0

        result = await ch.process_audio_chunk(_make_pcm_silence(50))
        assert result is None
        assert ch._audio_buffer._silence_start > 0  # Timer started
