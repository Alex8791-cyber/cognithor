"""Tests für die Voice-WebSocket-Bridge."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.voice_ws_bridge import VoiceWebSocketBridge


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    d = tmp_path / "voice_ws"
    d.mkdir()
    return d


@pytest.fixture
def bridge(workspace: Path) -> VoiceWebSocketBridge:
    return VoiceWebSocketBridge(workspace_dir=workspace)


class TestVoiceWebSocketBridgeInit:
    def test_default_init(self, bridge: VoiceWebSocketBridge) -> None:
        assert bridge._media is None  # Lazy
        assert bridge._workspace.exists()

    def test_lazy_media(self, bridge: VoiceWebSocketBridge) -> None:
        media = bridge._get_media()
        assert media is not None
        # Zweiter Aufruf gibt gleiche Instanz zurück
        assert bridge._get_media() is media


class TestTranscribeVoiceMessage:
    @pytest.mark.asyncio
    async def test_empty_audio(self, bridge: VoiceWebSocketBridge) -> None:
        """Leere Base64-Daten führen zu None."""
        result = await bridge.transcribe_voice_message("")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_base64(self, bridge: VoiceWebSocketBridge) -> None:
        """Ungültige Base64-Daten führen zu None."""
        result = await bridge.transcribe_voice_message("not-valid-base64!!!")
        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_with_mock(self, bridge: VoiceWebSocketBridge) -> None:
        """Erfolgreiche Transkription mit gemockter MediaPipeline."""
        from jarvis.mcp.media import MediaResult

        audio_bytes = b"\x00" * 100
        audio_b64 = base64.b64encode(audio_bytes).decode()

        mock_media = MagicMock()
        mock_media.transcribe_audio = AsyncMock(
            return_value=MediaResult(success=True, text="Hallo Welt")
        )
        bridge._media = mock_media

        # ffmpeg-Konvertierung mocken (not found → Fallback auf Original)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await bridge.transcribe_voice_message(audio_b64, "audio/webm")

        assert result == "Hallo Welt"

    @pytest.mark.asyncio
    async def test_transcribe_empty_result(self, bridge: VoiceWebSocketBridge) -> None:
        """Leere Transkription gibt None zurück."""
        from jarvis.mcp.media import MediaResult

        audio_b64 = base64.b64encode(b"\x00" * 50).decode()

        mock_media = MagicMock()
        mock_media.transcribe_audio = AsyncMock(
            return_value=MediaResult(success=True, text="   ")
        )
        bridge._media = mock_media

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await bridge.transcribe_voice_message(audio_b64)

        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_failure(self, bridge: VoiceWebSocketBridge) -> None:
        """Fehlgeschlagene Transkription gibt None zurück."""
        from jarvis.mcp.media import MediaResult

        audio_b64 = base64.b64encode(b"\xff" * 50).decode()

        mock_media = MagicMock()
        mock_media.transcribe_audio = AsyncMock(
            return_value=MediaResult(success=False, error="Model not found")
        )
        bridge._media = mock_media

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await bridge.transcribe_voice_message(audio_b64)

        assert result is None


class TestSynthesizeResponse:
    @pytest.mark.asyncio
    async def test_empty_text(self, bridge: VoiceWebSocketBridge) -> None:
        result = await bridge.synthesize_response("")
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_text(self, bridge: VoiceWebSocketBridge) -> None:
        result = await bridge.synthesize_response("   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_success(self, bridge: VoiceWebSocketBridge, workspace: Path) -> None:
        from jarvis.mcp.media import MediaResult

        # Fake WAV-Datei erstellen die nach TTS "da" sein soll
        wav_path = workspace / "voice_response.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_media = MagicMock()
        mock_media.text_to_speech = AsyncMock(
            return_value=MediaResult(success=True, text=f"Audio erzeugt: {wav_path}")
        )
        bridge._media = mock_media

        result = await bridge.synthesize_response("Hallo")
        assert result is not None
        # Prüfe dass es gültiges Base64 ist
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_synthesis_failure(self, bridge: VoiceWebSocketBridge) -> None:
        from jarvis.mcp.media import MediaResult

        mock_media = MagicMock()
        mock_media.text_to_speech = AsyncMock(
            return_value=MediaResult(success=False, error="No TTS backend")
        )
        bridge._media = mock_media

        result = await bridge.synthesize_response("Test")
        assert result is None


class TestConvertToWav:
    @pytest.mark.asyncio
    async def test_no_ffmpeg(self, bridge: VoiceWebSocketBridge, workspace: Path) -> None:
        f = workspace / "test.webm"
        f.write_bytes(b"\x00" * 10)

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await bridge._convert_to_wav(f)

        assert result is None

    @pytest.mark.asyncio
    async def test_ffmpeg_failure(self, bridge: VoiceWebSocketBridge, workspace: Path) -> None:
        f = workspace / "bad.webm"
        f.write_bytes(b"\x00" * 10)

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: invalid data"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await bridge._convert_to_wav(f)

        assert result is None


class TestHandleWSVoiceMessage:
    @pytest.mark.asyncio
    async def test_no_audio_data(self, bridge: VoiceWebSocketBridge) -> None:
        result = await bridge.handle_ws_voice_message({"type": "voice_message"})
        assert result["type"] == "error"

    @pytest.mark.asyncio
    async def test_successful_transcription(self, bridge: VoiceWebSocketBridge) -> None:
        from jarvis.mcp.media import MediaResult

        audio_b64 = base64.b64encode(b"\x00" * 50).decode()

        mock_media = MagicMock()
        mock_media.transcribe_audio = AsyncMock(
            return_value=MediaResult(success=True, text="Test Nachricht")
        )
        bridge._media = mock_media

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await bridge.handle_ws_voice_message({
                "type": "voice_message",
                "audio_base64": audio_b64,
                "audio_type": "audio/webm",
            })

        assert result["type"] == "voice_transcription"
        assert result["text"] == "Test Nachricht"

    @pytest.mark.asyncio
    async def test_no_speech_detected(self, bridge: VoiceWebSocketBridge) -> None:
        from jarvis.mcp.media import MediaResult

        audio_b64 = base64.b64encode(b"\x00" * 50).decode()

        mock_media = MagicMock()
        mock_media.transcribe_audio = AsyncMock(
            return_value=MediaResult(success=True, text="")
        )
        bridge._media = mock_media

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await bridge.handle_ws_voice_message({
                "type": "voice_message",
                "audio_base64": audio_b64,
            })

        assert result["type"] == "voice_transcription"
        assert result["text"] == ""
        assert "error" in result
