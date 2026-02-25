"""Tests für die Voice-WebSocket-Bridge."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.voice_bridge import AudioAccumulator, VoiceWebSocketBridge


# ============================================================================
# AudioAccumulator
# ============================================================================


class TestAudioAccumulator:
    def test_empty(self) -> None:
        acc = AudioAccumulator()
        assert acc.is_empty
        assert acc.duration_estimate_seconds == 0.0
        assert acc.get_blob() == b""

    def test_add_chunk(self) -> None:
        acc = AudioAccumulator()
        data = base64.b64encode(b"\x00" * 1200).decode()
        acc.add_chunk(data)
        assert not acc.is_empty
        assert len(acc.get_blob()) == 1200

    def test_multiple_chunks(self) -> None:
        acc = AudioAccumulator()
        for _ in range(5):
            acc.add_chunk(base64.b64encode(b"\x01" * 100).decode())
        assert len(acc.get_blob()) == 500
        assert len(acc.chunks) == 5

    def test_clear(self) -> None:
        acc = AudioAccumulator()
        acc.add_chunk(base64.b64encode(b"\x00" * 100).decode())
        assert not acc.is_empty
        acc.clear()
        assert acc.is_empty

    def test_duration_estimate_webm(self) -> None:
        acc = AudioAccumulator(format="webm")
        acc.add_chunk(base64.b64encode(b"\x00" * 12_000).decode())
        # ~1 Sekunde bei 12kB/s
        assert 0.9 < acc.duration_estimate_seconds < 1.1

    def test_duration_estimate_pcm(self) -> None:
        acc = AudioAccumulator(format="pcm", sample_rate=16000)
        # 16000 samples * 2 bytes = 32000 bytes = 1 Sekunde
        acc.add_chunk(base64.b64encode(b"\x00" * 32_000).decode())
        assert 0.9 < acc.duration_estimate_seconds < 1.1


# ============================================================================
# VoiceWebSocketBridge — Initialisierung
# ============================================================================


class TestBridgeInit:
    def test_defaults(self, tmp_path: Path) -> None:
        bridge = VoiceWebSocketBridge(workspace_dir=tmp_path)
        assert bridge.active_sessions == 0
        assert bridge._stt_engine is None

    @pytest.mark.asyncio
    async def test_initialize_no_whisper(self, tmp_path: Path) -> None:
        bridge = VoiceWebSocketBridge(workspace_dir=tmp_path)
        with patch.dict("sys.modules", {"faster_whisper": None}):
            result = await bridge.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success_mock(self, tmp_path: Path) -> None:
        bridge = VoiceWebSocketBridge(workspace_dir=tmp_path)
        mock_model = MagicMock()

        with patch("jarvis.channels.voice_bridge.WhisperModel", return_value=mock_model, create=True):
            # Manuell initialisieren ohne echten Import
            bridge._stt_engine = mock_model
            assert bridge._stt_engine is not None


# ============================================================================
# VoiceWebSocketBridge — Message-Handling
# ============================================================================


class TestBridgeMessageHandling:
    @pytest.fixture
    def bridge(self, tmp_path: Path) -> VoiceWebSocketBridge:
        return VoiceWebSocketBridge(workspace_dir=tmp_path)

    @pytest.fixture
    def send_fn(self) -> AsyncMock:
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_audio_start(self, bridge: VoiceWebSocketBridge, send_fn: AsyncMock) -> None:
        result = await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_start", "format": "webm", "sample_rate": 16000},
            send_fn,
        )
        assert result is None
        assert bridge.active_sessions == 1
        send_fn.assert_awaited_once()
        call_data = send_fn.call_args[0][0]
        assert call_data["type"] == "voice_status"
        assert call_data["status"] == "listening"

    @pytest.mark.asyncio
    async def test_audio_chunk_without_start(self, bridge: VoiceWebSocketBridge) -> None:
        """Chunks ohne vorheriges audio_start werden ignoriert."""
        result = await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_chunk", "data": base64.b64encode(b"\x00" * 100).decode()},
            AsyncMock(),
        )
        assert result is None
        assert bridge.active_sessions == 0

    @pytest.mark.asyncio
    async def test_audio_chunk_adds_data(self, bridge: VoiceWebSocketBridge, send_fn: AsyncMock) -> None:
        await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_start", "format": "webm"},
            send_fn,
        )

        await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_chunk", "data": base64.b64encode(b"\x00" * 500).decode()},
            send_fn,
        )

        acc = bridge._active_sessions.get("session_1")
        assert acc is not None
        assert not acc.is_empty

    @pytest.mark.asyncio
    async def test_audio_stop_empty(self, bridge: VoiceWebSocketBridge, send_fn: AsyncMock) -> None:
        """audio_stop ohne Start gibt Fehler."""
        result = await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_stop"},
            send_fn,
        )
        assert result is None
        send_fn.assert_awaited_once()
        assert "Keine Audio-Daten" in send_fn.call_args[0][0]["error"]

    @pytest.mark.asyncio
    async def test_audio_stop_too_short(self, bridge: VoiceWebSocketBridge, send_fn: AsyncMock) -> None:
        """Sehr kurze Aufnahmen werden verworfen."""
        await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_start", "format": "webm"},
            send_fn,
        )
        # Nur wenige Bytes — ergibt < 0.3s
        await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_chunk", "data": base64.b64encode(b"\x00" * 10).decode()},
            send_fn,
        )

        send_fn.reset_mock()
        result = await bridge.handle_ws_message(
            "session_1",
            {"type": "audio_stop"},
            send_fn,
        )

        assert result is None
        assert "zu kurz" in send_fn.call_args[0][0]["error"]
        assert bridge.active_sessions == 0

    @pytest.mark.asyncio
    async def test_unknown_message_type(self, bridge: VoiceWebSocketBridge) -> None:
        result = await bridge.handle_ws_message(
            "session_1",
            {"type": "unknown_type"},
            AsyncMock(),
        )
        assert result is None

    def test_cancel_session(self, bridge: VoiceWebSocketBridge) -> None:
        bridge._active_sessions["test"] = AudioAccumulator()
        assert bridge.active_sessions == 1
        bridge.cancel_session("test")
        assert bridge.active_sessions == 0

    def test_cancel_nonexistent_session(self, bridge: VoiceWebSocketBridge) -> None:
        bridge.cancel_session("nonexistent")  # Sollte nicht crashen
        assert bridge.active_sessions == 0


# ============================================================================
# Full-Flow mit Mock-Transkription
# ============================================================================


class TestBridgeFullFlow:
    @pytest.mark.asyncio
    async def test_full_flow_mock(self, tmp_path: Path) -> None:
        """Kompletter Flow: start → chunks → stop → Transkription."""
        bridge = VoiceWebSocketBridge(workspace_dir=tmp_path)
        send_fn = AsyncMock()

        # Mock-Transkription
        async def mock_transcribe(acc):
            return "Hallo Welt"

        bridge._transcribe = mock_transcribe  # type: ignore[assignment]
        bridge._stt_engine = MagicMock()  # Damit _transcribe nicht auf None prüft

        # Start
        await bridge.handle_ws_message(
            "sess1",
            {"type": "audio_start", "format": "webm"},
            send_fn,
        )

        # Genug Daten für > 0.3s (> 3600 bytes bei 12kB/s)
        await bridge.handle_ws_message(
            "sess1",
            {"type": "audio_chunk", "data": base64.b64encode(b"\x00" * 5000).decode()},
            send_fn,
        )

        send_fn.reset_mock()

        # Stop
        result = await bridge.handle_ws_message(
            "sess1",
            {"type": "audio_stop"},
            send_fn,
        )

        assert result == "Hallo Welt"
        assert bridge.active_sessions == 0

        # Prüfe gesendete Nachrichten
        calls = [c[0][0] for c in send_fn.call_args_list]
        types = [c["type"] for c in calls]
        assert "voice_status" in types  # processing
        assert "transcription" in types

        # Transkription enthält Text
        transcription = [c for c in calls if c["type"] == "transcription"][0]
        assert transcription["text"] == "Hallo Welt"
        assert transcription["final"] is True

    @pytest.mark.asyncio
    async def test_transcription_error(self, tmp_path: Path) -> None:
        """Fehler bei Transkription wird an Client gemeldet."""
        bridge = VoiceWebSocketBridge(workspace_dir=tmp_path)
        send_fn = AsyncMock()

        async def mock_fail(acc):
            raise RuntimeError("Whisper crashed")

        bridge._transcribe = mock_fail  # type: ignore[assignment]
        bridge._stt_engine = MagicMock()

        await bridge.handle_ws_message(
            "sess1",
            {"type": "audio_start", "format": "webm"},
            send_fn,
        )
        await bridge.handle_ws_message(
            "sess1",
            {"type": "audio_chunk", "data": base64.b64encode(b"\x00" * 5000).decode()},
            send_fn,
        )

        send_fn.reset_mock()
        result = await bridge.handle_ws_message(
            "sess1",
            {"type": "audio_stop"},
            send_fn,
        )

        assert result is None
        calls = [c[0][0] for c in send_fn.call_args_list]
        error_msgs = [c for c in calls if c["type"] == "voice_error"]
        assert len(error_msgs) == 1
        assert "Whisper crashed" in error_msgs[0]["error"]
