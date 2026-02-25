"""Tests fuer den ElevenLabs TTS Backend.

Testet ElevenLabsTTS Klasse mit gemockten httpx-Aufrufen.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.tts_elevenlabs import ElevenLabsConfig, ElevenLabsTTS


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tts() -> ElevenLabsTTS:
    return ElevenLabsTTS(
        api_key="test-api-key-123",
        voice_id="test-voice-id-456",
        model="eleven_multilingual_v2",
    )


@pytest.fixture
def elevenlabs_config() -> ElevenLabsConfig:
    return ElevenLabsConfig(
        api_key="cfg-key",
        voice_id="cfg-voice",
    )


# ============================================================================
# Tests
# ============================================================================


class TestElevenLabsTTSInit:
    """Tests fuer __init__ und Konfiguration."""

    def test_init_stores_config(self, tts: ElevenLabsTTS) -> None:
        """Verify init parameters are stored correctly."""
        assert tts.api_key == "test-api-key-123"
        assert tts.voice_id == "test-voice-id-456"
        assert tts.model == "eleven_multilingual_v2"

    def test_init_default_model(self) -> None:
        """Verify default model is eleven_multilingual_v2."""
        tts = ElevenLabsTTS(api_key="k", voice_id="v")
        assert tts.model == "eleven_multilingual_v2"

    def test_init_custom_model(self) -> None:
        """Verify custom model is stored."""
        tts = ElevenLabsTTS(api_key="k", voice_id="v", model="eleven_turbo_v2")
        assert tts.model == "eleven_turbo_v2"

    def test_build_url(self, tts: ElevenLabsTTS) -> None:
        """Verify URL construction."""
        url = tts._build_url()
        assert "text-to-speech/test-voice-id-456" in url
        assert not url.endswith("/stream")

    def test_build_url_stream(self, tts: ElevenLabsTTS) -> None:
        """Verify streaming URL construction."""
        url = tts._build_url(stream=True)
        assert url.endswith("/stream")

    def test_build_headers(self, tts: ElevenLabsTTS) -> None:
        """Verify headers include API key."""
        headers = tts._build_headers()
        assert headers["xi-api-key"] == "test-api-key-123"
        assert "application/json" in headers["Content-Type"]

    def test_build_payload(self, tts: ElevenLabsTTS) -> None:
        """Verify payload structure."""
        payload = tts._build_payload("Hello")
        assert payload["text"] == "Hello"
        assert payload["model_id"] == "eleven_multilingual_v2"
        assert "voice_settings" in payload
        assert payload["voice_settings"]["stability"] == 0.5


class TestElevenLabsTTSSynthesize:
    """Tests fuer die synthesize-Methode."""

    @pytest.mark.asyncio
    async def test_synthesize_success(self, tts: ElevenLabsTTS) -> None:
        """Mock httpx response, verify audio bytes returned."""
        fake_audio = b"FAKE_AUDIO_DATA_BYTES"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("jarvis.channels.tts_elevenlabs.httpx", create=True) as mock_httpx:
            mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
            mock_httpx.HTTPStatusError = Exception
            mock_httpx.RequestError = Exception

            # Patch the import inside synthesize
            import sys
            mock_httpx_module = MagicMock()
            mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)
            mock_httpx_module.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
            mock_httpx_module.RequestError = type("RequestError", (Exception,), {})

            with patch.dict(sys.modules, {"httpx": mock_httpx_module}):
                result = await tts.synthesize("Hallo Welt")

            assert result == fake_audio
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_api_error(self, tts: ElevenLabsTTS) -> None:
        """Mock 400 response, verify empty bytes returned."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request: invalid voice_id"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import sys
        mock_httpx_module = MagicMock()
        mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)
        mock_httpx_module.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        mock_httpx_module.RequestError = type("RequestError", (Exception,), {})

        with patch.dict(sys.modules, {"httpx": mock_httpx_module}):
            result = await tts.synthesize("Test text")

        assert result == b""
        assert len(result) == 0


class TestElevenLabsTTSStream:
    """Tests fuer die stream-Methode."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, tts: ElevenLabsTTS) -> None:
        """Mock streaming response, verify chunks are yielded."""
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        # Create an async iterator for the chunks
        async def mock_aiter_bytes(chunk_size=4096):
            for c in chunks:
                yield c

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import sys
        mock_httpx_module = MagicMock()
        mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)
        mock_httpx_module.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        mock_httpx_module.RequestError = type("RequestError", (Exception,), {})

        received = []
        with patch.dict(sys.modules, {"httpx": mock_httpx_module}):
            async for chunk in tts.stream("Streaming test"):
                received.append(chunk)

        assert len(received) == 3
        assert received[0] == b"chunk1"
        assert received[1] == b"chunk2"
        assert received[2] == b"chunk3"

    @pytest.mark.asyncio
    async def test_stream_api_error_yields_nothing(self, tts: ElevenLabsTTS) -> None:
        """Mock error streaming response, verify no chunks yielded."""

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import sys
        mock_httpx_module = MagicMock()
        mock_httpx_module.AsyncClient = MagicMock(return_value=mock_client)
        mock_httpx_module.RequestError = type("RequestError", (Exception,), {})

        received = []
        with patch.dict(sys.modules, {"httpx": mock_httpx_module}):
            async for chunk in tts.stream("Error test"):
                received.append(chunk)

        assert len(received) == 0


class TestElevenLabsConfig:
    """Tests fuer die ElevenLabsConfig Dataclass."""

    def test_default_values(self) -> None:
        """Verify default config values."""
        cfg = ElevenLabsConfig()
        assert cfg.api_key == ""
        assert cfg.voice_id == ""
        assert cfg.model == "eleven_multilingual_v2"
        assert cfg.output_format == "mp3_44100_128"
        assert cfg.stability == 0.5
        assert cfg.similarity_boost == 0.75

    def test_voice_config_default_voice_id(self) -> None:
        """Verify that VoiceConfig (config.py + voice.py) has correct default voice_id."""
        from jarvis.config import VoiceConfig as ConfigVoiceConfig
        from jarvis.channels.voice import VoiceConfig as ChannelVoiceConfig

        cfg_voice = ConfigVoiceConfig()
        assert cfg_voice.elevenlabs_voice_id == "hJAaR77ekN23CNyp0byH"

        ch_voice = ChannelVoiceConfig()
        assert ch_voice.elevenlabs_voice_id == "hJAaR77ekN23CNyp0byH"

    def test_custom_values(self, elevenlabs_config: ElevenLabsConfig) -> None:
        """Verify custom config values from fixture."""
        assert elevenlabs_config.api_key == "cfg-key"
        assert elevenlabs_config.voice_id == "cfg-voice"
