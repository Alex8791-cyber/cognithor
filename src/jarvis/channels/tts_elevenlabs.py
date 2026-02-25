"""ElevenLabs TTS Backend: Cloud-basierte Sprachsynthese via ElevenLabs API.

Hochwertige, natuerlich klingende Sprachsynthese mit
mehrsprachiger Unterstuetzung ueber die ElevenLabs REST API.
Unterstuetzt sowohl vollstaendige Synthese als auch Streaming.

Features:
  - Vollstaendige Synthese (synthesize) -> komplette Audio-Bytes
  - Streaming-Synthese (stream) -> Audio-Chunks als AsyncIterator
  - Mehrsprachige Unterstuetzung via eleven_multilingual_v2
  - Konfigurierbare Stimme, Modell und API-Key

Bibel-Referenz: §9.3 (Voice Channel), §12.2 (Optionale Dependencies)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ElevenLabs API base URL
_BASE_URL = "https://api.elevenlabs.io/v1"


@dataclass
class ElevenLabsConfig:
    """Konfiguration fuer ElevenLabs TTS."""

    api_key: str = ""
    voice_id: str = ""
    model: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    stability: float = 0.5
    similarity_boost: float = 0.75


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech Backend.

    Verwendet die ElevenLabs REST API fuer hochwertige
    Sprachsynthese. Erfordert einen gueltigen API-Key.

    Usage::

        tts = ElevenLabsTTS(api_key="sk-...", voice_id="abc123")
        audio = await tts.synthesize("Hallo Welt")

        async for chunk in tts.stream("Hallo Welt"):
            play(chunk)
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = "eleven_multilingual_v2",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._base_url = _BASE_URL
        log.info(
            "elevenlabs_tts_init",
            voice_id=voice_id,
            model=model,
        )

    @property
    def api_key(self) -> str:
        """Returns the configured API key."""
        return self._api_key

    @property
    def voice_id(self) -> str:
        """Returns the configured voice ID."""
        return self._voice_id

    @property
    def model(self) -> str:
        """Returns the configured model name."""
        return self._model

    def _build_url(self, *, stream: bool = False) -> str:
        """Builds the ElevenLabs TTS API URL.

        Args:
            stream: If True, appends /stream to the URL.

        Returns:
            The full API endpoint URL.
        """
        url = f"{self._base_url}/text-to-speech/{self._voice_id}"
        if stream:
            url += "/stream"
        return url

    def _build_headers(self) -> dict[str, str]:
        """Builds request headers with API key."""
        return {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

    def _build_payload(self, text: str) -> dict:
        """Builds the JSON payload for the TTS request.

        Args:
            text: The text to synthesize.

        Returns:
            The request payload dict.
        """
        return {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

    async def synthesize(self, text: str) -> bytes:
        """Synthesizes text to audio bytes via ElevenLabs API.

        Sends the full text to ElevenLabs and returns the complete
        audio response as bytes (MP3 format by default).

        Args:
            text: The text to convert to speech.

        Returns:
            Audio data as bytes. Returns empty bytes on error.
        """
        try:
            import httpx
        except ImportError:
            log.error("httpx_not_installed")
            raise ImportError(
                "httpx ist fuer ElevenLabs TTS erforderlich: pip install httpx"
            ) from None

        url = self._build_url(stream=False)
        headers = self._build_headers()
        payload = self._build_payload(text)

        log.info(
            "elevenlabs_synthesize_start",
            text_length=len(text),
            voice_id=self._voice_id,
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    log.error(
                        "elevenlabs_api_error",
                        status_code=response.status_code,
                        detail=response.text[:200],
                    )
                    return b""

                audio_data = response.content
                log.info(
                    "elevenlabs_synthesize_complete",
                    audio_size=len(audio_data),
                )
                return audio_data

        except httpx.HTTPStatusError as exc:
            log.error(
                "elevenlabs_http_status_error",
                status_code=exc.response.status_code,
                detail=str(exc),
            )
            return b""
        except httpx.RequestError as exc:
            log.error(
                "elevenlabs_request_error",
                error=str(exc),
            )
            return b""
        except Exception as exc:
            log.error(
                "elevenlabs_unexpected_error",
                error=str(exc),
            )
            return b""

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Streams synthesized audio chunks from ElevenLabs API.

        Uses the streaming endpoint to yield audio chunks as they
        are generated. Useful for low-latency playback.

        Args:
            text: The text to convert to speech.

        Yields:
            Audio data chunks as bytes.
        """
        try:
            import httpx
        except ImportError:
            log.error("httpx_not_installed")
            raise ImportError(
                "httpx ist fuer ElevenLabs TTS erforderlich: pip install httpx"
            ) from None

        url = self._build_url(stream=True)
        headers = self._build_headers()
        payload = self._build_payload(text)

        log.info(
            "elevenlabs_stream_start",
            text_length=len(text),
            voice_id=self._voice_id,
        )

        chunk_count = 0
        total_bytes = 0

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        log.error(
                            "elevenlabs_stream_api_error",
                            status_code=response.status_code,
                            detail=body.decode("utf-8", errors="replace")[:200],
                        )
                        return

                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            chunk_count += 1
                            total_bytes += len(chunk)
                            yield chunk

        except httpx.RequestError as exc:
            log.error(
                "elevenlabs_stream_request_error",
                error=str(exc),
            )
            return
        except Exception as exc:
            log.error(
                "elevenlabs_stream_unexpected_error",
                error=str(exc),
            )
            return
        finally:
            log.info(
                "elevenlabs_stream_complete",
                chunk_count=chunk_count,
                total_bytes=total_bytes,
            )
