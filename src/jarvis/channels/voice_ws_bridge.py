"""Voice-WebSocket-Bridge: Audio-Streaming zwischen Browser und Jarvis.

Erweitert die WebUI um bidirektionales Audio-Streaming:
  - Browser → Jarvis: Sprachnachricht (WebM/Opus) → Whisper → Text → Agent
  - Jarvis → Browser: Antworttext → Piper TTS → WAV → Browser

Wird vom WebChat-Widget über WebSocket-Messages mit type='voice_message'
angesprochen. Benötigt keine zusätzlichen Dependencies -- nutzt die
vorhandene Media-Pipeline für Transkription und TTS.

Bibel-Referenz: §9.3 (Voice Channel), §12.2 (Optionale Dependencies)
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class VoiceMessageHandler:
    """Verarbeitet Audio-Nachrichten aus WebSocket-Verbindungen.

    Wird von der WebUI als Handler für Voice-Messages eingebunden.
    Unterstützt:
      1. Eingehend: Base64-Audio → Whisper STT → Text
      2. Ausgehend: Text → Piper TTS → Base64-Audio

    Hinweis: Vormals 'VoiceWebSocketBridge' -- umbenannt um Kollision mit
    channels/voice_bridge.py:VoiceWebSocketBridge zu vermeiden.
    """

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._workspace = workspace_dir or Path.home() / ".jarvis" / "workspace" / "voice_ws"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._media: Any = None  # Lazy-loaded MediaPipeline

    def _get_media(self) -> Any:
        """Lazy-Load der MediaPipeline."""
        if self._media is None:
            from jarvis.mcp.media import MediaPipeline

            self._media = MediaPipeline(workspace_dir=self._workspace)
        return self._media

    async def transcribe_voice_message(
        self,
        audio_base64: str,
        audio_type: str = "audio/webm",
        language: str = "de",
    ) -> str | None:
        """Transkribiert eine Base64-encodierte Sprachnachricht.

        Args:
            audio_base64: Base64-encodierte Audio-Daten.
            audio_type: MIME-Type (audio/webm, audio/ogg, audio/wav).
            language: Sprache für STT.

        Returns:
            Transkribierter Text oder None bei Fehler.
        """
        # Dateiendung aus MIME-Type ableiten
        ext_map = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
            "audio/mp3": ".mp3",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
        }
        ext = ext_map.get(audio_type, ".webm")

        try:
            # Audio dekodieren und speichern
            audio_bytes = base64.b64decode(audio_base64)
            audio_path = self._workspace / f"voice_input{ext}"
            audio_path.write_bytes(audio_bytes)

            log.info(
                "voice_ws_received",
                size=len(audio_bytes),
                type=audio_type,
            )

            # Konvertierung zu WAV wenn nötig (für Whisper)
            wav_path = audio_path
            if ext != ".wav":
                wav_path = await self._convert_to_wav(audio_path)
                if wav_path is None:
                    # Fallback: direkt versuchen (Whisper kann manche Formate)
                    wav_path = audio_path

            # Transkription
            media = self._get_media()
            result = await media.transcribe_audio(
                str(wav_path),
                language=language,
                model="base",
            )

            if result.success and result.text.strip():
                log.info("voice_ws_transcribed", text=result.text[:80])
                return result.text.strip()

            log.warning("voice_ws_empty_transcription")
            return None

        except Exception as exc:
            log.error("voice_ws_transcribe_error", error=str(exc))
            return None

    async def synthesize_response(
        self,
        text: str,
        voice: str = "de_DE-thorsten-high",
    ) -> str | None:
        """Synthetisiert Text zu Base64-Audio für den Browser.

        Args:
            text: Zu sprechender Text.
            voice: Piper-Stimmenmodell.

        Returns:
            Base64-encodierte WAV-Daten oder None bei Fehler.
        """
        if not text.strip():
            return None

        try:
            media = self._get_media()
            output = self._workspace / "voice_response.wav"
            result = await media.text_to_speech(
                text,
                output_path=str(output),
                voice=voice,
            )

            if result.success and output.exists():
                audio_bytes = output.read_bytes()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                log.info("voice_ws_synthesized", size=len(audio_bytes))
                return audio_b64

            return None

        except Exception as exc:
            log.error("voice_ws_synthesis_error", error=str(exc))
            return None

    async def _convert_to_wav(self, input_path: Path) -> Path | None:
        """Konvertiert Audio-Datei zu WAV via ffmpeg.

        Returns:
            Pfad zur WAV-Datei oder None bei Fehler.
        """
        wav_path = input_path.with_suffix(".wav")

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i", str(input_path),
                "-ar", "16000",
                "-ac", "1",
                "-y",
                str(wav_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode == 0 and wav_path.exists():
                return wav_path

            log.warning("ffmpeg_conversion_failed", error=stderr.decode()[:200])
            return None

        except FileNotFoundError:
            log.warning("ffmpeg_not_available")
            return None

    async def handle_ws_voice_message(
        self,
        msg: dict[str, Any],
    ) -> dict[str, Any]:
        """Verarbeitet eine Voice-WebSocket-Nachricht.

        Erwartet:
          {
            "type": "voice_message",
            "audio_base64": "...",
            "audio_type": "audio/webm",
            "language": "de"
          }

        Returns:
          {
            "type": "voice_transcription",
            "text": "...",
            "audio_response_base64": "..." (optional)
          }
        """
        audio_b64 = msg.get("audio_base64", "")
        audio_type = msg.get("audio_type", "audio/webm")
        language = msg.get("language", "de")

        if not audio_b64:
            return {
                "type": "error",
                "error": "Keine Audio-Daten empfangen",
            }

        text = await self.transcribe_voice_message(audio_b64, audio_type, language)

        if not text:
            return {
                "type": "voice_transcription",
                "text": "",
                "error": "Keine Sprache erkannt",
            }

        return {
            "type": "voice_transcription",
            "text": text,
        }


# Rückwärtskompatibilitäts-Alias
VoiceWebSocketBridge = VoiceMessageHandler
