"""Voice-WebSocket-Bridge: Audio-Streaming vom Browser zum Voice-Channel.

Empfängt Audio-Chunks über WebSocket, transkribiert via Whisper
und leitet den Text an den Gateway weiter. Ermöglicht Echtzeit-
Sprachsteuerung direkt aus dem WebChat-Widget.

Protokoll (Client → Server):
  { "type": "audio_start", "format": "webm", "sample_rate": 16000 }
  { "type": "audio_chunk", "data": "<base64>" }
  { "type": "audio_stop" }

Protokoll (Server → Client):
  { "type": "transcription", "text": "...", "final": true }
  { "type": "voice_status", "status": "listening|processing|ready" }
  { "type": "voice_error", "error": "..." }

Bibel-Referenz: §9.3 (Voice), §9.3 (WebUI)
"""

from __future__ import annotations

import asyncio
import base64
import io
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Audio-Accumulator
# ============================================================================


@dataclass
class AudioAccumulator:
    """Sammelt Base64-encoded Audio-Chunks und gibt WAV zurück."""

    chunks: list[bytes] = field(default_factory=list)
    format: str = "webm"
    sample_rate: int = 16000
    _total_bytes: int = 0

    def add_chunk(self, base64_data: str) -> None:
        """Fügt einen Base64-kodierten Audio-Chunk hinzu."""
        raw = base64.b64decode(base64_data)
        self.chunks.append(raw)
        self._total_bytes += len(raw)

    def clear(self) -> None:
        self.chunks.clear()
        self._total_bytes = 0

    @property
    def duration_estimate_seconds(self) -> float:
        """Grobe Schätzung der Dauer (für WebM ~12kB/s bei Opus)."""
        if self.format == "webm":
            return self._total_bytes / 12_000
        # PCM 16-bit mono
        return self._total_bytes / (self.sample_rate * 2)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def get_blob(self) -> bytes:
        """Gibt alle Chunks als einen Blob zurück."""
        return b"".join(self.chunks)


# ============================================================================
# Voice-WebSocket-Bridge
# ============================================================================


class VoiceWebSocketBridge:
    """Brücke zwischen WebSocket-Audio und dem Voice-Channel.

    Verarbeitet Audio-Streams aus dem Browser:
    1. Sammelt Chunks in einem Accumulator
    2. Konvertiert via ffmpeg zu WAV (wenn nötig)
    3. Transkribiert via Whisper
    4. Sendet Transkription zurück an den Client
    """

    def __init__(
        self,
        *,
        workspace_dir: Path | None = None,
        whisper_model: str = "base",
        language: str = "de",
    ) -> None:
        self._workspace = workspace_dir or Path.home() / ".jarvis" / "workspace" / "voice_ws"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._whisper_model = whisper_model
        self._language = language
        self._stt_engine: Any = None
        self._active_sessions: dict[str, AudioAccumulator] = {}

    async def initialize(self) -> bool:
        """Lädt das Whisper-Modell. Gibt False zurück wenn nicht verfügbar."""
        try:
            from faster_whisper import WhisperModel

            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

            compute_type = "float16" if device == "cuda" else "int8"

            log.info(
                "voice_bridge_loading_whisper",
                model=self._whisper_model,
                device=device,
            )

            loop = asyncio.get_running_loop()
            self._stt_engine = await loop.run_in_executor(
                None,
                lambda: WhisperModel(self._whisper_model, device=device, compute_type=compute_type),
            )

            log.info("voice_bridge_ready")
            return True

        except ImportError:
            log.warning("voice_bridge_whisper_not_available")
            return False
        except Exception as exc:
            log.error("voice_bridge_init_failed", error=str(exc))
            return False

    async def handle_ws_message(
        self,
        session_id: str,
        msg: dict[str, Any],
        send_fn: Any,
    ) -> str | None:
        """Verarbeitet eine Voice-WebSocket-Nachricht.

        Args:
            session_id: WebSocket-Session-ID.
            msg: Eingehende Nachricht (parsed JSON).
            send_fn: Async-Funktion zum Senden an den Client.

        Returns:
            Transkribierter Text bei audio_stop, sonst None.
        """
        msg_type = msg.get("type", "")

        if msg_type == "audio_start":
            return await self._handle_audio_start(session_id, msg, send_fn)

        if msg_type == "audio_chunk":
            return await self._handle_audio_chunk(session_id, msg)

        if msg_type == "audio_stop":
            return await self._handle_audio_stop(session_id, send_fn)

        return None

    async def _handle_audio_start(
        self,
        session_id: str,
        msg: dict[str, Any],
        send_fn: Any,
    ) -> None:
        """Startet eine neue Audio-Aufnahme-Session."""
        acc = AudioAccumulator(
            format=msg.get("format", "webm"),
            sample_rate=msg.get("sample_rate", 16000),
        )
        self._active_sessions[session_id] = acc

        await send_fn({
            "type": "voice_status",
            "status": "listening",
        })

        log.info("voice_session_started", session_id=session_id, format=acc.format)
        return None

    async def _handle_audio_chunk(
        self,
        session_id: str,
        msg: dict[str, Any],
    ) -> None:
        """Fügt einen Audio-Chunk zur aktiven Session hinzu."""
        acc = self._active_sessions.get(session_id)
        if acc is None:
            return None

        data = msg.get("data", "")
        if data:
            acc.add_chunk(data)

        return None

    async def _handle_audio_stop(
        self,
        session_id: str,
        send_fn: Any,
    ) -> str | None:
        """Beendet die Aufnahme und transkribiert das Audio."""
        acc = self._active_sessions.pop(session_id, None)
        if acc is None or acc.is_empty:
            await send_fn({
                "type": "voice_error",
                "error": "Keine Audio-Daten empfangen.",
            })
            return None

        # Zu kurze Aufnahmen verwerfen
        if acc.duration_estimate_seconds < 0.3:
            await send_fn({
                "type": "voice_error",
                "error": "Aufnahme zu kurz.",
            })
            return None

        await send_fn({
            "type": "voice_status",
            "status": "processing",
        })

        try:
            text = await self._transcribe(acc)

            if not text or not text.strip():
                await send_fn({
                    "type": "transcription",
                    "text": "",
                    "final": True,
                })
                await send_fn({
                    "type": "voice_status",
                    "status": "ready",
                })
                return None

            await send_fn({
                "type": "transcription",
                "text": text,
                "final": True,
            })
            await send_fn({
                "type": "voice_status",
                "status": "ready",
            })

            log.info("voice_transcription", session_id=session_id, text_length=len(text))
            return text

        except Exception as exc:
            log.error("voice_transcription_failed", error=str(exc))
            await send_fn({
                "type": "voice_error",
                "error": f"Transkription fehlgeschlagen: {exc}",
            })
            await send_fn({
                "type": "voice_status",
                "status": "ready",
            })
            return None

    async def _transcribe(self, acc: AudioAccumulator) -> str:
        """Transkribiert gesammelte Audio-Daten.

        1. Schreibt Blob in temp-Datei
        2. Konvertiert zu WAV via ffmpeg (wenn WebM/Opus)
        3. Transkribiert via Whisper
        """
        if self._stt_engine is None:
            raise RuntimeError("Whisper nicht geladen. initialize() aufrufen.")

        blob = acc.get_blob()

        # Temp-Datei mit Original-Format
        suffix = f".{acc.format}" if acc.format != "webm" else ".webm"
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            dir=str(self._workspace),
            delete=False,
        ) as tmp:
            tmp.write(blob)
            input_path = tmp.name

        wav_path = input_path

        # WebM/Opus → WAV konvertieren
        if acc.format in ("webm", "ogg", "mp3"):
            wav_path = input_path + ".wav"
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-i", input_path,
                    "-ar", str(acc.sample_rate),
                    "-ac", "1",
                    "-f", "wav",
                    "-y",
                    wav_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg: {stderr.decode()[:200]}")

            except FileNotFoundError:
                raise RuntimeError(
                    "ffmpeg nicht installiert. Für Voice-Streaming: apt install ffmpeg"
                ) from None

        # Whisper-Transkription (in Thread)
        loop = asyncio.get_running_loop()
        engine = self._stt_engine
        language = self._language

        def _transcribe_sync() -> str:
            segments, info = engine.transcribe(
                wav_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments)

        try:
            text = await loop.run_in_executor(None, _transcribe_sync)
            return text
        finally:
            # Temp-Dateien aufräumen
            for p in (input_path, wav_path):
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass

    def cancel_session(self, session_id: str) -> None:
        """Bricht eine laufende Audio-Session ab."""
        self._active_sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._active_sessions)
