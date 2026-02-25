"""Voice Channel: Sprachein-/ausgabe mit Whisper STT + Piper TTS.

Lokale Sprachverarbeitung ohne Cloud-Abhängigkeit.
Whisper (faster-whisper) für Speech-to-Text,
Piper TTS für Text-to-Speech. Voice Activity Detection
für automatische Aufnahmeerkennung.

Features:
  - Whisper STT (GPU-beschleunigt via faster-whisper)
  - Piper TTS (lokal, Deutsch, schnell)
  - Voice Activity Detection (VAD) mit Silero
  - Hotkey-Aktivierung (Push-to-Talk)
  - Audio-Streaming über WebSocket
  - Konfigurierbare Sprache und Stimme

Bibel-Referenz: §9.3 (Voice Channel), §12.2 (Optionale Dependencies)
"""

from __future__ import annotations

import asyncio
import io
import struct
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Konfiguration
# ============================================================================


class STTBackend(StrEnum):
    """Unterstützte Speech-to-Text Backends."""

    WHISPER = "whisper"  # faster-whisper (lokal, GPU)
    WHISPER_CPP = "whisper_cpp"  # whisper.cpp (lokal, CPU-optimiert)


class TTSBackend(StrEnum):
    """Unterstützte Text-to-Speech Backends."""

    PIPER = "piper"  # Piper TTS (lokal, schnell)
    ESPEAK = "espeak"  # eSpeak-NG (Fallback, immer verfügbar)
    ELEVENLABS = "elevenlabs"  # ElevenLabs (Cloud, hochwertig)


@dataclass
class VoiceConfig:
    """Konfiguration für den Voice-Channel."""

    # STT
    stt_backend: STTBackend = STTBackend.WHISPER
    stt_model: str = "large-v3"
    stt_language: str = "de"
    stt_device: str = "auto"  # "auto", "cuda", "cpu"

    # TTS
    tts_backend: TTSBackend = TTSBackend.PIPER
    tts_model: str = "de_DE-thorsten-high"
    tts_speed: float = 1.0

    # Audio
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024

    # VAD (Voice Activity Detection)
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    silence_duration_ms: int = 800  # Stille-Dauer für End-of-Speech

    # Verhalten
    auto_listen: bool = False  # Automatisch nach TTS wieder zuhören
    beep_on_listen: bool = True

    # ElevenLabs TTS
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "hJAaR77ekN23CNyp0byH"
    elevenlabs_model: str = "eleven_multilingual_v2"


# ============================================================================
# Audio-Puffer für VAD
# ============================================================================


@dataclass
class AudioBuffer:
    """Sammelt Audio-Chunks und erkennt Sprache via VAD."""

    config: VoiceConfig
    _chunks: list[bytes] = field(default_factory=list)
    _is_speaking: bool = False
    _silence_start: float = 0.0
    _total_frames: int = 0

    def add_chunk(self, chunk: bytes) -> None:
        """Fügt einen Audio-Chunk hinzu."""
        self._chunks.append(chunk)
        self._total_frames += len(chunk) // 2  # 16-bit mono

    def clear(self) -> None:
        """Leert den Puffer."""
        self._chunks.clear()
        self._is_speaking = False
        self._silence_start = 0.0
        self._total_frames = 0

    def get_audio_data(self) -> bytes:
        """Gibt alle gesammelten Audio-Daten zurück."""
        return b"".join(self._chunks)

    @property
    def duration_seconds(self) -> float:
        """Dauer des aufgenommenen Audio in Sekunden."""
        if self.config.sample_rate == 0:
            return 0.0
        return self._total_frames / self.config.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def to_wav_bytes(self) -> bytes:
        """Konvertiert die Rohdaten in WAV-Format."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(self.get_audio_data())
        return buf.getvalue()


# ============================================================================
# STT Engine (Speech-to-Text)
# ============================================================================


class STTEngine:
    """Speech-to-Text Engine. Abstrahiert verschiedene Backends."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model: Any = None

    async def load(self) -> None:
        """Lädt das STT-Modell."""
        if self._config.stt_backend == STTBackend.WHISPER:
            await self._load_whisper()
        else:
            log.warning("stt_backend_not_implemented", backend=self._config.stt_backend)

    async def _load_whisper(self) -> None:
        """Lädt faster-whisper Modell."""
        try:
            from faster_whisper import WhisperModel

            device = self._config.stt_device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute_type = "float16" if device == "cuda" else "int8"

            log.info("loading_whisper_model", model=self._config.stt_model, device=device)

            self._model = WhisperModel(
                self._config.stt_model,
                device=device,
                compute_type=compute_type,
            )
            log.info("whisper_model_loaded")
        except ImportError:
            log.error("faster_whisper_not_installed")
            raise ImportError(
                "faster-whisper ist für STT erforderlich: pip install faster-whisper"
            ) from None

    async def transcribe(self, audio_data: bytes) -> str:
        """Transkribiert Audio-Daten zu Text.

        Args:
            audio_data: WAV-formatierte Audio-Daten

        Returns:
            Transkribierter Text
        """
        if self._model is None:
            raise RuntimeError("STT-Modell nicht geladen. Zuerst load() aufrufen.")

        # In temporäre Datei schreiben (faster-whisper braucht Dateipfad)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_data)
            tmp.flush()

            # In Thread ausführen (CPU-bound)
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._transcribe_sync, tmp.name)
            return text

    def _transcribe_sync(self, audio_path: str) -> str:
        """Synchrone Transkription (für run_in_executor)."""
        segments, info = self._model.transcribe(
            audio_path,
            language=self._config.stt_language,
            beam_size=5,
            vad_filter=True,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        result = " ".join(text_parts)
        log.info(
            "transcription_complete",
            text_length=len(result),
            language=info.language,
            probability=round(info.language_probability, 2),
        )
        return result


# ============================================================================
# TTS Engine (Text-to-Speech)
# ============================================================================


class TTSEngine:
    """Text-to-Speech Engine. Abstrahiert verschiedene Backends."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._synthesizer: Any = None

    async def load(self) -> None:
        """Lädt das TTS-Modell."""
        if self._config.tts_backend == TTSBackend.PIPER:
            await self._load_piper()
        elif self._config.tts_backend == TTSBackend.ESPEAK:
            log.info("using_espeak_tts_fallback")
        elif self._config.tts_backend == TTSBackend.ELEVENLABS:
            await self._load_elevenlabs()
        else:
            log.warning("tts_backend_not_implemented", backend=self._config.tts_backend)

    async def _load_piper(self) -> None:
        """Lädt Piper TTS Modell."""
        try:
            import piper  # noqa: F401

            log.info("piper_tts_available", model=self._config.tts_model)
        except ImportError:
            log.warning("piper_not_installed_using_espeak_fallback")
            self._config.tts_backend = TTSBackend.ESPEAK

    async def _load_elevenlabs(self) -> None:
        """Initialisiert ElevenLabs TTS Backend."""
        from jarvis.channels.tts_elevenlabs import ElevenLabsTTS

        if not self._config.elevenlabs_api_key:
            log.error("elevenlabs_api_key_missing")
            raise ValueError("ElevenLabs API-Key ist erforderlich")
        if not self._config.elevenlabs_voice_id:
            log.error("elevenlabs_voice_id_missing")
            raise ValueError("ElevenLabs Voice-ID ist erforderlich")

        self._elevenlabs_tts = ElevenLabsTTS(
            api_key=self._config.elevenlabs_api_key,
            voice_id=self._config.elevenlabs_voice_id,
            model=self._config.elevenlabs_model,
        )
        log.info(
            "elevenlabs_tts_loaded",
            voice_id=self._config.elevenlabs_voice_id,
            model=self._config.elevenlabs_model,
        )

    async def synthesize(self, text: str) -> bytes:
        """Synthetisiert Text zu Audio (WAV-Format).

        Args:
            text: Zu sprechender Text

        Returns:
            WAV-Audio-Daten als bytes
        """
        if self._config.tts_backend == TTSBackend.PIPER:
            return await self._synthesize_piper(text)
        elif self._config.tts_backend == TTSBackend.ELEVENLABS:
            return await self._synthesize_elevenlabs(text)
        return await self._synthesize_espeak(text)

    async def _synthesize_piper(self, text: str) -> bytes:
        """TTS via Piper."""
        loop = asyncio.get_running_loop()

        def _sync_piper() -> bytes:
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "piper",
                        "--model",
                        self._config.tts_model,
                        "--output-raw",
                    ],
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Piper error: {result.stderr.decode()}")
                return self._raw_to_wav(result.stdout, self._config.sample_rate)
            except FileNotFoundError:
                raise RuntimeError("Piper binary nicht gefunden") from None

        return await loop.run_in_executor(None, _sync_piper)

    async def _synthesize_espeak(self, text: str) -> bytes:
        """TTS via eSpeak-NG (Fallback)."""
        proc = await asyncio.create_subprocess_exec(
            "espeak-ng",
            "-v",
            "de",
            "--stdout",
            text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            log.warning("espeak_error", stderr=stderr.decode())
            return b""

        return stdout

    async def _synthesize_elevenlabs(self, text: str) -> bytes:
        """TTS via ElevenLabs API."""
        if not hasattr(self, "_elevenlabs_tts"):
            await self._load_elevenlabs()
        return await self._elevenlabs_tts.synthesize(text)

    @staticmethod
    def _raw_to_wav(raw_data: bytes, sample_rate: int) -> bytes:
        """Konvertiert Rohdaten (16-bit mono PCM) zu WAV."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_data)
        return buf.getvalue()


# ============================================================================
# VAD (Voice Activity Detection)
# ============================================================================


class VADDetector:
    """Voice Activity Detection mit Energie-basiertem Fallback.

    Erkennt ob der User gerade spricht. Verwendet Silero VAD
    wenn verfügbar, sonst einfache Energie-Schwelle.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model: Any = None
        self._use_silero = False

    async def load(self) -> None:
        """Lädt das VAD-Modell."""
        try:
            import torch

            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            self._use_silero = True
            log.info("silero_vad_loaded")
        except Exception:
            log.info("using_energy_based_vad_fallback")
            self._use_silero = False

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Prüft ob ein Audio-Chunk Sprache enthält.

        Args:
            audio_chunk: 16-bit mono PCM Audio

        Returns:
            True wenn Sprache erkannt
        """
        if self._use_silero and self._model is not None:
            return self._silero_detect(audio_chunk)
        return self._energy_detect(audio_chunk)

    def _silero_detect(self, chunk: bytes) -> bool:
        """VAD via Silero-Modell."""
        try:
            import torch

            # Bytes → Float-Tensor
            samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
            tensor = torch.FloatTensor(samples) / 32768.0
            confidence = self._model(tensor, self._config.sample_rate).item()
            return bool(confidence > self._config.vad_threshold)
        except Exception:
            return self._energy_detect(chunk)

    def _energy_detect(self, chunk: bytes) -> bool:
        """Einfache Energie-basierte Spracherkennung (Fallback)."""
        if len(chunk) < 2:
            return False
        # RMS-Energie berechnen
        samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
        if not samples:
            return False
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        # Normalisieren auf 0-1 Bereich
        normalized = rms / 32768.0
        return bool(normalized > 0.01)  # Einfacher Schwellwert


# ============================================================================
# Voice Channel
# ============================================================================


class VoiceChannel(Channel):
    """Voice Channel: Sprachein-/ausgabe für Jarvis. [B§9.3]

    Nutzt Whisper für STT, Piper für TTS und optional
    Silero für Voice Activity Detection.
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self._config = config or VoiceConfig()
        self._handler: MessageHandler | None = None
        self._stt = STTEngine(self._config)
        self._tts = TTSEngine(self._config)
        self._vad = VADDetector(self._config)
        self._audio_buffer = AudioBuffer(config=self._config)
        self._is_listening = False
        self._is_processing = False

    @property
    def name(self) -> str:
        return "voice"

    async def start(self, handler: MessageHandler) -> None:
        """Lädt alle Modelle und startet den Voice-Channel."""
        self._handler = handler
        log.info("voice_channel_loading_models")

        # Modelle parallel laden
        await asyncio.gather(
            self._stt.load(),
            self._tts.load(),
            self._vad.load() if self._config.vad_enabled else asyncio.sleep(0),
        )
        log.info("voice_channel_ready")

    async def stop(self) -> None:
        """Stoppt den Voice-Channel."""
        self._is_listening = False
        self._audio_buffer.clear()
        log.info("voice_channel_stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Spricht die Antwort aus (TTS)."""
        if message.text:
            audio = await self._tts.synthesize(message.text)
            if audio:
                await self._play_audio(audio)

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Fragt den User per Sprache um Bestätigung.

        Spricht die Frage, wartet auf 'Ja' oder 'Nein'.
        """
        prompt = f"Darf ich {action.tool} ausführen? Grund: {reason}. Sage Ja oder Nein."
        audio = await self._tts.synthesize(prompt)
        if audio:
            await self._play_audio(audio)

        # Aufnahme starten für Antwort
        response = await self.listen_once(timeout=10.0)
        if response:
            normalized = response.lower().strip()
            return any(w in normalized for w in ("ja", "yes", "ok", "mach"))
        return False

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Voice-Channel sammelt Tokens und spricht am Ende."""
        pass

    async def listen_once(self, timeout: float = 30.0) -> str | None:
        """Nimmt eine einzelne Äußerung auf und transkribiert sie.

        Returns:
            Transkribierter Text oder None bei Timeout
        """
        self._audio_buffer.clear()
        self._is_listening = True

        try:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                if not self._audio_buffer.is_empty and self._audio_buffer.duration_seconds > 0.5:
                    # Audio transkribieren
                    wav_data = self._audio_buffer.to_wav_bytes()
                    text = await self._stt.transcribe(wav_data)
                    if text.strip():
                        return text.strip()
                await asyncio.sleep(0.1)
            return None
        finally:
            self._is_listening = False

    async def process_audio_chunk(self, chunk: bytes) -> str | None:
        """Verarbeitet einen Audio-Chunk (von WebSocket oder Mikrofon).

        Verwendet VAD zur Erkennung von Sprech-Pausen.
        Gibt transkribierten Text zurück wenn eine Äußerung
        erkannt wurde, sonst None.

        Args:
            chunk: 16-bit mono PCM Audio-Daten

        Returns:
            Transkribierter Text oder None
        """
        is_speech = self._vad.is_speech(chunk)

        if is_speech:
            self._audio_buffer.add_chunk(chunk)
            self._audio_buffer._is_speaking = True
            self._audio_buffer._silence_start = 0.0
        elif self._audio_buffer._is_speaking:
            # Sprache hat aufgehört
            if self._audio_buffer._silence_start == 0.0:
                self._audio_buffer._silence_start = time.monotonic()
            else:
                silence_ms = (time.monotonic() - self._audio_buffer._silence_start) * 1000
                if silence_ms >= self._config.silence_duration_ms:
                    # End-of-Speech erkannt
                    if self._audio_buffer.duration_seconds >= 0.3:
                        wav_data = self._audio_buffer.to_wav_bytes()
                        self._audio_buffer.clear()
                        text = await self._stt.transcribe(wav_data)
                        if text.strip():
                            return text.strip()
                    self._audio_buffer.clear()

        return None

    async def handle_voice_message(self, audio_data: bytes) -> str | None:
        """Verarbeitet eine komplette Sprachnachricht.

        Nimmt fertige WAV-Daten entgegen (z.B. vom WebSocket),
        transkribiert und sendet an den Handler.

        Args:
            audio_data: WAV-formatierte Audio-Daten

        Returns:
            Antworttext oder None
        """
        if not self._handler:
            return None

        self._is_processing = True
        try:
            # Transkribieren
            text = await self._stt.transcribe(audio_data)
            if not text.strip():
                return None

            log.info("voice_transcription", text=text)

            # An Handler senden
            incoming = IncomingMessage(
                text=text,
                channel="voice",
                session_id="voice_session",
                user_id="voice_user",
                metadata={"source": "voice", "stt_backend": self._config.stt_backend.value},
            )
            response = await self._handler(incoming)

            # Antwort sprechen
            if response.text:
                audio = await self._tts.synthesize(response.text)
                if audio:
                    await self._play_audio(audio)

            return response.text

        finally:
            self._is_processing = False

    async def _play_audio(self, wav_data: bytes) -> None:
        """Spielt Audio-Daten ab.

        Versucht zuerst pyaudio/sounddevice, dann aplay als Fallback.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "aplay",
                "-q",
                "-",
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(input=wav_data)
        except FileNotFoundError:
            # aplay nicht verfügbar → Daten einfach loggen
            log.warning("audio_playback_not_available", data_size=len(wav_data))

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    @property
    def is_processing(self) -> bool:
        return self._is_processing
