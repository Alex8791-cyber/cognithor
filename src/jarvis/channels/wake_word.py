"""Wake Word Detection: Erkennt Aktivierungswort im Audio-Stream.

Unterstützt mehrere Backends:
  - Vosk (offline, kostenlos, gutes Deutsch)
  - Porcupine (Picovoice, offline, bessere Accuracy)
  - Energy-based Fallback (einfache Audio-Schwelle)

Bibel-Referenz: §9.3 (Voice Channel Extension)
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any, AsyncIterator

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class WakeWordDetector:
    """Erkennt Wake Word im Audio-Stream."""

    def __init__(
        self,
        keywords: list[str] | None = None,
        backend: str = "vosk",
        sensitivity: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self._keywords = keywords or ["jarvis"]
        self._backend = backend
        self._sensitivity = sensitivity
        self._sample_rate = sample_rate
        self._model: Any = None
        self._running = False

    async def load(self) -> None:
        """Lädt das Wake-Word-Modell."""
        if self._backend == "vosk":
            await self._load_vosk()
        elif self._backend == "porcupine":
            await self._load_porcupine()
        else:
            log.info("using_energy_based_wake_word_fallback")

    async def _load_vosk(self) -> None:
        """Lädt Vosk für Keyword-Spotting."""
        try:
            from vosk import KaldiRecognizer, Model
            import json as _json

            model = Model(lang="de")
            self._model = KaldiRecognizer(model, self._sample_rate)
            self._model.SetWords(True)
            log.info("vosk_wake_word_loaded", keywords=self._keywords)
        except ImportError:
            log.warning("vosk_not_installed_using_energy_fallback")
            self._backend = "energy"
        except Exception as exc:
            log.warning("vosk_load_failed", error=str(exc))
            self._backend = "energy"

    async def _load_porcupine(self) -> None:
        """Lädt Porcupine für Wake-Word-Detection."""
        try:
            import pvporcupine
            self._model = pvporcupine.create(
                keywords=self._keywords,
                sensitivities=[self._sensitivity] * len(self._keywords),
            )
            log.info("porcupine_wake_word_loaded", keywords=self._keywords)
        except ImportError:
            log.warning("porcupine_not_installed_using_energy_fallback")
            self._backend = "energy"
        except Exception as exc:
            log.warning("porcupine_load_failed", error=str(exc))
            self._backend = "energy"

    async def listen(self, audio_stream: AsyncIterator[bytes]) -> bool:
        """Lauscht kontinuierlich und gibt True zurück wenn Wake Word erkannt.

        Args:
            audio_stream: Async Iterator von Audio-Chunks (16-bit mono PCM).

        Returns:
            True wenn Wake Word erkannt wurde.
        """
        self._running = True
        try:
            async for chunk in audio_stream:
                if not self._running:
                    return False
                if self._detect_in_chunk(chunk):
                    log.info("wake_word_detected", backend=self._backend)
                    return True
        except asyncio.CancelledError:
            return False
        return False

    def detect_in_chunk(self, chunk: bytes) -> bool:
        """Prüft einen einzelnen Audio-Chunk auf das Wake Word.

        Args:
            chunk: 16-bit mono PCM Audio-Daten.

        Returns:
            True wenn Wake Word erkannt.
        """
        return self._detect_in_chunk(chunk)

    def _detect_in_chunk(self, chunk: bytes) -> bool:
        """Internes Detection dispatching."""
        if self._backend == "vosk":
            return self._detect_vosk(chunk)
        elif self._backend == "porcupine":
            return self._detect_porcupine(chunk)
        return self._detect_energy(chunk)

    def _detect_vosk(self, chunk: bytes) -> bool:
        """Wake-Word-Detection via Vosk."""
        if self._model is None:
            return False
        import json as _json
        if self._model.AcceptWaveform(chunk):
            result = _json.loads(self._model.Result())
            text = result.get("text", "").lower()
            return any(kw.lower() in text for kw in self._keywords)
        partial = _json.loads(self._model.PartialResult())
        text = partial.get("partial", "").lower()
        return any(kw.lower() in text for kw in self._keywords)

    def _detect_porcupine(self, chunk: bytes) -> bool:
        """Wake-Word-Detection via Porcupine."""
        if self._model is None:
            return False
        try:
            pcm = struct.unpack(f"<{len(chunk) // 2}h", chunk)
            frame_length = self._model.frame_length
            for i in range(0, len(pcm) - frame_length + 1, frame_length):
                frame = pcm[i:i + frame_length]
                keyword_index = self._model.process(frame)
                if keyword_index >= 0:
                    return True
        except Exception:
            pass
        return False

    def _detect_energy(self, chunk: bytes) -> bool:
        """Einfache Energie-basierte Erkennung (letzter Fallback).

        Erkennt nicht wirklich das Wake Word, sondern nur ob
        Sprache vorhanden ist. Muss mit STT kombiniert werden.
        """
        if len(chunk) < 2:
            return False
        samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        normalized = rms / 32768.0
        return bool(normalized > 0.02)

    def stop(self) -> None:
        """Stoppt die Wake-Word-Detection."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def keywords(self) -> list[str]:
        return self._keywords
