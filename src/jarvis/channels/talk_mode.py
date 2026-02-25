"""Talk Mode: Kontinuierlicher Gesprächsmodus.

Flow:
  [Idle: Wake Word Detection aktiv]
      ↓ "Jarvis" erkannt
  [Beep/Vibration als Bestätigung]
      ↓
  [Zuhören: STT mit VAD-basiertem End-of-Speech]
      ↓ Stille erkannt → Transkription
  [Verarbeiten: Text → Gateway → Agent-Loop]
      ↓ Antwort
  [Sprechen: TTS streamt Antwort]
      ↓ Sprache beendet
  [Zurück zu Idle / oder auto_listen=True → weiter zuhören]

Bibel-Referenz: §9.3 (Voice Channel Extension)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.channels.voice import VoiceChannel
    from jarvis.channels.wake_word import WakeWordDetector

log = get_logger(__name__)


class TalkMode:
    """Kontinuierlicher Gesprächsmodus: Wake Word → Zuhören → Antwort → Wiederholen."""

    def __init__(
        self,
        voice_channel: VoiceChannel,
        wake_detector: WakeWordDetector,
        *,
        auto_listen: bool = False,
        confirmation_beep: bool = True,
    ) -> None:
        self._voice = voice_channel
        self._wake = wake_detector
        self._auto_listen = auto_listen
        self._confirmation_beep = confirmation_beep
        self._active = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Startet die Talk-Mode-Endlosschleife."""
        if self._active:
            log.warning("talk_mode_already_active")
            return

        self._active = True
        self._task = asyncio.create_task(self._loop())
        log.info("talk_mode_started", auto_listen=self._auto_listen)

    async def stop(self) -> None:
        """Stoppt den Talk Mode."""
        self._active = False
        self._wake.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        log.info("talk_mode_stopped")

    async def _loop(self) -> None:
        """Hauptschleife des Talk Mode."""
        try:
            while self._active:
                # Phase 1: Wake Word Detection
                log.debug("talk_mode_waiting_for_wake_word")
                detected = await self._wait_for_wake_word()
                if not detected or not self._active:
                    continue

                # Phase 2: Bestätigung
                if self._confirmation_beep:
                    await self._play_confirmation()

                # Phase 3: Zuhören + Transkribieren
                text = await self._voice.listen_once(timeout=15.0)
                if not text or not self._active:
                    log.debug("talk_mode_no_speech_detected")
                    continue

                log.info("talk_mode_heard", text=text[:100])

                # Phase 4: Verarbeiten (via Voice-Channel Handler)
                if self._voice._handler:
                    from jarvis.models import IncomingMessage

                    incoming = IncomingMessage(
                        text=text,
                        channel="voice",
                        session_id="talk_mode_session",
                        user_id="voice_user",
                        metadata={"source": "talk_mode"},
                    )
                    response = await self._voice._handler(incoming)

                    # Phase 5: Antwort sprechen
                    if response.text:
                        from jarvis.models import OutgoingMessage

                        await self._voice.send(
                            OutgoingMessage(
                                channel="voice",
                                text=response.text,
                                session_id="talk_mode_session",
                            )
                        )

                # Auto-Listen: Direkt weiter zuhören ohne Wake Word
                if self._auto_listen and self._active:
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("talk_mode_error", error=str(exc))
        finally:
            self._active = False

    async def _wait_for_wake_word(self) -> bool:
        """Wartet auf Wake-Word via WakeWordDetector."""
        # Simplified: Use detect_in_chunk in a polling loop
        # In production, this would use a real audio stream
        try:
            # Wait with timeout to allow periodic activity checks
            await asyncio.sleep(0.1)
            return False  # Placeholder: real implementation uses audio stream
        except asyncio.CancelledError:
            return False

    async def _play_confirmation(self) -> None:
        """Spielt einen kurzen Bestätigungston."""
        # Generate a simple beep tone
        import io
        import struct
        import wave

        sample_rate = 16000
        duration = 0.15
        frequency = 880  # A5

        num_samples = int(sample_rate * duration)
        import math

        samples = [
            int(16000 * math.sin(2 * math.pi * frequency * t / sample_rate))
            for t in range(num_samples)
        ]

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        await self._voice._play_audio(buf.getvalue())

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def auto_listen(self) -> bool:
        return self._auto_listen

    @auto_listen.setter
    def auto_listen(self, value: bool) -> None:
        self._auto_listen = value
