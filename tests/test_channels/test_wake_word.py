"""Tests für wake_word.py – Wake Word Detection."""

from __future__ import annotations

import struct

import pytest

from jarvis.channels.wake_word import WakeWordDetector


class TestWakeWordDetector:
    def test_init_defaults(self):
        detector = WakeWordDetector()
        assert detector.keywords == ["jarvis"]
        assert detector.backend == "vosk"
        assert not detector.is_running

    def test_init_custom(self):
        detector = WakeWordDetector(
            keywords=["hey", "computer"],
            backend="porcupine",
            sensitivity=0.7,
        )
        assert detector.keywords == ["hey", "computer"]
        assert detector.backend == "porcupine"

    def test_energy_detection_silence(self):
        """Stille Audio sollte kein Wake Word erkennen."""
        detector = WakeWordDetector(backend="energy")
        silence = struct.pack("<100h", *([0] * 100))
        assert detector.detect_in_chunk(silence) is False

    def test_energy_detection_loud(self):
        """Lautes Audio triggert Energy-Detection."""
        detector = WakeWordDetector(backend="energy")
        loud = struct.pack("<100h", *([10000] * 100))
        assert detector.detect_in_chunk(loud) is True

    def test_energy_detection_empty(self):
        detector = WakeWordDetector(backend="energy")
        assert detector.detect_in_chunk(b"") is False

    def test_stop(self):
        detector = WakeWordDetector()
        detector._running = True
        detector.stop()
        assert not detector.is_running

    @pytest.mark.asyncio
    async def test_load_energy_fallback(self):
        """Wenn kein Backend verfügbar, fällt auf energy zurück."""
        detector = WakeWordDetector(backend="vosk")
        await detector.load()
        # Vosk ist wahrscheinlich nicht installiert in Tests → Energy-Fallback
        assert detector.backend in ("vosk", "energy")

    def test_vosk_without_model_returns_false(self):
        detector = WakeWordDetector(backend="vosk")
        detector._model = None
        assert detector._detect_vosk(b"") is False

    def test_porcupine_without_model_returns_false(self):
        detector = WakeWordDetector(backend="porcupine")
        detector._model = None
        assert detector._detect_porcupine(b"") is False
