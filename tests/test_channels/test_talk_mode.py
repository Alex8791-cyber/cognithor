"""Tests für talk_mode.py – Continuous Conversation Mode."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.talk_mode import TalkMode
from jarvis.channels.wake_word import WakeWordDetector


class TestTalkMode:
    def test_init(self):
        voice = MagicMock()
        wake = WakeWordDetector(backend="energy")
        tm = TalkMode(voice, wake)
        assert not tm.is_active
        assert not tm.auto_listen

    def test_auto_listen_property(self):
        voice = MagicMock()
        wake = WakeWordDetector(backend="energy")
        tm = TalkMode(voice, wake, auto_listen=True)
        assert tm.auto_listen is True
        tm.auto_listen = False
        assert tm.auto_listen is False

    @pytest.mark.asyncio
    async def test_start_sets_active(self):
        voice = MagicMock()
        voice._handler = None
        voice.listen_once = AsyncMock(return_value=None)
        voice._play_audio = AsyncMock()
        wake = WakeWordDetector(backend="energy")

        tm = TalkMode(voice, wake)
        await tm.start()
        assert tm.is_active
        await tm.stop()
        assert not tm.is_active

    @pytest.mark.asyncio
    async def test_stop_when_not_active(self):
        voice = MagicMock()
        wake = WakeWordDetector(backend="energy")
        tm = TalkMode(voice, wake)
        # Should not raise
        await tm.stop()
        assert not tm.is_active

    @pytest.mark.asyncio
    async def test_double_start_warns(self):
        voice = MagicMock()
        voice._handler = None
        voice.listen_once = AsyncMock(return_value=None)
        voice._play_audio = AsyncMock()
        wake = WakeWordDetector(backend="energy")

        tm = TalkMode(voice, wake)
        await tm.start()
        await tm.start()  # Should warn but not crash
        await tm.stop()
