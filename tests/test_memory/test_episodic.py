"""Tests für memory/episodic.py · Tier 2."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from jarvis.memory.episodic import EpisodicMemory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ep_dir(tmp_path: Path) -> Path:
    return tmp_path / "episodes"


@pytest.fixture
def ep(ep_dir: Path) -> EpisodicMemory:
    return EpisodicMemory(ep_dir)


class TestEpisodicMemory:
    def test_ensure_directory(self, ep: EpisodicMemory, ep_dir: Path):
        assert not ep_dir.exists()
        ep.ensure_directory()
        assert ep_dir.exists()

    def test_append_entry(self, ep: EpisodicMemory):
        ts = datetime(2026, 2, 21, 14, 30)
        result = ep.append_entry("Test-Thema", "Details hier", timestamp=ts)
        assert "Test-Thema" in result
        assert "Details hier" in result
        assert "14:30" in result

    def test_append_creates_file(self, ep: EpisodicMemory):
        ts = datetime(2026, 2, 21, 10, 0)
        ep.append_entry("Thema", "Inhalt", timestamp=ts)
        file_path = ep.directory / "2026-02-21.md"
        assert file_path.exists()
        content = file_path.read_text(encoding="utf-8")
        assert "# 2026-02-21" in content
        assert "## 10:00 · Thema" in content

    def test_append_multiple_entries(self, ep: EpisodicMemory):
        ts1 = datetime(2026, 2, 21, 10, 0)
        ts2 = datetime(2026, 2, 21, 14, 30)
        ep.append_entry("Morgens", "Erster Eintrag", timestamp=ts1)
        ep.append_entry("Nachmittags", "Zweiter Eintrag", timestamp=ts2)
        content = ep.get_date(date(2026, 2, 21))
        assert "Morgens" in content
        assert "Nachmittags" in content

    def test_get_date_empty(self, ep: EpisodicMemory):
        assert ep.get_date(date(2026, 1, 1)) == ""

    def test_get_date_existing(self, ep: EpisodicMemory):
        ts = datetime(2026, 3, 15, 12, 0)
        ep.append_entry("Test", "Inhalt", timestamp=ts)
        content = ep.get_date(date(2026, 3, 15))
        assert "Test" in content
        assert "Inhalt" in content

    def test_get_recent(self, ep: EpisodicMemory):
        today = date.today()
        yesterday = today - timedelta(days=1)

        ep.append_entry(
            "Heute", "H", timestamp=datetime.combine(today, datetime.min.time().replace(hour=10))
        )
        ep.append_entry(
            "Gestern",
            "G",
            timestamp=datetime.combine(yesterday, datetime.min.time().replace(hour=10)),
        )

        recent = ep.get_recent(days=2)
        assert len(recent) == 2
        assert recent[0][0] == today  # Neueste zuerst
        assert recent[1][0] == yesterday

    def test_get_recent_empty(self, ep: EpisodicMemory):
        assert ep.get_recent() == []

    def test_list_dates(self, ep: EpisodicMemory):
        for d in [date(2026, 2, 19), date(2026, 2, 20), date(2026, 2, 21)]:
            ts = datetime.combine(d, datetime.min.time().replace(hour=10))
            ep.append_entry("Test", "X", timestamp=ts)

        dates = ep.list_dates()
        assert len(dates) == 3
        assert dates[0] == date(2026, 2, 21)  # Neueste zuerst
        assert dates[-1] == date(2026, 2, 19)

    def test_list_dates_no_dir(self, tmp_path: Path):
        ep = EpisodicMemory(tmp_path / "nonexistent")
        assert ep.list_dates() == []

    def test_append_default_timestamp(self, ep: EpisodicMemory):
        ep.append_entry("Auto-Zeit", "Kein timestamp angegeben")
        today_content = ep.get_date(date.today())
        assert "Auto-Zeit" in today_content

    def test_directory_property(self, ep: EpisodicMemory, ep_dir: Path):
        assert ep.directory == ep_dir
