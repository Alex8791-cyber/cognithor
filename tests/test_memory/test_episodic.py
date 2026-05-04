"""Tests für memory/episodic.py · Tier 2."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from cognithor.memory.episodic import EpisodicMemory

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


class TestEpisodicMemoryProvenance:
    """TRUST-9 wiring: passing ``provenance_source_type`` +
    ``provenance_source_id`` to ``append_entry`` writes a tag to the
    canonical PROVENANCE_LEDGER keyed by a deterministic episode id.
    """

    def test_episode_item_id_shape(self) -> None:
        ts = datetime(2026, 2, 21, 14, 30)
        item_id = EpisodicMemory._episode_item_id(ts, "  My Topic / Slug!  ")
        assert item_id == "episode:2026-02-21:14:30:my-topic-slug"

    def test_episode_item_id_empty_topic_falls_back(self) -> None:
        ts = datetime(2026, 2, 21, 14, 30)
        item_id = EpisodicMemory._episode_item_id(ts, "!!!")
        assert item_id == "episode:2026-02-21:14:30:untitled"

    def test_append_without_provenance_does_not_tag(self, ep: EpisodicMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            ep.append_entry(
                "Probe",
                "content",
                timestamp=datetime(2026, 2, 21, 14, 30),
            )
            assert len(isolated) == 0
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_append_with_provenance_tags_ledger(self, ep: EpisodicMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            ts = datetime(2026, 2, 21, 14, 30)
            ep.append_entry(
                "User says hello",
                "content",
                timestamp=ts,
                provenance_source_type="chat_utterance",
                provenance_source_id="msg-7",
                provenance_notes="from telegram",
            )
            tag = isolated.current("episode:2026-02-21:14:30:user-says-hello")
            assert tag is not None
            assert tag.source_type == SourceType.CHAT_UTTERANCE
            assert tag.source_id == "msg-7"
            assert tag.notes == "from telegram"
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_unknown_source_type_falls_back(self, ep: EpisodicMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            ts = datetime(2026, 2, 21, 14, 30)
            ep.append_entry(
                "Probe",
                "content",
                timestamp=ts,
                provenance_source_type="not_real",
                provenance_source_id="x",
            )
            tag = isolated.current("episode:2026-02-21:14:30:probe")
            assert tag is not None
            assert tag.source_type == SourceType.UNKNOWN
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_partial_provenance_args_skip_tag(self, ep: EpisodicMemory) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            ts = datetime(2026, 2, 21, 14, 30)
            ep.append_entry(
                "Only-Type",
                "c",
                timestamp=ts,
                provenance_source_type="chat_utterance",
            )
            ep.append_entry(
                "Only-Id",
                "c",
                timestamp=ts,
                provenance_source_id="msg-7",
            )
            assert len(isolated) == 0
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    def test_empty_source_id_does_not_break_append(self, ep: EpisodicMemory) -> None:
        # ProvenanceTag construction rejects empty source_id; the
        # episodic helper must swallow that ValueError so log-write
        # still succeeds.
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            # source_id is "" → fails the both-required check, no tag.
            result = ep.append_entry(
                "Probe",
                "c",
                timestamp=datetime(2026, 2, 21, 14, 30),
                provenance_source_type="chat_utterance",
                provenance_source_id="",
            )
            assert "Probe" in result
            assert len(isolated) == 0
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]
