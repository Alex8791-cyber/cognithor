"""Tests für memory/procedural.py · Tier 4 Skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.memory.procedural import ProceduralMemory
from jarvis.models import ProcedureMetadata

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def proc_dir(tmp_path: Path) -> Path:
    return tmp_path / "procedures"


@pytest.fixture
def proc(proc_dir: Path) -> ProceduralMemory:
    return ProceduralMemory(proc_dir)


class TestProceduralMemory:
    def test_save_and_load(self, proc: ProceduralMemory):
        meta = ProcedureMetadata(
            name="bu-angebot",
            trigger_keywords=["Recherche", "Bericht"],
            tools_required=["calculator"],
            success_count=3,
            total_uses=4,
            avg_score=0.85,
        )
        proc.save_procedure("bu-angebot", "# BU Angebot\n\nSchritt 1: ...", meta)

        result = proc.load_procedure("bu-angebot")
        assert result is not None
        loaded_meta, body = result
        assert loaded_meta.name == "bu-angebot"
        assert loaded_meta.trigger_keywords == ["Recherche", "Bericht"]
        assert loaded_meta.success_count == 3
        assert loaded_meta.avg_score == pytest.approx(0.85)
        assert "Schritt 1" in body

    def test_load_nonexistent(self, proc: ProceduralMemory):
        assert proc.load_procedure("nope") is None

    def test_list_procedures(self, proc: ProceduralMemory):
        proc.save_procedure("alpha", "Body A")
        proc.save_procedure("beta", "Body B")
        procs = proc.list_procedures()
        names = [p.name for p in procs]
        assert "alpha" in names
        assert "beta" in names

    def test_list_empty(self, proc: ProceduralMemory):
        assert proc.list_procedures() == []

    def test_find_by_keywords_trigger(self, proc: ProceduralMemory):
        meta = ProcedureMetadata(name="bu-test", trigger_keywords=["Recherche", "Bericht"])
        proc.save_procedure("bu-test", "Body", meta)

        results = proc.find_by_keywords(["BU"])
        assert len(results) == 1
        assert results[0][0].name == "bu-test"
        assert results[0][2] > 0  # Score > 0

    def test_find_by_keywords_name_match(self, proc: ProceduralMemory):
        proc.save_procedure("email-nachfassen", "Body")
        results = proc.find_by_keywords(["email"])
        assert len(results) == 1

    def test_find_by_keywords_body_match(self, proc: ProceduralMemory):
        proc.save_procedure("test-skill", "Hier geht es um Kundenberatung")
        results = proc.find_by_keywords(["Kundenberatung"])
        assert len(results) == 1

    def test_find_no_match(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        results = proc.find_by_keywords(["xyznope"])
        assert results == []

    def test_record_usage_success(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        meta = proc.record_usage("test", success=True, score=0.9, session_id="s1")
        assert meta is not None
        assert meta.success_count == 1
        assert meta.total_uses == 1
        assert meta.avg_score == pytest.approx(0.9)
        assert "s1" in meta.learned_from

    def test_record_usage_failure(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        meta = proc.record_usage("test", success=False, score=0.2)
        assert meta is not None
        assert meta.failure_count == 1

    def test_record_usage_nonexistent(self, proc: ProceduralMemory):
        assert proc.record_usage("nope", success=True, score=1.0) is None

    def test_record_multiple_usage(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        proc.record_usage("test", True, 1.0)
        proc.record_usage("test", True, 0.8)
        proc.record_usage("test", False, 0.3)

        result = proc.load_procedure("test")
        assert result is not None
        meta = result[0]
        assert meta.total_uses == 3
        assert meta.success_count == 2
        assert meta.failure_count == 1

    def test_add_failure_pattern(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        assert proc.add_failure_pattern("test", "Timeout bei API-Call")
        result = proc.load_procedure("test")
        assert result is not None
        assert "Timeout bei API-Call" in result[0].failure_patterns

    def test_add_improvement(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        assert proc.add_improvement("test", "Retry-Logic hinzufügen")
        result = proc.load_procedure("test")
        assert result is not None
        assert "Retry-Logic hinzufügen" in result[0].improvements

    def test_delete_procedure(self, proc: ProceduralMemory):
        proc.save_procedure("test", "Body")
        assert proc.delete_procedure("test")
        assert proc.load_procedure("test") is None

    def test_delete_nonexistent(self, proc: ProceduralMemory):
        assert not proc.delete_procedure("nope")

    def test_stats(self, proc: ProceduralMemory):
        s = proc.stats()
        assert s["total"] == 0

        proc.save_procedure("a", "Body")
        s = proc.stats()
        assert s["total"] == 1

    def test_filename_sanitization(self, proc: ProceduralMemory):
        proc.save_procedure("BU Angebot/Erstellen!", "Body")
        # Should create a sanitized filename
        files = list(proc.directory.glob("*.md"))
        assert len(files) == 1
        assert "/" not in files[0].name
        assert "!" not in files[0].name

    def test_reliable_procedure(self, proc: ProceduralMemory):
        meta = ProcedureMetadata(name="test", success_count=7, total_uses=9)
        proc.save_procedure("test", "Body", meta)
        procs = proc.list_procedures()
        # Not reliable yet (needs 10+ uses)
        assert not procs[0].is_reliable

        meta.success_count = 10
        meta.total_uses = 11
        proc.save_procedure("test", "Body", meta)
        procs = proc.list_procedures()
        assert procs[0].is_reliable


class TestFindByQuery:
    """Tests für die natürliche Query-Suche. [B§6.3]"""

    @pytest.fixture()
    def proc_with_data(self, tmp_path) -> ProceduralMemory:
        proc = ProceduralMemory(tmp_path / "procedures")
        proc.ensure_directory()
        proc.save_procedure(
            "bu-angebot-erstellen",
            "# Recherche-Bericht\n\n1. Kundendaten laden\n2. Tarif berechnen",
            ProcedureMetadata(
                name="bu-angebot-erstellen",
                trigger_keywords=["Recherche", "Bericht", "Zusammenfassung"],
                tools_required=["memory_search"],
            ),
        )
        proc.save_procedure(
            "kfz-versicherung-vergleich",
            "# KFZ-Vergleich\n\n1. Fahrzeugdaten\n2. Tarife vergleichen",
            ProcedureMetadata(
                name="kfz-versicherung-vergleich",
                trigger_keywords=["Server", "Cloud", "Hosting", "Vergleich"],
                tools_required=["web_search"],
            ),
        )
        return proc

    def test_finds_matching_procedure(self, proc_with_data: ProceduralMemory) -> None:
        """Natürliche Anfrage findet passende Prozedur."""
        results = proc_with_data.find_by_query("Erstelle mir bitte ein BU Angebot")
        assert len(results) >= 1
        assert results[0][0].name == "bu-angebot-erstellen"

    def test_finds_kfz_procedure(self, proc_with_data: ProceduralMemory) -> None:
        """KFZ-Anfrage findet KFZ-Prozedur."""
        results = proc_with_data.find_by_query("Ich brauche einen Server Hosting Vergleich")
        assert len(results) >= 1
        assert results[0][0].name == "kfz-versicherung-vergleich"

    def test_empty_query_returns_nothing(self, proc_with_data: ProceduralMemory) -> None:
        """Leere Query gibt keine Ergebnisse."""
        results = proc_with_data.find_by_query("")
        assert results == []

    def test_stopword_only_query(self, proc_with_data: ProceduralMemory) -> None:
        """Query nur aus Stoppwörtern gibt keine Ergebnisse."""
        results = proc_with_data.find_by_query("ich und du und sie")
        assert results == []

    def test_max_results_respected(self, proc_with_data: ProceduralMemory) -> None:
        """max_results begrenzt die Ergebnisse."""
        results = proc_with_data.find_by_query("Hosting Angebot", max_results=1)
        assert len(results) <= 1

    def test_unrelated_query_returns_empty(self, proc_with_data: ProceduralMemory) -> None:
        """Unbekannte Query findet nichts."""
        results = proc_with_data.find_by_query("Wetter in Nürnberg morgen")
        assert results == []
