"""Tests für JobStore – Laden, Speichern, Verwalten von Cron-Jobs.

Testet YAML-Persistenz, Default-Erzeugung, und Runtime-Operationen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from jarvis.cron.jobs import DEFAULT_JOBS, JobStore
from jarvis.models import CronJob

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def jobs_yaml(tmp_path: Path) -> Path:
    """Pfad für eine temporäre jobs.yaml."""
    return tmp_path / "cron" / "jobs.yaml"


@pytest.fixture()
def store(jobs_yaml: Path) -> JobStore:
    """Frischer JobStore mit tmp-Pfad."""
    return JobStore(jobs_yaml)


# ── Laden ──────────────────────────────────────────────────────────────────


class TestJobStoreLoad:
    """Tests für das Laden von Jobs aus YAML."""

    def test_creates_defaults_if_no_file(self, store: JobStore, jobs_yaml: Path) -> None:
        """Erzeugt Default-Datei wenn nichts existiert."""
        assert not jobs_yaml.exists()
        store.load()
        assert jobs_yaml.exists()
        assert len(store.jobs) == len(DEFAULT_JOBS)

    def test_loads_dict_format(self, store: JobStore, jobs_yaml: Path) -> None:
        """Lädt Jobs im Dict-Format (name → definition)."""
        jobs_yaml.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": {
                "test_job": {
                    "schedule": "0 8 * * *",
                    "prompt": "Test",
                    "channel": "cli",
                    "model": "qwen3:8b",
                    "enabled": True,
                }
            }
        }
        jobs_yaml.write_text(yaml.dump(data), encoding="utf-8")
        store.load()
        assert "test_job" in store.jobs
        assert store.jobs["test_job"].schedule == "0 8 * * *"

    def test_loads_list_format(self, store: JobStore, jobs_yaml: Path) -> None:
        """Lädt Jobs im List-Format ([{name:..., schedule:...}])."""
        jobs_yaml.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": [
                {
                    "name": "list_job",
                    "schedule": "0 9 * * *",
                    "prompt": "List test",
                    "channel": "telegram",
                }
            ]
        }
        jobs_yaml.write_text(yaml.dump(data), encoding="utf-8")
        store.load()
        assert "list_job" in store.jobs

    def test_skips_invalid_jobs(self, store: JobStore, jobs_yaml: Path) -> None:
        """Ungültige Job-Definitionen werden übersprungen."""
        jobs_yaml.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": {
                "good": {"schedule": "0 8 * * *", "prompt": "OK"},
                "bad": "not a dict",
            }
        }
        jobs_yaml.write_text(yaml.dump(data), encoding="utf-8")
        store.load()
        assert "good" in store.jobs
        assert "bad" not in store.jobs

    def test_handles_empty_yaml(self, store: JobStore, jobs_yaml: Path) -> None:
        """Leere YAML-Datei → leere Jobs."""
        jobs_yaml.parent.mkdir(parents=True, exist_ok=True)
        jobs_yaml.write_text("", encoding="utf-8")
        store.load()
        assert len(store.jobs) == 0

    def test_handles_corrupt_yaml(self, store: JobStore, jobs_yaml: Path) -> None:
        """Kaputte YAML → leere Jobs, kein Crash."""
        jobs_yaml.parent.mkdir(parents=True, exist_ok=True)
        jobs_yaml.write_text("{{{invalid yaml::", encoding="utf-8")
        store.load()
        assert len(store.jobs) == 0


# ── Get Enabled ────────────────────────────────────────────────────────────


class TestGetEnabled:
    """Tests für Filterung nach enabled-Status."""

    def test_returns_only_enabled(self, store: JobStore) -> None:
        store.jobs = {
            "on": CronJob(name="on", schedule="* * * * *", prompt="a", enabled=True),
            "off": CronJob(name="off", schedule="* * * * *", prompt="b", enabled=False),
        }
        enabled = store.get_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "on"

    def test_empty_when_all_disabled(self, store: JobStore) -> None:
        store.jobs = {
            "a": CronJob(name="a", schedule="* * * * *", prompt="x", enabled=False),
        }
        assert store.get_enabled() == []


# ── Add / Remove / Toggle ─────────────────────────────────────────────────


class TestJobStoreMutations:
    """Tests für Hinzufügen, Entfernen, Togglen von Jobs."""

    def test_add_job_persists(self, store: JobStore, jobs_yaml: Path) -> None:
        """add_job speichert in Datei."""
        job = CronJob(name="new", schedule="0 12 * * *", prompt="Mittag")
        store.add_job(job)
        assert "new" in store.jobs
        assert jobs_yaml.exists()
        # Reload und prüfen
        store2 = JobStore(jobs_yaml)
        store2.load()
        assert "new" in store2.jobs

    def test_remove_job_persists(self, store: JobStore) -> None:
        """remove_job entfernt und speichert."""
        store.jobs["tmp"] = CronJob(name="tmp", schedule="* * * * *", prompt="x")
        assert store.remove_job("tmp") is True
        assert "tmp" not in store.jobs

    def test_remove_nonexistent_returns_false(self, store: JobStore) -> None:
        assert store.remove_job("nope") is False

    def test_toggle_job(self, store: JobStore) -> None:
        store.jobs["j"] = CronJob(name="j", schedule="* * * * *", prompt="x", enabled=True)
        assert store.toggle_job("j", False) is True
        assert store.jobs["j"].enabled is False

    def test_toggle_nonexistent_returns_false(self, store: JobStore) -> None:
        assert store.toggle_job("nope", True) is False


# ── Defaults ───────────────────────────────────────────────────────────────


class TestDefaults:
    """Tests für Default-Job-Definitionen."""

    def test_defaults_have_required_fields(self) -> None:
        """Alle Default-Jobs müssen gültige CronJobs erzeugen."""
        for job_def in DEFAULT_JOBS:
            job = CronJob(**job_def)
            assert job.name
            assert job.schedule
            assert job.prompt

    def test_defaults_are_disabled(self) -> None:
        """Default-Jobs sind initial deaktiviert."""
        for job_def in DEFAULT_JOBS:
            assert job_def.get("enabled") is False

    def test_morning_briefing_weekdays_only(self) -> None:
        """Morning Briefing nur Mo-Fr."""
        morning = next(j for j in DEFAULT_JOBS if j["name"] == "morning_briefing")
        assert "1-5" in morning["schedule"]
