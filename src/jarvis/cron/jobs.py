"""Cron-Job-Verwaltung: Laden, Speichern, Validieren.

Lädt Job-Definitionen aus einer YAML-Datei und stellt sie
als typisierte CronJob-Modelle bereit.

Bibel-Referenz: §10.1 (Job-Typen)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from jarvis.models import CronJob

logger = logging.getLogger(__name__)

# Default-Jobs gemäß Architektur-Bibel §10.1
DEFAULT_JOBS: list[dict[str, Any]] = [
    {
        "name": "morning_briefing",
        "schedule": "0 7 * * 1-5",
        "prompt": (
            "Erstelle mein Morning Briefing:\n"
            "1. Heutige Termine\n"
            "2. Ungelesene E-Mails (Zusammenfassung)\n"
            "3. Offene Aufgaben aus gestern\n"
            "4. Wetter für Nürnberg"
        ),
        "channel": "telegram",
        "model": "qwen3:8b",
        "enabled": False,  # Disabled by default bis Telegram konfiguriert
    },
    {
        "name": "weekly_review",
        "schedule": "0 18 * * 5",
        "prompt": (
            "Wochenrückblick:\n"
            "- Was wurde diese Woche erledigt? (Episodic Memory)\n"
            "- Welche neuen Prozeduren wurden gelernt?\n"
            "- Was ist noch offen?"
        ),
        "channel": "telegram",
        "model": "qwen3:32b",
        "enabled": False,
    },
    {
        "name": "memory_maintenance",
        "schedule": "0 3 1 * *",
        "prompt": (
            "Memory-Wartung:\n"
            "- Alte Sessions archivieren (>30 Tage)\n"
            "- Embedding-Cache aufräumen\n"
            "- Entitäten mit confidence < 0.3 markieren"
        ),
        "channel": "none",
        "model": "qwen3:8b",
        "enabled": False,
    },
]


async def governance_analysis(gateway: Any) -> None:
    """Taeglich: Governance-Agent analysiert und erzeugt Vorschlaege."""
    gov = getattr(gateway, "_governance_agent", None)
    if gov is None:
        return
    try:
        proposals = gov.analyze()
        if proposals:
            logger.info("governance_proposals_created: %d", len(proposals))
    except Exception as exc:
        logger.warning("governance_analysis_failed: %s", exc)


class JobStore:
    """Lädt und verwaltet CronJob-Definitionen aus einer YAML-Datei.

    Attributes:
        path: Pfad zur jobs.yaml Datei.
        jobs: Aktuell geladene Jobs (nach Name indexiert).
    """

    def __init__(self, path: Path | str) -> None:
        """Initialisiert den JobStore.

        Args:
            path: Pfad zur YAML-Datei mit Job-Definitionen.
        """
        self.path = Path(path)
        self.jobs: dict[str, CronJob] = {}

    def load(self) -> dict[str, CronJob]:
        """Lädt Jobs aus der YAML-Datei.

        Erstellt die Datei mit Default-Jobs falls sie nicht existiert.

        Returns:
            Dictionary: Job-Name → CronJob
        """
        if not self.path.exists():
            logger.info("Keine jobs.yaml gefunden, erstelle Defaults: %s", self.path)
            self._write_defaults()

        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as exc:
            logger.error("Fehler beim Laden von %s: %s", self.path, exc)
            raw = {}

        job_list = raw.get("jobs", {})
        self.jobs = {}

        if isinstance(job_list, dict):
            # Format: jobs: { name: {schedule:..., prompt:...} }
            for name, definition in job_list.items():
                if not isinstance(definition, dict):
                    continue
                try:
                    self.jobs[name] = CronJob(name=name, **definition)
                except Exception as exc:
                    logger.warning("Ungültiger Job '%s': %s", name, exc)
        elif isinstance(job_list, list):
            # Format: jobs: [{name:..., schedule:..., prompt:...}]
            for definition in job_list:
                if not isinstance(definition, dict) or "name" not in definition:
                    continue
                try:
                    self.jobs[definition["name"]] = CronJob(**definition)
                except Exception as exc:
                    logger.warning("Ungültiger Job '%s': %s", definition.get("name"), exc)

        logger.info("Geladene Cron-Jobs: %d", len(self.jobs))
        return self.jobs

    def get_enabled(self) -> list[CronJob]:
        """Gibt nur aktivierte Jobs zurück.

        Returns:
            Liste der aktivierten CronJobs.
        """
        return [job for job in self.jobs.values() if job.enabled]

    def add_job(self, job: CronJob) -> None:
        """Fügt einen Job hinzu und speichert die Datei.

        Args:
            job: Der hinzuzufügende CronJob.
        """
        self.jobs[job.name] = job
        self._save()

    def remove_job(self, name: str) -> bool:
        """Entfernt einen Job und speichert die Datei.

        Args:
            name: Name des zu entfernenden Jobs.

        Returns:
            True wenn der Job existierte und entfernt wurde.
        """
        if name in self.jobs:
            del self.jobs[name]
            self._save()
            return True
        return False

    def toggle_job(self, name: str, enabled: bool) -> bool:
        """Aktiviert/deaktiviert einen Job.

        Args:
            name: Job-Name.
            enabled: Neuer Status.

        Returns:
            True wenn der Job existiert.
        """
        if name not in self.jobs:
            return False
        old = self.jobs[name]
        self.jobs[name] = old.model_copy(update={"enabled": enabled})
        self._save()
        return True

    def _save(self) -> None:
        """Schreibt aktuelle Jobs in die YAML-Datei."""
        data: dict[str, dict[str, Any]] = {}
        for name, job in self.jobs.items():
            entry: dict[str, Any] = {
                "schedule": job.schedule,
                "prompt": job.prompt,
                "channel": job.channel,
                "model": job.model,
                "enabled": job.enabled,
            }
            if job.agent:
                entry["agent"] = job.agent
            data[name] = entry

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.dump({"jobs": data}, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _write_defaults(self) -> None:
        """Schreibt die Default-Jobs in die Datei."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, dict[str, Any]] = {}
        for job_def in DEFAULT_JOBS:
            name = job_def["name"]
            data[name] = {k: v for k, v in job_def.items() if k != "name"}

        self.path.write_text(
            yaml.dump({"jobs": data}, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
