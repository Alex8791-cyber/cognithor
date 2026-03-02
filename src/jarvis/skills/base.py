"""Basisklasse fuer alle Jarvis-Skills.

Jeder Skill erbt von ``BaseSkill`` und implementiert ``execute()``.
Der SkillScaffolder (``jarvis.tools.skill_cli``) generiert automatisch
Code, der ``BaseSkill`` importiert und als Elternklasse verwendet.

Beispiel:
    class WetterSkill(BaseSkill):
        NAME = "wetter_abfrage"
        DESCRIPTION = "Aktuelle Wetterdaten abrufen"
        VERSION = "0.1.0"
        REQUIRES_NETWORK = True

        async def execute(self, params: dict) -> dict:
            city = params.get("city", "Berlin")
            ...
            return {"status": "ok", "result": data}

Bibel-Referenz: §6.2 (Prozedurale Skills), §4.6 (Working Memory Injection)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SkillError(Exception):
    """Fehler bei der Skill-Ausfuehrung."""


class BaseSkill(ABC):
    """Abstrakte Basisklasse fuer alle Jarvis-Skills.

    Klassen-Attribute:
        NAME:              Eindeutiger Skill-Bezeichner (slug).
        DESCRIPTION:       Kurzbeschreibung des Skills.
        VERSION:           Semantische Version (z.B. ``0.1.0``).
        REQUIRES_NETWORK:  ``True`` wenn der Skill Netzwerkzugriff benoetigt.
        API_BASE:          Basis-URL fuer API-Skills (optional).
        CRON:              Cron-Ausdruck fuer automatisierte Skills (optional).
    """

    NAME: str = ""
    DESCRIPTION: str = ""
    VERSION: str = "0.1.0"
    REQUIRES_NETWORK: bool = False
    API_BASE: str = ""
    CRON: str = ""

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fuehrt den Skill aus.

        Args:
            params: Parameter-Dictionary aus dem Planner/Executor.

        Returns:
            Ergebnis-Dictionary mit mindestens ``status`` (``ok`` oder ``error``).

        Raises:
            SkillError: Bei Ausfuehrungsfehlern.
        """

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    @property
    def version(self) -> str:
        return self.VERSION

    @property
    def is_automated(self) -> bool:
        """True wenn der Skill einen Cron-Schedule hat."""
        return bool(self.CRON)

    @property
    def is_network_skill(self) -> bool:
        """True wenn der Skill Netzwerkzugriff benoetigt."""
        return self.REQUIRES_NETWORK

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validiert Parameter. Ueberschreibbar fuer spezifische Pruefungen.

        Returns:
            Liste von Fehlermeldungen (leer = OK).
        """
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.NAME!r} v{self.VERSION}>"
