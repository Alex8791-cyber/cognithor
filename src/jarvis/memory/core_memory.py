"""Core Memory · Tier 1 — Identität, Regeln, Präferenzen. [B§4.2]

Wird IMMER geladen. In jeder Session. Komplett.
Änderungen nur durch User oder expliziten Befehl.
Kein Recency-Decay.
"""

from __future__ import annotations

import re
from pathlib import Path


class CoreMemory:
    """Verwaltet die CORE.md Datei — Jarvis' Identität.

    Source of Truth: ~/.jarvis/memory/CORE.md
    """

    def __init__(self, core_file: str | Path) -> None:
        """Initialisiert CoreMemory mit dem Pfad zur CORE.md."""
        self._path = Path(core_file)
        self._content: str = ""
        self._sections: dict[str, str] = {}

    @property
    def path(self) -> Path:
        """Gibt den Pfad zur CORE.md zurück."""
        return self._path

    @property
    def content(self) -> str:
        """Kompletter CORE.md Inhalt."""
        return self._content

    @property
    def sections(self) -> dict[str, str]:
        """Geparste Sektionen als {header: content}."""
        return dict(self._sections)

    def load(self) -> str:
        """Lädt CORE.md von Disk. Erstellt Default wenn nicht vorhanden.

        Returns:
            Kompletter Inhalt als String.
        """
        if not self._path.exists():
            self._content = ""
            self._sections = {}
            return ""

        self._content = self._path.read_text(encoding="utf-8")
        self._sections = self._parse_sections(self._content)
        return self._content

    def get_section(self, name: str) -> str:
        """Gibt den Inhalt einer Sektion zurück.

        Args:
            name: Sektionsname (case-insensitive, ohne '#').

        Returns:
            Sektionsinhalt oder leerer String.
        """
        name_lower = name.lower().strip()
        for key, value in self._sections.items():
            if key.lower().strip() == name_lower:
                return value
        return ""

    def save(self, content: str | None = None) -> None:
        """Speichert CORE.md auf Disk.

        Args:
            content: Neuer Inhalt. Wenn None, wird aktueller Inhalt gespeichert.
        """
        if content is not None:
            self._content = content
            self._sections = self._parse_sections(content)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._content, encoding="utf-8")

    def create_default(self) -> str:
        """Erstellt eine Standard-CORE.md und gibt den Inhalt zurück."""
        default = (
            "# Identität\n"
            "Ich bin Jarvis, ein lokaler AI-Assistent.\n\n"
            "# Regeln\n"
            "- Kundendaten NIEMALS in Logs schreiben\n"
            "- E-Mails IMMER zur Bestätigung vorlegen\n\n"
            "# Präferenzen\n"
            "- Codesprache: Python\n"
            "- Kommunikation: Direkt, keine Floskeln\n"
            "- Zeitzone: Europe/Berlin\n"
        )
        self.save(default)
        return default

    @staticmethod
    def _parse_sections(text: str) -> dict[str, str]:
        """Parst Markdown in Sektionen anhand von H1/H2 Headers.

        Returns:
            Dict mit {header_name: content_text}.
        """
        sections: dict[str, str] = {}
        current_header: str | None = None
        current_lines: list[str] = []

        for line in text.split("\n"):
            match = re.match(r"^(#{1,2})\s+(.+)$", line)
            if match:
                # Vorherige Sektion speichern
                if current_header is not None:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Letzte Sektion
        if current_header is not None:
            sections[current_header] = "\n".join(current_lines).strip()

        return sections
