"""Episodic Memory · Tier 2 — Tageslog. [B§4.3]

Was ist wann passiert? Zeitlich geordnete Einträge.
Append-only: Einträge werden nie geändert, nur hinzugefügt.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


class EpisodicMemory:
    """Verwaltet Tageslog-Dateien unter ~/.jarvis/memory/episodes/.

    Format: episodes/YYYY-MM-DD.md
    Einträge: ## HH:MM · Thema
    """

    def __init__(self, episodes_dir: str | Path) -> None:
        """Initialisiert EpisodicMemory mit dem Episoden-Verzeichnis."""
        self._dir = Path(episodes_dir)

    @property
    def directory(self) -> Path:
        """Gibt das Episoden-Verzeichnis zurück."""
        return self._dir

    def _file_for_date(self, d: date) -> Path:
        """Gibt den Pfad zur Tageslog-Datei zurück."""
        return self._dir / f"{d.isoformat()}.md"

    def ensure_directory(self) -> None:
        """Erstellt das Episodes-Verzeichnis wenn nötig."""
        self._dir.mkdir(parents=True, exist_ok=True)

    def append_entry(
        self,
        topic: str,
        content: str,
        *,
        timestamp: datetime | None = None,
    ) -> str:
        """Fügt einen Eintrag zum Tageslog hinzu. Append-only.

        Args:
            topic: Kurzer Titel des Eintrags.
            content: Detailtext (kann mehrzeilig sein).
            timestamp: Zeitpunkt (default: jetzt).

        Returns:
            Der geschriebene Eintrag als String.
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.ensure_directory()

        file_path = self._file_for_date(timestamp.date())
        time_str = timestamp.strftime("%H:%M")

        entry = f"\n## {time_str} · {topic}\n{content}\n"

        # Datei erstellen falls nicht vorhanden (mit Tages-Header)
        if not file_path.exists():
            header = f"# {timestamp.date().isoformat()}\n"
            file_path.write_text(header + entry, encoding="utf-8")
        else:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(entry)

        return entry.strip()

    def get_today(self) -> str:
        """Gibt den heutigen Tageslog zurück."""
        return self.get_date(date.today())

    def get_date(self, d: date) -> str:
        """Gibt den Tageslog für ein bestimmtes Datum zurück.

        Args:
            d: Das gewünschte Datum.

        Returns:
            Dateiinhalt oder leerer String.
        """
        file_path = self._file_for_date(d)
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8")

    def get_recent(self, days: int = 2) -> list[tuple[date, str]]:
        """Gibt die letzten N Tage zurück.

        Args:
            days: Anzahl Tage (default: 2 = heute + gestern).

        Returns:
            Liste von (datum, inhalt) Tupeln, neueste zuerst.
        """
        from datetime import timedelta

        results: list[tuple[date, str]] = []
        today = date.today()

        for i in range(days):
            d = today - timedelta(days=i)
            content = self.get_date(d)
            if content:
                results.append((d, content))

        return results

    def list_dates(self) -> list[date]:
        """Listet alle verfügbaren Tageslog-Daten.

        Returns:
            Sortierte Liste von Daten (neueste zuerst).
        """
        if not self._dir.exists():
            return []

        dates: list[date] = []
        for f in self._dir.glob("????-??-??.md"):
            try:
                d = date.fromisoformat(f.stem)
                dates.append(d)
            except ValueError:
                continue

        return sorted(dates, reverse=True)

    # ------------------------------------------------------------------
    # Retention / Pruning
    #
    # Um eine unkontrollierte Ansammlung alter Episoden zu verhindern,
    # kann die Anzahl der gespeicherten Tageslogs zeitlich begrenzt werden.
    # Der MemoryManager ruft diese Methode beim Initialisieren auf.
    # Alte Dateien werden gelöscht, wenn sie älter als ``retention_days`` sind.
    def prune_old(self, retention_days: int) -> int:
        """Löscht Episoden-Dateien, die älter als ``retention_days`` sind.

        Args:
            retention_days: Maximales Alter in Tagen. Dateien, die älter
                sind, werden entfernt. Wenn ``retention_days`` <= 0,
                passiert nichts.

        Returns:
            Anzahl der gelöschten Dateien.
        """
        from datetime import date, timedelta

        if retention_days <= 0:
            return 0
        if not self._dir.exists():
            return 0
        deleted = 0
        today = date.today()
        threshold = today - timedelta(days=retention_days)
        for f in self._dir.glob("????-??-??.md"):
            try:
                d = date.fromisoformat(f.stem)
            except ValueError:
                continue
            if d < threshold:
                try:
                    f.unlink()
                    deleted += 1
                except OSError:
                    pass
        return deleted
