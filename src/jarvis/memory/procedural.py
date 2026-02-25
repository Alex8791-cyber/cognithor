"""Procedural Memory · Tier 4 -- Gelernte Skills/Routinen. [B§4.5, B§6]

Inspiration: Voyager Skill Library, SAGE Framework.
Jede Prozedur ist eine Markdown-Datei mit YAML-Frontmatter.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from jarvis.models import ProcedureMetadata

logger = logging.getLogger("jarvis.memory.procedural")

# YAML Frontmatter Pattern
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class ProceduralMemory:
    """Verwaltet gelernte Prozeduren unter ~/.jarvis/memory/procedures/.

    Dateiformat:
    ```
    ---
    name: morgen-briefing
    trigger_keywords: [Morgen, Briefing, Tagesstart]
    tools_required: [search_memory, write_file]
    success_count: 5
    failure_count: 1
    total_uses: 6
    avg_score: 0.85
    last_used: 2026-02-20T15:30:00
    learned_from: [session-123, session-456]
    ---
    # Morgen-Briefing erstellen

    ## Ablauf
    1. Episoden laden
    2. Zusammenfassen
    ...
    ```
    """

    def __init__(self, procedures_dir: str | Path) -> None:
        """Initialisiert ProceduralMemory mit dem Prozeduren-Verzeichnis."""
        self._dir = Path(procedures_dir)

    @property
    def directory(self) -> Path:
        """Gibt das Prozeduren-Verzeichnis zurück."""
        return self._dir

    def ensure_directory(self) -> None:
        """Erstellt das Procedures-Verzeichnis."""
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ─────────────────────────────────────────────────────

    def save_procedure(
        self,
        name: str,
        body: str,
        metadata: ProcedureMetadata | None = None,
    ) -> Path:
        """Speichert eine Prozedur als Markdown-Datei.

        Args:
            name: Prozedur-Name (wird als Dateiname verwendet).
            body: Markdown-Body (ohne Frontmatter).
            metadata: Prozedur-Metadaten.

        Returns:
            Pfad zur erstellten Datei.
        """
        self.ensure_directory()

        if metadata is None:
            metadata = ProcedureMetadata(name=name)

        # Sanitize filename
        safe_name = re.sub(r"[^\w\-]", "-", name.lower()).strip("-")
        file_path = self._dir / f"{safe_name}.md"

        frontmatter = self._metadata_to_dict(metadata)
        front = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        content = f"---\n{front}---\n{body}"

        file_path.write_text(content, encoding="utf-8")
        logger.info("Prozedur gespeichert: %s → %s", name, file_path)
        return file_path

    def load_procedure(self, name: str) -> tuple[ProcedureMetadata, str] | None:
        """Lädt eine Prozedur.

        Args:
            name: Prozedur-Name.

        Returns:
            (metadata, body) Tupel oder None wenn nicht gefunden.
        """
        safe_name = re.sub(r"[^\w\-]", "-", name.lower()).strip("-")
        file_path = self._dir / f"{safe_name}.md"

        if not file_path.exists():
            return None

        return self._parse_file(file_path)

    def list_procedures(self) -> list[ProcedureMetadata]:
        """Listet alle Prozedur-Metadaten."""
        if not self._dir.exists():
            return []

        results: list[ProcedureMetadata] = []
        for f in sorted(self._dir.glob("*.md")):
            parsed = self._parse_file(f)
            if parsed:
                results.append(parsed[0])

        return results

    def find_by_keywords(self, keywords: list[str]) -> list[tuple[ProcedureMetadata, str, float]]:
        """Sucht Prozeduren anhand von Keywords.

        Args:
            keywords: Suchbegriffe.

        Returns:
            Liste von (metadata, body, relevance_score), sortiert nach Relevanz.
        """
        if not self._dir.exists():
            return []

        keywords_lower = [k.lower() for k in keywords]
        results: list[tuple[ProcedureMetadata, str, float]] = []

        for f in self._dir.glob("*.md"):
            parsed = self._parse_file(f)
            if not parsed:
                continue

            meta, body = parsed

            # Score berechnen: Keyword-Overlap
            trigger_lower = [t.lower() for t in meta.trigger_keywords]
            name_lower = meta.name.lower()
            body_lower = body.lower()

            score = 0.0
            for kw in keywords_lower:
                if kw in trigger_lower:
                    score += 3.0  # Direkter Trigger-Match
                elif kw in name_lower:
                    score += 2.0  # Name-Match
                elif kw in body_lower:
                    score += 1.0  # Body-Match

            if score > 0:
                # Bonus für Zuverlässigkeit
                if meta.is_reliable:
                    score *= 1.2
                # Malus für Review-Bedarf
                if meta.needs_review:
                    score *= 0.5

                results.append((meta, body, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def find_by_query(
        self, query: str, max_results: int = 2
    ) -> list[tuple[ProcedureMetadata, str, float]]:
        """Sucht Prozeduren passend zu einer natürlichen User-Query. [B§6.3]

        Extrahiert automatisch Schlüsselwörter aus der Query und
        durchsucht die Prozeduren. Für den Planner-Kontext.

        Args:
            query: Natürliche Benutzer-Anfrage.
            max_results: Maximale Anzahl Ergebnisse.

        Returns:
            Beste Matches als (metadata, body, score).
        """
        # Einfache Tokenisierung: Stoppwörter rausfiltern, Rest als Keywords
        words = query.lower().split()
        # Deutsche Stoppwörter (kompakt)
        stopwords = {
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "die",
            "der",
            "das",
            "ein",
            "eine",
            "einen",
            "einem",
            "einer",
            "und",
            "oder",
            "aber",
            "wenn",
            "weil",
            "dass",
            "was",
            "wie",
            "wo",
            "wer",
            "ist",
            "sind",
            "hat",
            "haben",
            "kann",
            "können",
            "will",
            "wollen",
            "soll",
            "muss",
            "für",
            "mit",
            "von",
            "zu",
            "auf",
            "an",
            "in",
            "aus",
            "bei",
            "nach",
            "über",
            "unter",
            "vor",
            "hinter",
            "zwischen",
            "bitte",
            "mir",
            "mich",
            "dich",
            "dir",
            "uns",
            "euch",
            "mal",
            "doch",
            "noch",
            "schon",
            "auch",
            "nur",
            "nicht",
            "den",
            "dem",
            "des",
            "im",
            "am",
            "zum",
            "zur",
            "mein",
            "dein",
            "sein",
            "unser",
            "euer",
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        if not keywords:
            return []

        results = self.find_by_keywords(keywords)
        return results[:max_results]

    def record_usage(
        self,
        name: str,
        success: bool,
        score: float = 0.0,
        session_id: str = "",
    ) -> ProcedureMetadata | None:
        """Zeichnet eine Nutzung auf und aktualisiert Statistiken.

        Args:
            name: Prozedur-Name.
            success: War die Nutzung erfolgreich?
            score: Bewertungs-Score (0-1).
            session_id: Session in der die Nutzung stattfand.

        Returns:
            Aktualisierte Metadaten oder None.
        """
        result = self.load_procedure(name)
        if result is None:
            return None

        meta, body = result

        meta.total_uses += 1
        if success:
            meta.success_count += 1
        else:
            meta.failure_count += 1

        # Gleitender Durchschnitt
        if meta.total_uses == 1:
            meta.avg_score = score
        else:
            meta.avg_score = (meta.avg_score * (meta.total_uses - 1) + score) / meta.total_uses

        meta.last_used = datetime.now()
        if session_id:
            meta.learned_from.append(session_id)

        self.save_procedure(name, body, meta)
        return meta

    def add_failure_pattern(self, name: str, pattern: str) -> bool:
        """Fügt ein Fehler-Muster zu einer Prozedur hinzu."""
        result = self.load_procedure(name)
        if result is None:
            return False

        meta, body = result
        if pattern not in meta.failure_patterns:
            meta.failure_patterns.append(pattern)
            self.save_procedure(name, body, meta)
        return True

    def add_improvement(self, name: str, improvement: str) -> bool:
        """Fügt eine Verbesserung zu einer Prozedur hinzu."""
        result = self.load_procedure(name)
        if result is None:
            return False

        meta, body = result
        if improvement not in meta.improvements:
            meta.improvements.append(improvement)
            self.save_procedure(name, body, meta)
        return True

    def delete_procedure(self, name: str) -> bool:
        """Löscht eine Prozedur."""
        safe_name = re.sub(r"[^\w\-]", "-", name.lower()).strip("-")
        file_path = self._dir / f"{safe_name}.md"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Statistiken über Prozeduren."""
        procs = self.list_procedures()
        return {
            "total": len(procs),
            "reliable": sum(1 for p in procs if p.is_reliable),
            "needs_review": sum(1 for p in procs if p.needs_review),
            "total_uses": sum(p.total_uses for p in procs),
            "avg_success_rate": (sum(p.success_rate for p in procs) / len(procs) if procs else 0.0),
        }

    # ── Internal ─────────────────────────────────────────────────

    def _parse_file(self, path: Path) -> tuple[ProcedureMetadata, str] | None:
        """Parst eine Prozedur-Datei in Metadata + Body."""
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Kann %s nicht lesen: %s", path, e)
            return None

        match = _FRONTMATTER_RE.match(content)
        if not match:
            # Kein Frontmatter → Nur Body
            return ProcedureMetadata(name=path.stem, source_file=str(path)), content

        try:
            front = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as e:
            logger.warning("YAML-Fehler in %s: %s", path, e)
            return ProcedureMetadata(name=path.stem, source_file=str(path)), content

        body = content[match.end() :]
        meta = self._dict_to_metadata(front, str(path))
        return meta, body

    @staticmethod
    def _metadata_to_dict(meta: ProcedureMetadata) -> dict[str, Any]:
        """Konvertiert Metadata zu YAML-kompatiblem Dict."""
        d: dict[str, Any] = {
            "name": meta.name,
            "trigger_keywords": meta.trigger_keywords,
            "tools_required": meta.tools_required,
            "success_count": meta.success_count,
            "failure_count": meta.failure_count,
            "total_uses": meta.total_uses,
            "avg_score": round(meta.avg_score, 3),
        }
        if meta.last_used:
            d["last_used"] = meta.last_used.isoformat()
        if meta.learned_from:
            d["learned_from"] = meta.learned_from
        if meta.failure_patterns:
            d["failure_patterns"] = meta.failure_patterns
        if meta.improvements:
            d["improvements"] = meta.improvements
        return d

    @staticmethod
    def _dict_to_metadata(d: dict[str, Any], source_file: str = "") -> ProcedureMetadata:
        """Konvertiert YAML-Dict zu ProcedureMetadata."""
        last_used = d.get("last_used")
        if isinstance(last_used, str):
            try:
                last_used = datetime.fromisoformat(last_used)
            except ValueError:
                last_used = None

        return ProcedureMetadata(
            name=d.get("name", "unknown"),
            trigger_keywords=d.get("trigger_keywords", []),
            tools_required=d.get("tools_required", []),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            total_uses=d.get("total_uses", 0),
            avg_score=float(d.get("avg_score", 0.0)),
            last_used=last_used,
            learned_from=d.get("learned_from", []),
            failure_patterns=d.get("failure_patterns", []),
            improvements=d.get("improvements", []),
            source_file=source_file,
        )
