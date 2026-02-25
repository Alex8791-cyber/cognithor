"""Verwaltung für externe Skills (Prozeduren).

Skills werden als Markdown-Dateien mit Frontmatter definiert und im
``skills``-Verzeichnis innerhalb des Jarvis-Home gespeichert. Dieser
Manager bietet Funktionen zum Auflisten vorhandener Skills und zum
Erstellen neuer Vorlagen. Ein automatisches Installieren aus Remote-
Quellen kann später ergänzt werden.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


def list_skills(skills_dir: Path) -> list[str]:
    """Listet alle verfügbaren Skills.

    Args:
        skills_dir: Verzeichnis in dem die Skill-Dateien liegen.

    Returns:
        Liste der Skill-Dateinamen (ohne Pfad).
    """
    if not skills_dir.exists():
        return []
    return [p.name for p in skills_dir.glob("*.md") if p.is_file()]


def _slugify(name: str) -> str:
    """Erstellt einen Dateinamen-Slug aus einem beliebigen Namen.

    Konvertiert zu Kleinbuchstaben, ersetzt Leerzeichen durch Bindestriche
    und entfernt alle Zeichen außer Buchstaben, Zahlen und Bindestrichen.
    """
    slug = name.lower().strip()
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    return slug


def create_skill(skills_dir: Path, name: str, trigger_keywords: Optional[Iterable[str]] = None) -> Path:
    """Erstellt eine neue Skill-Datei mit einer minimalen Vorlage.

    Args:
        skills_dir: Speicherort der Skills.
        name: Klarname des neuen Skills. Wird in den Frontmatter-Titel
            übernommen und als Dateiname gesluggified.
        trigger_keywords: Schlüsselwörter, die diesen Skill auslösen.

    Returns:
        Pfad zur neu angelegten Skill-Datei.
    """
    if trigger_keywords is None:
        trigger_keywords = []
    slug = _slugify(name)
    filename = f"{slug}.md"
    path = skills_dir / filename
    if path.exists():
        raise FileExistsError(f"Skill '{name}' existiert bereits: {path}")
    # Frontmatter für eine Prozedur
    triggers = ", ".join(trigger_keywords)
    content = (
        "---\n"
        f"name: {name}\n"
        f"trigger_keywords: [{triggers}]\n"
        "---\n"
        "# " + name + "\n\n"
        "## Voraussetzungen\n\n"
        "Beschreibe hier die Voraussetzungen für diesen Skill.\n\n"
        "## Schritte\n\n"
        "1. Detaillierte Schritt-für-Schritt-Anleitung.\n\n"
        "## Hinweise\n\n"
        "Notiere bekannte Fehlerfälle oder Tipps.\n"
    )
    skills_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def search_remote_skills(query: str, limit: int = 10) -> list[str]:
    """Durchsucht lokale "Remote"-Skill-Repos nach passenden Skills.

    In Abwesenheit eines echten Marktplatzes durchsucht diese Funktion
    die bereitgestellten Beispiel-Prozeduren im Repository, um ähnlich
    wie eine Remote-Suche zu funktionieren. Es werden sowohl die Namen
    der Dateien als auch deren Inhaltsfrontmatter betrachtet. Das
    Ergebnis ist eine Liste von Skill-Dateinamen (ohne Erweiterung),
    sortiert nach einfacher Übereinstimmung.

    Args:
        query: Suchbegriff für Skills (Groß-/Kleinschreibung wird ignoriert).
        limit: Maximale Anzahl an Ergebnissen.

    Returns:
        Liste der Skill-Basenamen (ohne ``.md``), die zur Suchanfrage passen.
    """
    query_lower = query.lower().strip()
    results: list[str] = []

    # Bestimme potenzielle "Remote"-Verzeichnisse relativ zu diesem Modul
    here = Path(__file__).resolve()
    # Die Struktur ist: project/src/jarvis/skills/manager.py → parents[4] = project
    # Wir berücksichtigen zwei Orte als Quelle für "Remote"-Skills:
    #  1. <repo_root>/project/data/procedures
    #  2. <repo_root>/data/procedures
    # parents[4] -> <repo_root>/project, parents[5] -> <repo_root>
    project_dir = here.parents[4]
    repo_root = here.parents[5]
    candidate_dirs = [
        project_dir / "data" / "procedures",
        repo_root / "data" / "procedures",
    ]

    seen: set[str] = set()
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for file_path in directory.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            name = file_path.stem
            # Untersuche Frontmatter (erste ca. 20 Zeilen) nach name und trigger_keywords
            fm_name: str | None = None
            fm_triggers: list[str] = []
            try:
                lines = content.splitlines()
                for line in lines[:20]:
                    # Name-Zeile
                    if line.lower().startswith("name:"):
                        fm_name = line.split("name:", 1)[1].strip()
                    elif line.lower().startswith("trigger_keywords"):
                        # Extrahiere Liste zwischen eckigen Klammern
                        after = line.split("[", 1)
                        if len(after) > 1:
                            inside = after[1].split("]", 1)[0]
                            # Spalten nach Kommas
                            for kw in inside.split(","):
                                kw = kw.strip().strip("'\"")
                                if kw:
                                    fm_triggers.append(kw)
                    # Stoppe, wenn Frontmatter endet (erster Abschnitt nach "---")
                    if line.strip() == "---":
                        break
            except Exception:
                pass

            # Prüfe, ob Query im Dateinamen, Frontmatter-Name, Triggern oder Inhalt vorkommt
            match_found = False
            if query_lower in name.lower():
                match_found = True
            elif fm_name and query_lower in fm_name.lower():
                match_found = True
            elif any(query_lower in kw.lower() for kw in fm_triggers):
                match_found = True
            elif query_lower in content.lower():
                match_found = True

            if match_found and name not in seen:
                results.append(name)
                seen.add(name)
                if len(results) >= limit:
                    return results
    return results


def install_remote_skill(skills_dir: Path, name: str, repo_url: str | None = None) -> Path:
    """Installiert einen Skill aus einem lokalen "Remote"-Repository.

    Diese Funktion versucht, eine vorhandene Prozedur (Skill) aus den
    Beispiel-Prozeduren des Repositories zu kopieren und unter dem
    angegebenen Namen im Plugins-Verzeichnis abzulegen. Wenn der Skill
    bereits installiert ist, wird der bestehende Pfad zurückgegeben.
    Falls kein passender Skill gefunden wird, wird eine leere
    Vorlage erstellt.

    Args:
        skills_dir: Zielverzeichnis für den Skill.
        name: Name des zu installierenden Skills. Kann sowohl der
            Dateiname ohne Erweiterung als auch der sichtbare
            Frontmatter-Name sein.
        repo_url: Optionaler Verweis auf ein Remote-Repository (wird in
            dieser Offline-Variante ignoriert).

    Returns:
        Pfad zur installierten oder erstellten Skill-Datei.
    """
    # Normalisiere den Dateinamen
    slug = _slugify(name)
    target_filename = f"{slug}.md"
    target_path = skills_dir / target_filename

    # Falls bereits installiert, gib den Pfad zurück
    if target_path.exists():
        return target_path

    # Bestimme "Remote"-Quellverzeichnisse
    here = Path(__file__).resolve()
    # parents[4] -> <repo_root>/project, parents[5] -> <repo_root>
    project_dir = here.parents[4]
    repo_root = here.parents[5]
    source_dirs = [
        project_dir / "data" / "procedures",
        repo_root / "data" / "procedures",
    ]

    # Suche nach einer passenden Quelldatei
    source_file: Optional[Path] = None
    for directory in source_dirs:
        if not directory.exists():
            continue
        for file_path in directory.glob("*.md"):
            if file_path.stem.lower() == slug:
                source_file = file_path
                break
        if source_file:
            break

    # Wenn gefunden, kopiere den Inhalt in das Ziel
    if source_file is not None and source_file.exists():
        try:
            content = source_file.read_text(encoding="utf-8")
        except Exception:
            content = ""
        skills_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    # Andernfalls erstelle eine leere Vorlage wie zuvor
    content = (
        f"name: {name}\n"
        f"trigger_keywords: []\n"
        "---\n"
        f"# {name}\n\n"
        "## Beschreibung\n\n"
        "Dieser Skill wurde automatisch erstellt. Er muss manuell mit Inhalt befüllt werden.\n"
    )
    skills_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return target_path