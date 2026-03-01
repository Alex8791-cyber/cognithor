"""Skill-Management-Tools fuer Jarvis.

Ermoeglicht das Erstellen und Auflisten von Skills via MCP-Tools.
Skills werden als Markdown-Dateien mit YAML-Frontmatter geschrieben
und sind sofort nach Erstellung verfuegbar (Live-Reload der Registry).

Tools:
  - create_skill: Neuen Skill erstellen (Markdown-Datei + Registry-Reload)
  - list_skills: Registrierte Skills auflisten

Bibel-Referenz: §6.2 (Skill-System)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.skills.registry import SkillRegistry

log = get_logger(__name__)

__all__ = [
    "SkillTools",
    "register_skill_tools",
]


def _slugify(name: str) -> str:
    """Erzeugt einen URL/Dateinamen-sicheren Slug aus einem Skill-Namen."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s_-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug.strip("_") or "unnamed_skill"


class SkillTools:
    """Skill-Management-Operationen fuer Jarvis. [B§6.2]"""

    def __init__(
        self,
        skill_registry: SkillRegistry,
        skills_dirs: list[Path],
    ) -> None:
        self._registry = skill_registry
        self._skills_dirs = skills_dirs
        # Primaeres Schreibverzeichnis: erstes User-Skill-Verzeichnis
        self._write_dir = skills_dirs[-1] if skills_dirs else Path("~/.jarvis/skills")
        self._write_dir = Path(self._write_dir).expanduser()

    def create_skill(
        self,
        name: str,
        description: str,
        trigger_keywords: str,
        body: str,
        category: str = "general",
        tools_required: str = "",
    ) -> str:
        """Erstellt einen neuen Skill als Markdown-Datei und laedt die Registry neu.

        Args:
            name: Skill-Name (z.B. "PDF Export")
            description: Kurzbeschreibung des Skills
            trigger_keywords: Komma-separierte Keywords (z.B. "pdf,export,dokument")
            body: Markdown-Inhalt mit Anweisungen/Prozedur
            category: Kategorie (default: "general")
            tools_required: Komma-separierte Tool-Namen (optional)

        Returns:
            Erfolgsmeldung mit Dateipfad und Skill-Slug.
        """
        if not name or not name.strip():
            return "Fehler: name darf nicht leer sein."
        if not description or not description.strip():
            return "Fehler: description darf nicht leer sein."
        if not trigger_keywords or not trigger_keywords.strip():
            return "Fehler: trigger_keywords darf nicht leer sein."
        if not body or not body.strip():
            return "Fehler: body darf nicht leer sein."

        slug = _slugify(name)

        # Keywords und Tools als YAML-Listen formatieren
        keywords = [kw.strip() for kw in trigger_keywords.split(",") if kw.strip()]
        tools = [t.strip() for t in tools_required.split(",") if t.strip()] if tools_required else []

        keywords_yaml = ", ".join(keywords)
        tools_yaml = ", ".join(tools) if tools else ""

        # Markdown-Datei mit YAML-Frontmatter erstellen
        lines = [
            "---",
            f"name: {name.strip()}",
            f"slug: {slug}",
            f"description: \"{description.strip()}\"",
            f"trigger_keywords: [{keywords_yaml}]",
            f"category: {category.strip()}",
            f"priority: 0",
            f"enabled: true",
        ]
        if tools_yaml:
            lines.append(f"tools_required: [{tools_yaml}]")
        lines.append("---")
        lines.append("")
        lines.append(body.strip())
        lines.append("")

        content = "\n".join(lines)

        # Verzeichnis sicherstellen und Datei schreiben
        self._write_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._write_dir / f"{slug}.md"

        # Nicht ueberschreiben ohne Warnung
        if file_path.exists():
            return (
                f"Fehler: Skill-Datei existiert bereits: {file_path}\n"
                f"Verwende einen anderen Namen oder loesche die bestehende Datei."
            )

        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Fehler beim Schreiben der Skill-Datei: {exc}"

        # Registry neu laden damit der Skill sofort verfuegbar ist
        try:
            skill_count = self._registry.load_from_directories(self._skills_dirs)
            log.info(
                "skill_created_and_loaded",
                slug=slug,
                file=str(file_path),
                total_skills=skill_count,
            )
        except Exception as exc:
            log.warning("skill_registry_reload_failed", error=str(exc))
            return (
                f"Skill-Datei geschrieben: {file_path}\n"
                f"WARNUNG: Registry-Reload fehlgeschlagen: {exc}\n"
                f"Der Skill wird beim naechsten Neustart verfuegbar sein."
            )

        return (
            f"Skill erfolgreich erstellt.\n"
            f"  Name: {name.strip()}\n"
            f"  Slug: {slug}\n"
            f"  Datei: {file_path}\n"
            f"  Keywords: {', '.join(keywords)}\n"
            f"  Kategorie: {category}\n"
            f"  Registrierte Skills gesamt: {skill_count}\n"
            f"Der Skill ist sofort verfuegbar."
        )

    def list_skills(
        self,
        category: str = "",
        enabled_only: bool = True,
    ) -> str:
        """Listet alle registrierten Skills auf.

        Args:
            category: Filter nach Kategorie (leer = alle)
            enabled_only: Nur aktive Skills anzeigen (default: true)

        Returns:
            Formatierte Liste aller Skills.
        """
        skills = list(self._registry._skills.values())

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        if category:
            cat_lower = category.lower().strip()
            # "all"/"alle" = kein Filter (alle Kategorien)
            if cat_lower not in ("all", "alle", ""):
                skills = [s for s in skills if s.category.lower() == cat_lower]

        if not skills:
            filter_info = ""
            if category:
                filter_info += f" (Kategorie: {category})"
            if enabled_only:
                filter_info += " (nur aktive)"
            return f"Keine Skills gefunden{filter_info}."

        # Nach Kategorie und Name sortieren
        skills.sort(key=lambda s: (s.category, s.name))

        lines = [f"Registrierte Skills ({len(skills)}):"]
        lines.append("")

        current_cat = ""
        for skill in skills:
            if skill.category != current_cat:
                current_cat = skill.category
                lines.append(f"## {current_cat}")

            status = "aktiv" if skill.enabled else "inaktiv"
            keywords = ", ".join(skill.trigger_keywords[:5])
            total = skill.success_count + skill.failure_count
            if total == 0:
                success_info = "noch nicht getestet"
            else:
                success_info = f"Erfolg: {skill.success_rate:.0%} ({total}x genutzt)"

            lines.append(
                f"  - {skill.name} [{skill.slug}] "
                f"({status}, {success_info}, "
                f"Keywords: {keywords})"
            )

        return "\n".join(lines)


def register_skill_tools(
    mcp_client: Any,
    skill_registry: SkillRegistry,
    skills_dirs: list[Path],
) -> SkillTools:
    """Registriert Skill-Management-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        skill_registry: SkillRegistry-Instanz.
        skills_dirs: Liste der Skill-Verzeichnisse.

    Returns:
        SkillTools-Instanz fuer direkten Zugriff.
    """
    st = SkillTools(skill_registry, skills_dirs)

    mcp_client.register_builtin_handler(
        "create_skill",
        st.create_skill,
        description=(
            "Erstellt einen neuen Jarvis-Skill als Markdown-Datei. "
            "Der Skill ist sofort nach Erstellung verfuegbar."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill-Name (z.B. 'PDF Export')",
                },
                "description": {
                    "type": "string",
                    "description": "Kurzbeschreibung des Skills",
                },
                "trigger_keywords": {
                    "type": "string",
                    "description": "Komma-separierte Keywords (z.B. 'pdf,export,dokument')",
                },
                "body": {
                    "type": "string",
                    "description": "Markdown-Inhalt mit Anweisungen/Prozedur fuer den Skill",
                },
                "category": {
                    "type": "string",
                    "description": "Kategorie (default: 'general')",
                    "default": "general",
                },
                "tools_required": {
                    "type": "string",
                    "description": "Komma-separierte Tool-Namen die der Skill benoetigt (optional)",
                    "default": "",
                },
            },
            "required": ["name", "description", "trigger_keywords", "body"],
        },
    )

    mcp_client.register_builtin_handler(
        "list_skills",
        st.list_skills,
        description="Listet alle registrierten Jarvis-Skills mit Name, Kategorie, Keywords und Erfolgsrate.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Kategorie-Filter. Leer lassen oder 'all' fuer alle Kategorien. Beispiele: 'general', 'daily'.",
                    "default": "",
                },
                "enabled_only": {
                    "type": "boolean",
                    "description": "Nur aktive Skills anzeigen",
                    "default": True,
                },
            },
            "required": [],
        },
    )

    log.info("skill_tools_registered", tools=["create_skill", "list_skills"])
    return st
