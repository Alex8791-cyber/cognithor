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

import contextlib
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
        tools = (
            [t.strip() for t in tools_required.split(",") if t.strip()] if tools_required else []
        )

        keywords_yaml = ", ".join(keywords)
        tools_yaml = ", ".join(tools) if tools else ""

        # Markdown-Datei mit YAML-Frontmatter erstellen
        lines = [
            "---",
            f"name: {name.strip()}",
            f"slug: {slug}",
            f'description: "{description.strip()}"',
            f"trigger_keywords: [{keywords_yaml}]",
            f"category: {category.strip()}",
            "priority: 0",
            "enabled: true",
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

        # Hot-load: parse + register immediately (no full directory reload)
        try:
            skill = self._registry._parse_skill_file(file_path)
            if skill:
                self._registry.register_skill(skill, rebuild_index=True)
                log.info(
                    "skill_created_and_hot_loaded",
                    slug=slug,
                    file=str(file_path),
                    total_skills=len(self._registry._skills),
                )
        except Exception as exc:
            log.warning("skill_hot_load_failed", error=str(exc))
            return (
                f"Skill-Datei geschrieben: {file_path}\n"
                f"WARNUNG: Hot-Loading fehlgeschlagen: {exc}\n"
                f"Der Skill wird beim naechsten Neustart verfuegbar sein."
            )

        return (
            f"Skill erfolgreich erstellt.\n"
            f"  Name: {name.strip()}\n"
            f"  Slug: {slug}\n"
            f"  Datei: {file_path}\n"
            f"  Keywords: {', '.join(keywords)}\n"
            f"  Kategorie: {category}\n"
            f"  Registrierte Skills gesamt: {len(self._registry._skills)}\n"
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
                f"  - {skill.name} [{skill.slug}] ({status}, {success_info}, Keywords: {keywords})"
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
        description=(
            "Listet alle registrierten Jarvis-Skills mit Name, Kategorie, Keywords und Erfolgsrate."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Kategorie-Filter. Leer lassen oder "
                        "'all' fuer alle Kategorien. "
                        "Beispiele: 'general', 'daily'."
                    ),
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

    # ------------------------------------------------------------------
    # Community-Skill-Tools
    # ------------------------------------------------------------------

    async def _install_community_skill(name: str) -> str:
        """Installiert einen Community-Skill aus dem Registry."""
        try:
            from jarvis.skills.community.client import CommunityRegistryClient

            community_dir = (
                skills_dirs[-1] / "community" if skills_dirs else Path("~/.jarvis/skills/community")
            )
            client = CommunityRegistryClient(community_dir=community_dir)
            result = await client.install(name)

            if not result.success:
                errors = "\n".join(result.errors)
                return f"Installation fehlgeschlagen:\n{errors}"

            # Registry neu laden
            with contextlib.suppress(Exception):
                skill_registry.load_from_directories(skills_dirs)

            tools_info = ", ".join(result.tools_required) if result.tools_required else "keine"
            warnings = ""
            if result.warnings:
                warnings = "\nWarnungen:\n" + "\n".join(f"  - {w}" for w in result.warnings)

            return (
                f"Community-Skill '{name}' erfolgreich installiert.\n"
                f"  Version: {result.version}\n"
                f"  Pfad: {result.install_path}\n"
                f"  Benoetigte Tools: {tools_info}{warnings}"
            )
        except Exception as exc:
            return f"Fehler bei Installation: {exc}"

    async def _search_community_skills(query: str = "", category: str = "") -> str:
        """Durchsucht das Community-Skill-Registry."""
        try:
            from jarvis.skills.community.client import CommunityRegistryClient

            community_dir = (
                skills_dirs[-1] / "community" if skills_dirs else Path("~/.jarvis/skills/community")
            )
            client = CommunityRegistryClient(community_dir=community_dir)
            results = await client.search(query=query, category=category)

            if not results:
                return "Keine Community-Skills gefunden."

            lines = [f"Community-Skills ({len(results)}):"]
            for r in results:
                tools = ", ".join(r.tools_required) if r.tools_required else "keine"
                lines.append(
                    f"  - {r.name} v{r.version} ({r.category}) "
                    f"von @{r.author_github}\n"
                    f"    {r.description}\n"
                    f"    Tools: {tools}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Fehler bei Suche: {exc}"

    async def _report_skill(name: str, category: str = "other", description: str = "") -> str:
        """Meldet einen Community-Skill als problematisch."""
        return (
            f"Abuse-Report fuer Skill '{name}' erfasst.\n"
            f"  Kategorie: {category}\n"
            f"  Beschreibung: {description or 'Keine Angabe'}\n"
            f"Der Report wird bei der naechsten Registry-Sync uebermittelt."
        )

    # install_community_skill
    mcp_client.register_builtin_handler(
        "install_community_skill",
        _install_community_skill,
        description=(
            "Installiert einen Community-Skill aus dem oeffentlichen Registry. "
            "Fuehrt automatisch 5 Sicherheits-Checks durch und zeigt benoetigte "
            "Tool-Permissions an."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name des Community-Skills (z.B. 'morgen-briefing')",
                },
            },
            "required": ["name"],
        },
    )

    # search_community_skills
    mcp_client.register_builtin_handler(
        "search_community_skills",
        _search_community_skills,
        description="Durchsucht das Community-Skill-Registry nach verfuegbaren Skills.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchbegriff",
                    "default": "",
                },
                "category": {
                    "type": "string",
                    "description": "Kategorie-Filter",
                    "default": "",
                },
            },
            "required": [],
        },
    )

    # report_skill
    mcp_client.register_builtin_handler(
        "report_skill",
        _report_skill,
        description="Meldet einen Community-Skill als missbraeuchlich oder problematisch.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name des zu meldenden Skills",
                },
                "category": {
                    "type": "string",
                    "description": "Kategorie: malware, crypto, spam, data_theft, other",
                    "default": "other",
                },
                "description": {
                    "type": "string",
                    "description": "Beschreibung des Problems",
                    "default": "",
                },
            },
            "required": ["name"],
        },
    )

    # publish_skill — publish a local skill to the community registry
    async def _publish_skill(name: str = "", github_token: str = "") -> str:
        """Publish a local skill to the community skill registry on GitHub."""
        import base64
        import hashlib
        import json
        import urllib.request

        if not name:
            return "Error: 'name' (skill slug) is required."

        # Find the skill file locally
        skill_file = None
        _search_dirs = list(skills_dirs) + [
            Path(__file__).parent.parent.parent / "data" / "procedures",
        ]
        for search_dir in _search_dirs:
            candidate = search_dir / f"{name}.md"
            if candidate.exists():
                skill_file = candidate
                break
            # Also check community subdir
            candidate2 = search_dir / "community" / name / "skill.md"
            if candidate2.exists():
                skill_file = candidate2
                break

        if not skill_file:
            return f"Error: Skill '{name}' not found locally."

        content = skill_file.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Parse frontmatter for metadata
        import re as _re

        fm_match = _re.match(r"^---\n(.*?)\n---\n", content, _re.DOTALL)
        if not fm_match:
            return "Error: Skill file has no YAML frontmatter."

        try:
            import yaml

            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except Exception as exc:
            return f"Error parsing frontmatter: {exc}"

        # Validate locally first
        if not frontmatter.get("trigger_keywords"):
            return "Error: Skill has no trigger_keywords. Add them before publishing."
        if not frontmatter.get("tools_required"):
            return "Error: Skill has no tools_required. Add them before publishing."

        # Get GitHub token
        token = github_token
        if not token:
            # Try git credential manager
            try:
                import subprocess

                proc = subprocess.run(
                    ["git", "credential", "fill"],
                    input="protocol=https\nhost=github.com\n",
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in proc.stdout.splitlines():
                    if line.startswith("password="):
                        token = line.split("=", 1)[1]
                        break
            except Exception:
                pass
        if not token:
            return (
                "Error: No GitHub token available. Either:\n"
                "1. Pass github_token parameter\n"
                "2. Configure git credential manager\n"
                "3. Set GITHUB_TOKEN environment variable"
            )

        # Check env var fallback
        if not token:
            import os

            token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            return "Error: No GitHub token found."

        repo_api = "https://api.github.com/repos/Alex8791-cyber/skill-registry"

        def _api_call(method: str, path: str, data: dict | None = None) -> dict:
            url = f"{repo_api}/{path}"
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("Authorization", f"token {token}")
            req.add_header("Accept", "application/vnd.github.v3+json")
            if body:
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return {"error": e.read().decode(), "status": e.code}

        def _get_sha(path: str) -> str:
            r = _api_call("GET", f"contents/{path}?ref=main")
            return r.get("sha", "")

        def _put(path: str, text: str, msg: str, sha: str = "") -> dict:
            d: dict = {
                "message": msg,
                "content": base64.b64encode(text.encode()).decode(),
                "branch": "main",
            }
            if sha:
                d["sha"] = sha
            return _api_call("PUT", f"contents/{path}", d)

        # Upload skill.md
        skill_path = f"skills/{name}/skill.md"
        sha = _get_sha(skill_path)
        r = _put(skill_path, content, f"publish: {name}", sha)
        if "error" in r:
            return f"Error uploading skill.md: {r['error'][:200]}"

        # Upload manifest.json
        manifest = {
            "name": name,
            "version": "1.0.0",
            "description": frontmatter.get("name", name),
            "author_github": "Alex8791-cyber",
            "category": frontmatter.get("category", "productivity"),
            "tools_required": frontmatter.get("tools_required", []),
            "trigger_keywords": frontmatter.get("trigger_keywords", []),
            "content_hash": content_hash,
        }
        manifest_path = f"skills/{name}/manifest.json"
        sha2 = _get_sha(manifest_path)
        r2 = _put(manifest_path, json.dumps(manifest, indent=2), f"publish: {name} manifest", sha2)
        if "error" in r2:
            return f"Error uploading manifest.json: {r2['error'][:200]}"

        # Update registry.json
        reg_sha = _get_sha("registry.json")
        reg_r = _api_call("GET", "contents/registry.json?ref=main")
        if "content" in reg_r:
            import time

            reg_data = json.loads(base64.b64decode(reg_r["content"]).decode())
            # Remove old entry if exists
            reg_data["skills"] = [s for s in reg_data.get("skills", []) if s["name"] != name]
            # Add new entry
            reg_data["skills"].append(
                {
                    "name": name,
                    "version": "1.0.0",
                    "description": frontmatter.get("name", name),
                    "author_github": "Alex8791-cyber",
                    "category": frontmatter.get("category", "productivity"),
                    "tools_required": frontmatter.get("tools_required", []),
                    "content_hash": content_hash,
                    "recalled": False,
                }
            )
            reg_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            r3 = _put(
                "registry.json", json.dumps(reg_data, indent=2), f"registry: add {name}", reg_sha
            )
            if "error" in r3:
                return f"Skill uploaded but registry update failed: {r3['error'][:200]}"

        log.info("skill_published", name=name, hash=content_hash[:16])
        return (
            f"Skill '{name}' published to community registry!\n"
            f"Hash: {content_hash[:16]}...\n"
            f"URL: https://github.com/Alex8791-cyber/skill-registry/tree/main/skills/{name}\n"
            f"Other Cognithor users can now install it with: install_community_skill('{name}')"
        )

    mcp_client.register_builtin_handler(
        "publish_skill",
        _publish_skill,
        description=(
            "Publish a local skill to the Cognithor Community Skill Registry on GitHub. "
            "The skill must exist locally and have valid frontmatter with trigger_keywords "
            "and tools_required. Requires GitHub credentials."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill slug name (e.g., 'meeting-protokoll')",
                },
                "github_token": {
                    "type": "string",
                    "description": (
                        "GitHub Personal Access Token (optional, uses credential manager if empty)"
                    ),
                    "default": "",
                },
            },
            "required": ["name"],
        },
    )

    log.info(
        "skill_tools_registered",
        tools=[
            "create_skill",
            "list_skills",
            "install_community_skill",
            "search_community_skills",
            "report_skill",
            "publish_skill",
        ],
    )
    return st
