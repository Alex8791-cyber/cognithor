"""Skill Lifecycle Manager: Audits, repairs, and suggests skills.

Verantwortlich fuer die Gesundheitspruefung des Skill-Inventars:

  1. Findet kaputte, ungenutzte oder veraltete Skills
  2. Repariert behebbare Probleme (leerer Body, YAML-Fehler)
  3. Deaktiviert Skills die dauerhaft nicht genutzt werden
  4. Schlaegt neue Skills auf Basis von Lueckenanalyse vor

Kann manuell oder als periodischer Hintergrund-Job aufgerufen werden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.skills.registry import SkillRegistry

log = get_logger(__name__)

__all__ = [
    "SkillHealthStatus",
    "SkillLifecycleManager",
]

# Schwellenwert: Skills die aelter als N Tage und nie genutzt wurden → unused
_UNUSED_THRESHOLD_DAYS = 30

# Minimale Template-Laenge fuer regenerierten Body
_MIN_BODY_LENGTH = 10


# ============================================================================
# Datenmodell
# ============================================================================


@dataclass
class SkillHealthStatus:
    """Gesundheitsstatus eines einzelnen Skills."""

    slug: str
    name: str
    status: str  # "healthy", "broken", "unused", "deprecated"
    issues: list[str] = field(default_factory=list)
    last_used: str | None = None
    total_uses: int = 0


# ============================================================================
# Lifecycle Manager
# ============================================================================


class SkillLifecycleManager:
    """Prüft, repariert und optimiert das Skill-Inventar.

    Usage:
        mgr = SkillLifecycleManager(registry, generated_dir)
        results = mgr.audit_all()
        broken = mgr.get_broken_skills()
        report = mgr.get_report()
    """

    def __init__(self, registry: SkillRegistry, generated_dir: Path) -> None:
        self._registry = registry
        self._generated_dir = generated_dir

    # ========================================================================
    # Audit
    # ========================================================================

    def audit_all(self) -> list[SkillHealthStatus]:
        """Prueffe alle registrierten Skills auf Probleme.

        Checks:
          - Datei existiert?
          - Body nicht leer?
          - Trigger-Keywords vorhanden?
          - YAML-Frontmatter parsebar?
          - Seit 30+ Tagen ungenutzt?

        Returns:
            Liste von SkillHealthStatus fuer jeden Skill.
        """
        results: list[SkillHealthStatus] = []
        for slug, _skill in self._registry._skills.items():
            status = self._audit_skill(slug)
            if status is not None:
                results.append(status)
        return results

    def audit_single(self, slug: str) -> SkillHealthStatus | None:
        """Prueffe einen einzelnen Skill anhand seines Slugs.

        Returns:
            SkillHealthStatus oder None wenn Skill nicht gefunden.
        """
        if slug not in self._registry._skills:
            return None
        return self._audit_skill(slug)

    def _audit_skill(self, slug: str) -> SkillHealthStatus | None:
        """Interne Audit-Logik fuer einen Skill."""
        skill = self._registry._skills.get(slug)
        if skill is None:
            return None

        issues: list[str] = []
        status = "healthy"

        # 1. Datei-Existenz pruefen
        if not skill.file_path.exists():
            issues.append("File not found")
            status = "broken"

        # 2. Body-Inhalt pruefen
        if not skill.body or not skill.body.strip():
            issues.append("Empty body")
            if status != "broken":
                status = "broken"

        # 3. Trigger-Keywords pruefen
        if not skill.trigger_keywords:
            issues.append("No trigger keywords")

        # 4. YAML-Frontmatter parsebar?  (nur wenn Datei existiert)
        if skill.file_path.exists() and status != "broken":
            yaml_issue = self._check_yaml_frontmatter(skill.file_path)
            if yaml_issue:
                issues.append(yaml_issue)
                status = "broken"

        # 5. Ungenutzt seit N Tagen?
        if (
            status == "healthy"
            and skill.total_uses == 0
            and self._is_older_than(skill, _UNUSED_THRESHOLD_DAYS)
        ):
            status = "unused"

        return SkillHealthStatus(
            slug=slug,
            name=skill.name,
            status=status,
            issues=issues,
            last_used=skill.last_used,
            total_uses=skill.total_uses,
        )

    def _check_yaml_frontmatter(self, path: Path) -> str | None:
        """Versucht die Datei erneut zu parsen.

        Returns:
            Fehlermeldung oder None wenn OK.
        """
        try:
            content = path.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    yaml.safe_load(parts[1])
        except yaml.YAMLError as exc:
            return f"YAML parse error: {exc}"
        except OSError as exc:
            return f"File read error: {exc}"
        return None

    @staticmethod
    def _is_older_than(skill: object, days: int) -> bool:
        """Prueft ob der Skill aelter als N Tage ist.

        Verwendet last_used als Alterssignal.  Wenn kein Datum bekannt ist
        (Skill ist neu), wird False zurueckgegeben — ein Skill ohne jegliches
        Nutzungsdatum gilt als neu, nicht als veraltet.
        """
        last_used = getattr(skill, "last_used", None)
        if last_used is None:
            # Kein Nutzungsdatum → als neu betrachten, nicht als alt
            return False
        try:
            dt = datetime.fromisoformat(last_used)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            threshold = datetime.now(UTC) - timedelta(days=days)
            return dt < threshold
        except (ValueError, TypeError):
            return False

    # ========================================================================
    # Abfragen
    # ========================================================================

    def get_broken_skills(self) -> list[SkillHealthStatus]:
        """Alle Skills mit Status 'broken'."""
        return [s for s in self.audit_all() if s.status == "broken"]

    # ========================================================================
    # Reparatur
    # ========================================================================

    def repair_skill(self, slug: str) -> bool:
        """Versucht einen kaputten Skill zu reparieren.

        Strategien:
          - Fehlendes File → aus Registry entfernen (nicht reparierbar → False)
          - Leerer Body → Template-Body generieren
          - YAML-Fehler → Common YAML-Probleme beheben

        Returns:
            True wenn repariert, False wenn nicht reparierbar.
        """
        skill = self._registry._skills.get(slug)
        if skill is None:
            log.warning("repair_skill_not_found", slug=slug)
            return False

        # Fall 1: Datei fehlt → aus Registry entfernen
        if not skill.file_path.exists():
            log.warning("repair_remove_missing", slug=slug, path=str(skill.file_path))
            with self._registry._lock:
                self._registry._skills.pop(slug, None)
            self._registry._rebuild_index()
            return False

        # Fall 2: Leerer Body → Template generieren
        if not skill.body or not skill.body.strip():
            new_body = self._generate_template_body(skill.name, skill.description)
            skill.body = new_body
            self._registry.register_skill(skill, rebuild_index=True)
            log.info("repair_body_regenerated", slug=slug)
            return True

        # Fall 3: YAML-Fehler → versuche zu reparieren
        yaml_issue = self._check_yaml_frontmatter(skill.file_path)
        if yaml_issue:
            fixed = self._fix_yaml_file(skill.file_path)
            if fixed:
                # Skill neu laden
                reparsed = self._registry._parse_skill_file(skill.file_path)
                if reparsed:
                    self._registry.register_skill(reparsed, rebuild_index=True)
                    log.info("repair_yaml_fixed", slug=slug)
                    return True
            log.warning("repair_yaml_unfixable", slug=slug, issue=yaml_issue)
            return False

        log.info("repair_no_action_needed", slug=slug)
        return True

    def _generate_template_body(self, name: str, description: str) -> str:
        """Generiert einen minimalen Skill-Body aus Name und Beschreibung."""
        desc_line = description.strip() if description.strip() else f"Fuehrt '{name}' aus."
        return (
            f"# {name}\n\n"
            f"{desc_line}\n\n"
            "## Schritte\n\n"
            "1. Aufgabe analysieren\n"
            "2. Relevante Tools auswaehlen\n"
            "3. Ergebnis zusammenfassen\n"
        )

    def _fix_yaml_file(self, path: Path) -> bool:
        """Versucht haefige YAML-Probleme in einer Skill-Datei zu beheben.

        Behebt:
          - Fehlende Anfuehrungszeichen um Werte mit Sonderzeichen
          - Falsch eingerueckte Eintraege (nur einfache Faelle)

        Returns:
            True wenn die Datei veraendert und valide wurde, False sonst.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return False

        if not content.startswith("---"):
            return False

        parts = content.split("---", 2)
        if len(parts) < 3:
            return False

        fm_raw = parts[1]
        body_part = parts[2]

        # Behebe: Doppelpunkte in Werten ohne Anfuehrungszeichen
        fixed_lines: list[str] = []
        for line in fm_raw.splitlines():
            m = re.match(r'^(\s*\w[\w_-]*\s*:\s*)([^"\'\[{#].+)$', line)
            if m:
                key_part = m.group(1)
                val_part = m.group(2).strip()
                # Nur patchen wenn der Wert einen Doppelpunkt enthaelt
                if ":" in val_part and not val_part.startswith(('"', "'")):
                    line = f'{key_part}"{val_part}"'
            fixed_lines.append(line)

        fixed_fm = "\n".join(fixed_lines)
        new_content = f"---{fixed_fm}---{body_part}"

        # Validieren
        try:
            test_parts = new_content.split("---", 2)
            yaml.safe_load(test_parts[1])
        except yaml.YAMLError:
            return False

        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError:
            return False

        return True

    # ========================================================================
    # Vorschlaege
    # ========================================================================

    def suggest_skills(self, recent_queries: list[str] | None = None) -> list[dict[str, Any]]:
        """Analysiert Luecken im Skill-Inventar und macht Vorschlaege.

        Logik:
          1. Keine Skills in Kategorie 'automation' → Automatisierungs-Skill vorschlagen
          2. Keine Skills in 'research' → Recherche-Skill vorschlagen
          3. recent_queries ohne passenden Skill → Skill fuer diese Luecke vorschlagen

        Returns:
            Liste von Vorschlaegen: [{"name": ..., "description": ..., "reason": ...}]
            Maximal 3 Eintraege.
        """
        suggestions: list[dict[str, Any]] = []
        existing_categories = set(self._registry._categories.keys())
        existing_names_lower = {s.name.lower() for s in self._registry._skills.values()}

        # Vorschlag 1: Keine Automatisierungs-Skills
        if "automation" not in existing_categories:
            suggestions.append(
                {
                    "name": "Task Automation",
                    "description": "Automatisiert wiederkehrende Aufgaben wie Datei-Operationen, "
                    "geplante Jobs und Systemwartung.",
                    "reason": "No skills in category 'automation' found.",
                }
            )

        # Vorschlag 2: Keine Recherche-Skills
        has_research = "research" in existing_categories or any(
            "arc" in n or "search" in n or "research" in n for n in existing_names_lower
        )
        if not has_research:
            suggestions.append(
                {
                    "name": "Web Research",
                    "description": "Recherchiert Themen im Web, fasst Ergebnisse zusammen "
                    "und speichert Erkenntnisse in der Memory.",
                    "reason": "No skills in category 'research' and no ARC-like skill found.",
                }
            )

        # Vorschlag 3: Ungedeckte Anfragen aus recent_queries
        if recent_queries and len(suggestions) < 3:
            for query in recent_queries:
                if len(suggestions) >= 3:
                    break
                matches = self._registry.match(query, top_k=1)
                if not matches:
                    suggestions.append(
                        {
                            "name": f"Skill for: {query[:40]}",
                            "description": f"Behandelt Anfragen wie: '{query[:60]}'.",
                            "reason": f"No matching skill found for query: '{query[:60]}'.",
                        }
                    )

        return suggestions[:3]

    # ========================================================================
    # Bereinigung
    # ========================================================================

    def prune_unused(self, days: int = 30) -> list[str]:
        """Deaktiviert Skills die seit N Tagen nicht genutzt wurden.

        Regeln:
          - Nur Skills mit total_uses == 0
          - Nur non-builtin Skills (community oder generiert)
          - Kein Loeschen — nur deaktivieren (enabled=False)

        Args:
            days: Schwellenwert in Tagen.

        Returns:
            Liste der Slugs die deaktiviert wurden.
        """
        pruned: list[str] = []
        for slug, skill in list(self._registry._skills.items()):
            if skill.total_uses > 0:
                continue
            if skill.source == "builtin":
                continue
            if not self._is_older_than(skill, days):
                continue
            skill.enabled = False
            pruned.append(slug)
            log.info("skill_pruned", slug=slug, days=days)

        if pruned:
            self._registry._rebuild_index()

        return pruned

    # ========================================================================
    # Bericht
    # ========================================================================

    def get_report(self) -> str:
        """Erstellt einen menschenlesbaren Statusbericht.

        ASCII-sicher (kein Unicode).

        Returns:
            Formatierter Bericht als String.
        """
        all_statuses = self.audit_all()
        total = len(all_statuses)
        healthy = sum(1 for s in all_statuses if s.status == "healthy")
        broken = sum(1 for s in all_statuses if s.status == "broken")
        unused = sum(1 for s in all_statuses if s.status == "unused")
        suggestions = self.suggest_skills()

        lines: list[str] = [
            "=== Skill Lifecycle Report ===",
            f"Total skills  : {total}",
            f"Healthy       : {healthy}",
            f"Broken        : {broken}",
            f"Unused (>30d) : {unused}",
            "",
        ]

        if broken > 0:
            lines.append("-- Broken Skills --")
            for s in all_statuses:
                if s.status == "broken":
                    issues_str = ", ".join(s.issues) if s.issues else "unknown"
                    lines.append(f"  [{s.slug}] {s.name}: {issues_str}")
            lines.append("")

        if suggestions:
            lines.append("-- Suggestions --")
            for sg in suggestions:
                lines.append(f"  + {sg['name']}: {sg['reason']}")
            lines.append("")

        lines.append("==============================")
        return "\n".join(lines)
