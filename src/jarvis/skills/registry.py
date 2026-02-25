"""Skill Registry: Laden, Matchen & Injizieren von Skills in den Agent-Loop.

Skills sind Markdown-Dateien mit YAML-Frontmatter die beschreiben,
WANN und WIE Jarvis bestimmte Aufgaben ausführt. Die Registry:

  1. Lädt Skills beim Start aus ~/.jarvis/skills/ + data/procedures/
  2. Matched User-Nachrichten gegen Skill-Trigger (Keyword + Fuzzy)
  3. Injiziert den besten Skill als Kontext in die Working Memory
  4. Trackt Nutzungsstatistiken und Erfolgsraten

Architektur:
  User-Nachricht → SkillRegistry.match() → Top-Skill
                 → WorkingMemory.injected_procedures
                 → Planner sieht Skill-Kontext im System-Prompt

Bibel-Referenz: §6.2 (Prozedurale Skills), §4.6 (Working Memory Injection)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Matching-Scoring-Konstanten
KEYWORD_MATCH_SCORE = 0.8
FUZZY_MATCH_THRESHOLD = 0.7
FUZZY_MATCH_WEIGHT = 0.7
OVERLAP_SCORE_WEIGHT = 0.5
SUCCESS_RATE_BONUS = 0.1
HIGH_SUCCESS_RATE_THRESHOLD = 0.7
PRIORITY_WEIGHT = 0.05
DEFAULT_MIN_SCORE = 0.15

__all__ = [
    "SkillRegistry",
    "Skill",
    "SkillMatch",
]


# ============================================================================
# Datenmodelle
# ============================================================================


@dataclass
class Skill:
    """Ein geladener Skill mit Metadaten und Inhalt."""

    name: str
    slug: str
    file_path: Path

    # Frontmatter
    trigger_keywords: list[str] = field(default_factory=list)
    tools_required: list[str] = field(default_factory=list)
    description: str = ""
    category: str = "general"
    priority: int = 0  # Höher = bevorzugt bei Gleichstand
    enabled: bool = True
    model_preference: str = ""  # Bevorzugtes LLM-Modell
    agent: str = ""  # Zugeordneter Agent (für Multi-Agent-Routing)

    # Inhalt (Markdown ohne Frontmatter)
    body: str = ""

    # Statistiken
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    avg_score: float = 0.0
    last_used: str | None = None

    @property
    def success_rate(self) -> float:
        """Erfolgsrate (0.0-1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral für ungetestete Skills
        return self.success_count / total


@dataclass
class SkillMatch:
    """Ergebnis eines Skill-Matchings."""

    skill: Skill
    score: float  # 0.0-1.0
    matched_keywords: list[str] = field(default_factory=list)
    match_type: str = "keyword"  # keyword, fuzzy, combined


# ============================================================================
# Skill Registry
# ============================================================================


class SkillRegistry:
    """Zentrale Skill-Verwaltung mit Intent-basiertem Matching.

    Usage:
        registry = SkillRegistry()
        registry.load_from_directories([skills_dir, procedures_dir])

        matches = registry.match("Erstelle ein Morgen-Briefing")
        if matches:
            best = matches[0]
            working_memory.injected_procedures = [best.skill.body]
    """

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._skills: dict[str, Skill] = {}  # slug → Skill
        self._keyword_index: dict[str, list[str]] = {}  # keyword_lower → [slug, ...]
        self._categories: dict[str, list[str]] = {}  # category → [slug, ...]

    # ========================================================================
    # Laden
    # ========================================================================

    def load_from_directories(self, directories: list[Path]) -> int:
        """Lädt Skills aus mehreren Verzeichnissen.

        Spätere Verzeichnisse überschreiben frühere (User > Default).
        Unterstützt sowohl flache .md-Dateien als auch P2P-installierte
        Skills in Unterverzeichnissen (mit skill.md).

        Returns:
            Anzahl geladener Skills.
        """
        count = 0
        for directory in directories:
            if not directory.exists():
                continue
            # Flache .md-Dateien (Standard)
            for md_file in sorted(directory.glob("*.md")):
                try:
                    skill = self._parse_skill_file(md_file)
                    if skill:
                        self._register(skill)
                        count += 1
                except Exception as exc:
                    log.warning("skill_load_error", file=str(md_file), error=str(exc))

            # P2P-installierte Skills (Unterverzeichnisse mit skill.md)
            for sub_dir in sorted(directory.iterdir()):
                if not sub_dir.is_dir():
                    continue
                skill_md = sub_dir / "skill.md"
                if skill_md.exists():
                    try:
                        skill = self._parse_skill_file(skill_md)
                        if skill:
                            self._register(skill)
                            count += 1
                            log.debug("p2p_skill_loaded", name=skill.name, path=str(sub_dir))
                    except Exception as exc:
                        log.warning("p2p_skill_load_error", dir=str(sub_dir), error=str(exc))

        self._rebuild_index()
        log.info(
            "skill_registry_loaded",
            total=len(self._skills),
            categories=list(self._categories.keys()),
        )
        return count

    def _parse_skill_file(self, path: Path) -> Skill | None:
        """Parst eine Skill-Markdown-Datei mit YAML-Frontmatter."""
        content = path.read_text(encoding="utf-8")

        # Frontmatter extrahieren
        frontmatter = {}
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    # Fallback: Einfaches Key-Value Parsing
                    frontmatter = self._parse_simple_frontmatter(parts[1])
                body = parts[2].strip()

        name = frontmatter.get("name", path.stem)
        # P2P-installierte Skills: skill.md → Slug = Verzeichnisname
        if path.stem == "skill" and path.parent.name != path.parent.parent.name:
            slug = frontmatter.get("name", path.parent.name)
        else:
            slug = path.stem

        # Trigger-Keywords normalisieren
        triggers = frontmatter.get("trigger_keywords", [])
        if isinstance(triggers, str):
            triggers = [t.strip() for t in triggers.split(",")]

        return Skill(
            name=name,
            slug=slug,
            file_path=path,
            trigger_keywords=triggers,
            tools_required=frontmatter.get("tools_required", []),
            description=frontmatter.get("description", ""),
            category=frontmatter.get("category", "general"),
            priority=frontmatter.get("priority", 0),
            enabled=frontmatter.get("enabled", True),
            model_preference=frontmatter.get("model_preference", ""),
            agent=frontmatter.get("agent", ""),
            body=body,
            success_count=frontmatter.get("success_count", 0),
            failure_count=frontmatter.get("failure_count", 0),
            total_uses=frontmatter.get("total_uses", 0),
            avg_score=frontmatter.get("avg_score", 0.0),
            last_used=frontmatter.get("last_used"),
        )

    @staticmethod
    def _parse_simple_frontmatter(text: str) -> dict[str, Any]:
        """Fallback-Parser für einfaches Key-Value Frontmatter."""
        result: dict[str, Any] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                # Listen erkennen [a, b, c]
                if value.startswith("[") and value.endswith("]"):
                    items = value[1:-1].split(",")
                    result[key] = [i.strip().strip("'\"") for i in items if i.strip()]
                else:
                    result[key] = value
        return result

    def _register(self, skill: Skill) -> None:
        """Registriert einen Skill in der Registry."""
        with self._lock:
            self._skills[skill.slug] = skill

    def _rebuild_index(self) -> None:
        """Baut den Keyword-Index und Kategorie-Index neu auf."""
        with self._lock:
            self._keyword_index.clear()
            self._categories.clear()

            for slug, skill in self._skills.items():
                if not skill.enabled:
                    continue

                # Keyword-Index
                for keyword in skill.trigger_keywords:
                    kw_lower = keyword.lower().strip()
                    if kw_lower:
                        self._keyword_index.setdefault(kw_lower, []).append(slug)

                # Kategorie-Index
                self._categories.setdefault(skill.category, []).append(slug)

    # ========================================================================
    # Matching
    # ========================================================================

    def match(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = DEFAULT_MIN_SCORE,
        available_tools: list[str] | None = None,
    ) -> list[SkillMatch]:
        """Matched eine User-Nachricht gegen registrierte Skills.

        Scoring:
          - Exakter Keyword-Match: 1.0
          - Partial Keyword-Match: 0.5-0.9 (SequenceMatcher)
          - Wortüberlappung: 0.1-0.5
          - Bonus: +0.1 für hohe Erfolgsrate
          - Bonus: +0.05 * priority

        Args:
            query: User-Nachricht.
            top_k: Maximale Anzahl Ergebnisse.
            min_score: Mindest-Score.
            available_tools: Wenn angegeben, nur Skills deren Tools verfügbar sind.

        Returns:
            Sortierte Liste von SkillMatch (höchster Score zuerst).
        """
        if not query.strip():
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        matches: dict[str, SkillMatch] = {}

        for slug, skill in self._skills.items():
            if not skill.enabled:
                continue

            # Tool-Verfügbarkeit prüfen
            if available_tools is not None and skill.tools_required:
                if not all(t in available_tools for t in skill.tools_required):
                    continue

            score = 0.0
            matched_kws: list[str] = []
            match_type = "none"

            # 1. Exakte Keyword-Matches
            for kw in skill.trigger_keywords:
                kw_lower = kw.lower().strip()
                if not kw_lower:
                    continue

                if kw_lower in query_lower:
                    # Exakter Substring-Match
                    score = max(score, KEYWORD_MATCH_SCORE)
                    matched_kws.append(kw)
                    match_type = "keyword"
                else:
                    # Fuzzy-Match pro Keyword
                    for word in query_words:
                        ratio = SequenceMatcher(None, kw_lower, word).ratio()
                        if ratio > FUZZY_MATCH_THRESHOLD:
                            score = max(score, ratio * FUZZY_MATCH_WEIGHT)
                            matched_kws.append(kw)
                            match_type = "fuzzy"

            # 2. Wort-Überlappung mit Name und Description
            name_words = set(re.findall(r"\w+", skill.name.lower()))
            desc_words = set(re.findall(r"\w+", skill.description.lower())) if skill.description else set()
            all_skill_words = name_words | desc_words

            overlap = query_words & all_skill_words
            if overlap:
                overlap_score = len(overlap) / max(len(query_words), 1) * OVERLAP_SCORE_WEIGHT
                score = max(score, overlap_score)
                if match_type == "none":
                    match_type = "overlap"

            # 3. Bonus für bewährte Skills
            if skill.success_rate > HIGH_SUCCESS_RATE_THRESHOLD and skill.total_uses > 0:
                score += SUCCESS_RATE_BONUS

            # 4. Priority-Bonus
            score += skill.priority * PRIORITY_WEIGHT

            # Clamp
            score = min(score, 1.0)

            if score >= min_score:
                matches[slug] = SkillMatch(
                    skill=skill,
                    score=score,
                    matched_keywords=matched_kws,
                    match_type=match_type,
                )

        # Sortieren: Score absteigend, dann Priority
        ranked = sorted(
            matches.values(),
            key=lambda m: (m.score, m.skill.priority),
            reverse=True,
        )

        return ranked[:top_k]

    def match_best(
        self,
        query: str,
        **kwargs: Any,
    ) -> SkillMatch | None:
        """Gibt den besten Match zurück, oder None."""
        results = self.match(query, **kwargs)
        return results[0] if results else None

    # ========================================================================
    # Zugriff & Verwaltung
    # ========================================================================

    def get(self, slug: str) -> Skill | None:
        """Gibt einen Skill per Slug zurück."""
        return self._skills.get(slug)

    def list_all(self) -> list[Skill]:
        """Alle registrierten Skills (auch deaktivierte)."""
        return list(self._skills.values())

    def list_enabled(self) -> list[Skill]:
        """Nur aktive Skills."""
        return [s for s in self._skills.values() if s.enabled]

    def list_by_category(self, category: str) -> list[Skill]:
        """Skills einer Kategorie."""
        slugs = self._categories.get(category, [])
        return [self._skills[s] for s in slugs if s in self._skills]

    def list_by_agent(self, agent_name: str) -> list[Skill]:
        """Skills die einem bestimmten Agenten zugeordnet sind."""
        return [s for s in self._skills.values() if s.agent == agent_name and s.enabled]

    def enable(self, slug: str) -> bool:
        """Aktiviert einen Skill."""
        skill = self._skills.get(slug)
        if skill:
            skill.enabled = True
            self._rebuild_index()
            return True
        return False

    def disable(self, slug: str) -> bool:
        """Deaktiviert einen Skill."""
        skill = self._skills.get(slug)
        if skill:
            skill.enabled = False
            self._rebuild_index()
            return True
        return False

    def record_usage(self, slug: str, success: bool, score: float = 0.0) -> None:
        """Trackt Nutzung eines Skills."""
        with self._lock:
            skill = self._skills.get(slug)
            if not skill:
                return

            skill.total_uses += 1
            if success:
                skill.success_count += 1
            else:
                skill.failure_count += 1

            # Rolling Average Score
            if skill.total_uses > 0:
                skill.avg_score = (
                    skill.avg_score * (skill.total_uses - 1) + score
                ) / skill.total_uses

            skill.last_used = datetime.now(timezone.utc).isoformat()

    @property
    def count(self) -> int:
        return len(self._skills)

    @property
    def enabled_count(self) -> int:
        return len([s for s in self._skills.values() if s.enabled])

    def stats(self) -> dict[str, Any]:
        """Statistik-Übersicht."""
        return {
            "total": self.count,
            "enabled": self.enabled_count,
            "categories": {
                cat: len(slugs) for cat, slugs in self._categories.items()
            },
            "top_used": sorted(
                [s for s in self._skills.values() if s.total_uses > 0],
                key=lambda s: s.total_uses,
                reverse=True,
            )[:5],
        }

    # ========================================================================
    # Kontext-Injection
    # ========================================================================

    def inject_into_working_memory(
        self,
        query: str,
        working_memory: Any,
        *,
        available_tools: list[str] | None = None,
    ) -> SkillMatch | None:
        """Matched und injiziert den besten Skill in die Working Memory.

        Dies ist die Hauptintegration mit dem Agent-Loop.

        Args:
            query: User-Nachricht.
            working_memory: WorkingMemory-Objekt mit injected_procedures.
            available_tools: Liste verfügbarer Tool-Namen.

        Returns:
            Der verwendete SkillMatch, oder None.
        """
        best = self.match_best(query, available_tools=available_tools)

        if best is None:
            return None

        # Skill-Body in Working Memory injizieren
        if hasattr(working_memory, "injected_procedures"):
            # Nicht doppelt injizieren
            if best.skill.body not in working_memory.injected_procedures:
                working_memory.injected_procedures.append(best.skill.body)

        log.info(
            "skill_injected",
            skill=best.skill.slug,
            score=round(best.score, 2),
            match_type=best.match_type,
            keywords=best.matched_keywords,
        )

        return best
