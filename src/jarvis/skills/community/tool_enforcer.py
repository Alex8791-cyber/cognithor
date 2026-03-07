"""ToolEnforcer: Runtime-Tool-Allowlist fuer Community-Skills.

Kern der architektonischen Sicherheit: Community-Skills duerfen NUR
die in ``tools_required`` deklarierten Tools nutzen.  Alle anderen
Tool-Calls werden geblockt, BEVOR der Gatekeeper sie sieht.

Einbindung:
  - Im PGE-Loop (gateway.py) vor ``Gatekeeper.evaluate()``
  - Gatekeeper ruft ``_enforce_skill_tools()`` als ersten Schritt

Bible reference: §3.2 (Gatekeeper), §6.2 (Skills), §11 (Security)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.models import PlannedAction
    from jarvis.skills.registry import Skill

log = get_logger(__name__)


# ============================================================================
# Datenmodelle
# ============================================================================


@dataclass(frozen=True)
class ToolEnforcementResult:
    """Ergebnis einer ToolEnforcer-Pruefung."""

    allowed: bool
    tool: str
    skill_name: str
    declared_tools: list[str] = field(default_factory=list)
    reason: str = ""


# ============================================================================
# ToolEnforcer
# ============================================================================


class ToolEnforcer:
    """Runtime-Enforcement der Tool-Allowlist fuer Community-Skills.

    Garantie: Ein Community-Skill kann NUR die Tools nutzen, die er
    in ``tools_required`` deklariert hat.  Alles andere wird geblockt.

    Fuer builtin-Skills (``source != "community"``) wird immer ALLOW
    zurueckgegeben — die bestehende Gatekeeper-Logik gilt unveraendert.

    Usage::

        enforcer = ToolEnforcer()
        result = enforcer.check(action, skill)
        if not result.allowed:
            # → BLOCK, Grund in result.reason
    """

    def __init__(self, *, max_tool_calls: int = 0) -> None:
        """Initialisiert den ToolEnforcer.

        Args:
            max_tool_calls: Globaler Default fuer max Tool-Calls pro
                Skill-Aufruf.  0 = kein Limit (Manifest-Wert gilt).
        """
        self._default_max_tool_calls = max_tool_calls

        # Pro-Session Call-Counter: session_id:skill_slug → count
        self._call_counts: dict[str, int] = {}

        # Statistiken
        self._stats = _EnforcerStats()

    # ====================================================================
    # Public API
    # ====================================================================

    def check(
        self,
        action: PlannedAction,
        skill: Skill | None,
    ) -> ToolEnforcementResult:
        """Prueft ob ein Tool-Call fuer den aktiven Skill erlaubt ist.

        Args:
            action: Die geplante Aktion mit Tool-Name.
            skill: Der aktive Skill (None = kein Skill aktiv).

        Returns:
            ToolEnforcementResult — ``allowed=True`` wenn OK.
        """
        self._stats.total_checks += 1

        # Kein Skill aktiv → kein Enforcement
        if skill is None:
            return ToolEnforcementResult(
                allowed=True,
                tool=action.tool,
                skill_name="<none>",
                reason="Kein Skill aktiv",
            )

        # Builtin-Skills → kein Enforcement
        source = getattr(skill, "source", "builtin")
        if source != "community":
            return ToolEnforcementResult(
                allowed=True,
                tool=action.tool,
                skill_name=skill.name,
                declared_tools=skill.tools_required,
                reason="Builtin-Skill, kein Community-Enforcement",
            )

        # Community-Skill → tools_required enforcement
        declared = skill.tools_required
        if not declared:
            # Community-Skill ohne tools_required → BLOCK (Sicherheits-Default)
            self._stats.blocked += 1
            log.warning(
                "tool_enforcer_block",
                tool=action.tool,
                skill=skill.name,
                reason="Community-Skill ohne tools_required",
            )
            return ToolEnforcementResult(
                allowed=False,
                tool=action.tool,
                skill_name=skill.name,
                declared_tools=[],
                reason="Community-Skill hat keine tools_required deklariert — alle Tools geblockt",
            )

        if action.tool not in declared:
            self._stats.blocked += 1
            log.warning(
                "tool_enforcer_block",
                tool=action.tool,
                skill=skill.name,
                declared=declared,
            )
            return ToolEnforcementResult(
                allowed=False,
                tool=action.tool,
                skill_name=skill.name,
                declared_tools=declared,
                reason=f"Tool '{action.tool}' nicht in tools_required deklariert",
            )

        # Max-Tool-Calls Enforcement (aus Manifest)
        manifest = getattr(skill, "manifest", None)
        max_calls = (
            getattr(manifest, "max_tool_calls", 0)
            if manifest
            else self._default_max_tool_calls
        )
        if max_calls > 0:
            key = skill.slug
            current = self._call_counts.get(key, 0)
            if current >= max_calls:
                self._stats.blocked += 1
                return ToolEnforcementResult(
                    allowed=False,
                    tool=action.tool,
                    skill_name=skill.name,
                    declared_tools=declared,
                    reason=f"max_tool_calls ({max_calls}) fuer Skill '{skill.name}' erreicht",
                )
            self._call_counts[key] = current + 1

        self._stats.allowed += 1
        return ToolEnforcementResult(
            allowed=True,
            tool=action.tool,
            skill_name=skill.name,
            declared_tools=declared,
            reason="Tool in tools_required deklariert",
        )

    def reset_call_count(self, skill_slug: str) -> None:
        """Setzt den Call-Counter fuer einen Skill zurueck.

        Wird am Ende eines Skill-Aufrufs aufgerufen.
        """
        self._call_counts.pop(skill_slug, None)

    @property
    def stats(self) -> dict[str, int]:
        """Enforcement-Statistiken."""
        return {
            "total_checks": self._stats.total_checks,
            "allowed": self._stats.allowed,
            "blocked": self._stats.blocked,
        }


@dataclass
class _EnforcerStats:
    """Interne Statistiken."""

    total_checks: int = 0
    allowed: int = 0
    blocked: int = 0
