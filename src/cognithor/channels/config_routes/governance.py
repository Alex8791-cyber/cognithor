"""Cognithor · Governance routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_governance_routes()` — Reputation-Engine, Recall-Manager,
Abuse-Reporter, Governance-Policy, Interop, Economic-Governor, Hub,
Impact-Assessor und Ecosystem-Controller.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ======================================================================
# Governance routes
# ======================================================================


def _register_governance_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Marketplace governance, economics, governance hub, interop, impact."""

    # -- Marketplace-Governance (Phase 20) --------------------------------

    @app.get("/api/v1/governance/reputation/stats", dependencies=deps)
    async def governance_reputation_stats() -> dict[str, Any]:
        """Reputation-Engine Statistiken."""
        engine = getattr(gateway, "_reputation_engine", None)
        if engine is None:
            return {
                "total_entities": 0,
                "avg_score": 0,
                "flagged_count": 0,
                "trust_distribution": {},
            }
        return engine.stats()

    @app.get("/api/v1/governance/reputation/{entity_id}", dependencies=deps)
    async def governance_reputation_detail(entity_id: str) -> dict[str, Any]:
        """Reputation-Score fuer ein Entity."""
        engine = getattr(gateway, "_reputation_engine", None)
        if engine is None:
            return {"error": "Reputation-Engine nicht verfügbar"}
        score = engine.get_score(entity_id)
        if score is None:
            return {"error": f"Entity '{entity_id}' nicht gefunden"}
        return score.to_dict()

    @app.get("/api/v1/governance/recalls/stats", dependencies=deps)
    async def governance_recalls_stats() -> dict[str, Any]:
        """Recall-Manager Statistiken."""
        mgr = getattr(gateway, "_recall_manager", None)
        if mgr is None:
            return {"total_recalls": 0, "active_blocks": 0}
        return mgr.stats()

    @app.get("/api/v1/governance/recalls/active", dependencies=deps)
    async def governance_recalls_active() -> dict[str, Any]:
        """Aktive Recalls."""
        mgr = getattr(gateway, "_recall_manager", None)
        if mgr is None:
            return {"recalls": []}
        return {"recalls": [r.to_dict() for r in mgr.active_recalls()]}

    @app.get("/api/v1/governance/abuse/stats", dependencies=deps)
    async def governance_abuse_stats() -> dict[str, Any]:
        """Abuse-Reporter Statistiken."""
        reporter = getattr(gateway, "_abuse_reporter", None)
        if reporter is None:
            return {"total_reports": 0, "open": 0, "investigating": 0}
        return reporter.stats()

    @app.get("/api/v1/governance/policy/stats", dependencies=deps)
    async def governance_policy_stats() -> dict[str, Any]:
        """Governance-Policy Statistiken."""
        policy = getattr(gateway, "_governance_policy", None)
        if policy is None:
            return {"total_rules": 0, "enabled": 0, "total_triggered": 0}
        return policy.stats()

    # -- Cross-Agent Interop (Phase 22) -----------------------------------

    @app.get("/api/v1/interop/stats", dependencies=deps)
    async def interop_stats() -> dict[str, Any]:
        """Interop-Protokoll Statistiken."""
        interop = getattr(gateway, "_interop", None)
        if interop is None:
            return {"registered_agents": 0, "online": 0}
        return interop.stats()

    @app.get("/api/v1/interop/agents", dependencies=deps)
    async def interop_agents() -> dict[str, Any]:
        """Registrierte Agenten."""
        interop = getattr(gateway, "_interop", None)
        if interop is None:
            return {"agents": []}
        return {"agents": [a.to_dict() for a in interop.online_agents()]}

    @app.get("/api/v1/interop/federation", dependencies=deps)
    async def interop_federation() -> dict[str, Any]:
        """Federation-Status."""
        interop = getattr(gateway, "_interop", None)
        if interop is None:
            return {"links": [], "stats": {}}
        return {
            "links": [link.to_dict() for link in interop.federation.active_links()],
            "stats": interop.federation.stats(),
        }

    # -- Ethik- und Wirtschaftsgovernance (Phase 23) ----------------------

    @app.get("/api/v1/economics/stats", dependencies=deps)
    async def economics_stats() -> dict[str, Any]:
        """Wirtschaftsgovernance Uebersicht."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"budget": {}, "costs": {}, "bias": {}, "fairness": {}, "ethics": {}}
        return gov.stats()

    @app.get("/api/v1/economics/budget", dependencies=deps)
    async def economics_budget() -> dict[str, Any]:
        """Budget-Status."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_entities": 0}
        return gov.budget.stats()

    @app.get("/api/v1/economics/costs", dependencies=deps)
    async def economics_costs() -> dict[str, Any]:
        """Kosten-Tracking."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_entries": 0, "total_cost_eur": 0}
        return gov.costs.stats()

    @app.get("/api/v1/economics/fairness", dependencies=deps)
    async def economics_fairness() -> dict[str, Any]:
        """Fairness-Audit Ergebnisse."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_audits": 0, "pass_rate": 100}
        return gov.fairness.stats()

    @app.get("/api/v1/economics/ethics", dependencies=deps)
    async def economics_ethics() -> dict[str, Any]:
        """Ethik-Policy Status."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_violations": 0}
        return gov.ethics.stats()

    # -- Governance Hub (Phase 31) ----------------------------------------

    @app.get("/api/v1/governance/health", dependencies=deps)
    async def governance_health() -> dict[str, Any]:
        """Ecosystem-Gesundheit."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"skill_reviews": 0}
        return gh.ecosystem_health()

    @app.get("/api/v1/governance/curation", dependencies=deps)
    async def governance_curation() -> dict[str, Any]:
        """Kurations-Board Status."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_reviews": 0}
        return gh.curation.stats()

    @app.get("/api/v1/governance/diversity", dependencies=deps)
    async def governance_diversity() -> dict[str, Any]:
        """Diversity-Audit Ergebnisse."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_audits": 0}
        return gh.diversity.stats()

    @app.get("/api/v1/governance/budget", dependencies=deps)
    async def governance_budget_transfers() -> dict[str, Any]:
        """Cross-Agent-Budget Status."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_transfers": 0}
        return gh.budget.stats()

    @app.get("/api/v1/governance/explainer", dependencies=deps)
    async def governance_explainer() -> dict[str, Any]:
        """Decision-Explainer Statistiken."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_explanations": 0}
        return gh.explainer.stats()

    # -- AI Impact Assessment (Phase 32) ----------------------------------

    @app.get("/api/v1/impact/stats", dependencies=deps)
    async def impact_stats() -> dict[str, Any]:
        """Impact Assessment Statistiken."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"total_assessments": 0}
        return ia.stats()

    @app.get("/api/v1/impact/board", dependencies=deps)
    async def impact_board() -> dict[str, Any]:
        """Ethik-Board Status."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"board_members": 0}
        return ia.board.stats()

    @app.get("/api/v1/impact/stakeholders", dependencies=deps)
    async def impact_stakeholders() -> dict[str, Any]:
        """Stakeholder-Registry."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"total": 0}
        return ia.stakeholders.stats()

    @app.get("/api/v1/impact/mitigations", dependencies=deps)
    async def impact_mitigations() -> dict[str, Any]:
        """Mitigationsmassnahmen."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"total": 0}
        return ia.mitigations.stats()
