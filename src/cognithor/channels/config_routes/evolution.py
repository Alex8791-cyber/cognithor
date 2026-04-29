"""Cognithor · Evolution + Self-Improvement + GEPA-Evolution routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Bundle aus
drei thematisch verwandten Helfern fuer selbst-verbessernde Systeme:

  - `_register_prompt_evolution_routes()` — Prompt-Evolution Endpoints.
  - `_register_self_improvement_routes()` — Reflexion-getriebene
    Self-Improvement.
  - `_register_gepa_evolution_routes()` — GEPA Evolution Loop.

Inkl. `_proposal_to_dict` und `_trace_to_dict` Modul-Level-Helfer
(werden ausschliesslich von `_register_gepa_evolution_routes` genutzt).
"""

from __future__ import annotations

import contextlib
from typing import Any

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

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
# Prompt-Evolution routes
# ======================================================================


def _register_prompt_evolution_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Stats, manual evolve trigger, and enable/disable toggle."""

    @app.get("/api/v1/prompt-evolution/stats", dependencies=deps)
    async def prompt_evolution_stats() -> dict[str, Any]:
        engine = getattr(gateway, "_prompt_evolution", None)
        enabled = engine is not None
        stats: dict[str, Any] = {"enabled": enabled}
        if engine:
            with contextlib.suppress(Exception):
                stats.update(engine.get_stats("system_prompt"))
        return stats

    @app.post("/api/v1/prompt-evolution/evolve", dependencies=deps)
    async def prompt_evolution_evolve() -> dict[str, Any]:
        engine = getattr(gateway, "_prompt_evolution", None)
        if engine is None:
            return {"error": "prompt_evolution is disabled"}
        # Check ImprovementGate
        gate = getattr(gateway, "_improvement_gate", None)
        if gate is not None:
            from cognithor.governance.improvement_gate import GateVerdict, ImprovementDomain

            verdict = gate.check(ImprovementDomain.PROMPT_TUNING)
            if verdict != GateVerdict.ALLOWED:
                return {"error": f"gate_blocked: {verdict.value}"}
        try:
            result = await engine.maybe_evolve("system_prompt")
            return {"evolved": result is not None, "version_id": result}
        except Exception as exc:
            log.error("prompt_evolution_evolve_failed", error=str(exc))
            return {"error": "Prompt-Evolution fehlgeschlagen"}

    @app.post("/api/v1/prompt-evolution/toggle", dependencies=deps)
    async def prompt_evolution_toggle(request: Request) -> dict[str, Any]:
        body = await request.json()
        enabled = body.get("enabled", False)

        if enabled:
            if getattr(gateway, "_prompt_evolution", None) is None:
                try:
                    from cognithor.learning.prompt_evolution import PromptEvolutionEngine

                    cfg = gateway._config
                    pe_db = str(cfg.db_path.with_name("memory_prompt_evolution.db"))
                    engine = PromptEvolutionEngine(
                        db_path=pe_db,
                        min_sessions_per_arm=cfg.prompt_evolution.min_sessions_per_arm,
                        significance_threshold=cfg.prompt_evolution.significance_threshold,
                        max_concurrent_tests=cfg.prompt_evolution.max_concurrent_tests,
                    )
                    engine.set_evolution_interval_hours(
                        cfg.prompt_evolution.evolution_interval_hours
                    )
                    gateway._prompt_evolution = engine
                    planner = getattr(gateway, "_planner", None)
                    if planner:
                        planner._prompt_evolution = engine
                except Exception as exc:
                    log.error("prompt_evolution_toggle_failed", error=str(exc))
                    return {
                        "error": "Prompt-Evolution konnte nicht aktiviert werden",
                        "enabled": False,
                    }
        else:
            # Disable: disconnect from planner but keep engine for stats
            planner = getattr(gateway, "_planner", None)
            if planner:
                planner._prompt_evolution = None
            gateway._prompt_evolution = None

        return {"enabled": getattr(gateway, "_prompt_evolution", None) is not None}


# ======================================================================
# Self-improvement routes
# ======================================================================


def _register_self_improvement_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for the self-improvement engine."""

    def _get_improver() -> Any:
        return getattr(gateway, "_self_improver", None) if gateway else None

    @app.get("/api/v1/learning/self-improvement/stats", dependencies=deps)
    async def self_improvement_stats() -> dict[str, Any]:
        """Return self-improvement engine statistics."""
        improver = _get_improver()
        if not improver:
            return {"error": "Self-improvement engine not initialized", "status": 503}
        return improver.stats()

    @app.get("/api/v1/learning/self-improvement/pending", dependencies=deps)
    async def self_improvement_pending() -> dict[str, Any]:
        """Return pending improvement proposals."""
        improver = _get_improver()
        if not improver:
            return {"error": "Self-improvement engine not initialized", "status": 503}

        pending = improver.pending_improvements
        return {
            "pending": [
                {
                    "id": imp.id,
                    "pattern_id": imp.pattern_id,
                    "improvement_type": imp.improvement_type,
                    "before": imp.before,
                    "after": imp.after,
                    "confidence": round(imp.confidence, 3),
                    "created_at": imp.created_at.isoformat(),
                }
                for imp in pending
            ],
            "count": len(pending),
        }

    @app.post(
        "/api/v1/learning/self-improvement/{improvement_id}/apply",
        dependencies=deps,
    )
    async def self_improvement_apply(improvement_id: str) -> dict[str, Any]:
        """Apply a pending improvement."""
        improver = _get_improver()
        if not improver:
            return {"error": "Self-improvement engine not initialized", "status": 503}

        success = improver.apply_improvement(improvement_id)
        if not success:
            return {"error": f"Improvement '{improvement_id}' not found", "status": 404}

        return {"applied": True, "improvement_id": improvement_id}


# ======================================================================
# GEPA Evolution routes
# ======================================================================


def _register_gepa_evolution_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for GEPA (Guided Evolution through Pattern Analysis)."""

    def _get_orch() -> Any:
        return getattr(gateway, "_evolution_orchestrator", None) if gateway else None

    def _get_trace_store() -> Any:
        return getattr(gateway, "_trace_store", None) if gateway else None

    def _get_proposal_store() -> Any:
        return getattr(gateway, "_proposal_store", None) if gateway else None

    @app.get("/api/v1/evolution/status", dependencies=deps)
    async def get_evolution_status() -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"enabled": False, "message": "GEPA not enabled"}
        return orch.get_status()

    @app.get("/api/v1/learning/gepa/status", dependencies=deps)
    async def gepa_status() -> dict[str, Any]:
        """Get GEPA evolution cycle status (alias under /learning/)."""
        orch = _get_orch()
        if orch is None:
            return {"status": "not_initialized"}
        try:
            status = orch.get_status()
            return {"status": "ok", **status}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    @app.get("/api/v1/evolution/proposals", dependencies=deps)
    async def list_evolution_proposals(status: str = "all") -> dict[str, Any]:
        ps = _get_proposal_store()
        if not ps:
            return {"proposals": []}
        if status == "all":
            proposals = ps.get_history(limit=50)
        else:
            proposals = ps.get_by_status(status)
        return {"proposals": [_proposal_to_dict(p) for p in proposals]}

    @app.get("/api/v1/evolution/proposals/{proposal_id}", dependencies=deps)
    async def get_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        detail = orch.get_proposal_detail(proposal_id)
        if not detail:
            return {"error": "Proposal not found", "status": 404}
        return detail

    @app.post("/api/v1/evolution/proposals/{proposal_id}/apply", dependencies=deps)
    async def apply_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        ok = orch.apply_proposal(proposal_id)
        return {"applied": ok, "proposal_id": proposal_id}

    @app.post("/api/v1/evolution/proposals/{proposal_id}/reject", dependencies=deps)
    async def reject_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        ok = orch.reject_proposal(proposal_id)
        return {"rejected": ok, "proposal_id": proposal_id}

    @app.post("/api/v1/evolution/proposals/{proposal_id}/rollback", dependencies=deps)
    async def rollback_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        ok = orch.rollback_proposal(proposal_id)
        return {"rolled_back": ok, "proposal_id": proposal_id}

    @app.get("/api/v1/evolution/traces", dependencies=deps)
    async def list_evolution_traces(limit: int = 20) -> dict[str, Any]:
        ts = _get_trace_store()
        if not ts:
            return {"traces": []}
        traces = ts.get_recent_traces(limit=min(limit, 100))
        return {"traces": [_trace_to_dict(t) for t in traces]}

    @app.post("/api/v1/evolution/run", dependencies=deps)
    async def trigger_evolution_cycle() -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        result = orch.run_evolution_cycle()
        return {
            "cycle_id": result.cycle_id,
            "traces_analyzed": result.traces_analyzed,
            "findings": result.findings_count,
            "proposals_generated": result.proposals_generated,
            "proposal_applied": result.proposal_applied,
            "auto_rollbacks": result.auto_rollbacks,
            "duration_ms": result.duration_ms,
        }


def _proposal_to_dict(p: Any) -> dict[str, Any]:
    return {
        "proposal_id": p.proposal_id,
        "optimization_type": p.optimization_type,
        "target": p.target,
        "description": p.description,
        "confidence": p.confidence,
        "estimated_impact": p.estimated_impact,
        "failure_category": p.failure_category,
        "tool_name": p.tool_name,
        "status": p.status,
        "created_at": p.created_at,
        "applied_at": p.applied_at,
    }


def _trace_to_dict(t: Any) -> dict[str, Any]:
    return {
        "trace_id": t.trace_id,
        "session_id": t.session_id,
        "goal": t.goal[:200],
        "success_score": t.success_score,
        "model_used": t.model_used,
        "total_duration_ms": t.total_duration_ms,
        "step_count": len(t.steps),
        "failed_steps": len(t.failed_steps),
        "tool_sequence": t.tool_sequence,
        "created_at": t.created_at,
    }
