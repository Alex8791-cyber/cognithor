"""Cognithor · Config-Routes Factory — wird schrittweise in Sub-Module aufgeteilt.

Dieses Modul enthaelt aktuell die komplette `create_config_routes()`-Funktion
sowie alle 24 `_register_*_routes()`-Helper. Im Rahmen des Refactor-Plans
(siehe `docs/superpowers/plans/2026-04-29-config-routes-split.md`) wandern die
Helper schrittweise in eigene Sub-Module unter `cognithor.channels.config_routes/`.
Bis dahin bleiben sie hier — Public-API ist `create_config_routes()`, re-exportiert
ueber `cognithor.channels.config_routes.__init__`.

REST-Endpoints fuer die Konfigurationsverwaltung via WebUI:

  - GET/PATCH /api/v1/config          → Gesamte Konfiguration
  - GET/PATCH /api/v1/config/{section} → Einzelne Sektion
  - GET/POST/DELETE /api/v1/agents     → Agent-Verwaltung
  - GET/POST/DELETE /api/v1/credentials → Credential-Verwaltung
  - GET /api/v1/status                  → System-Status Dashboard

Architektur-Bibel: §12 (Konfiguration), §9.3 (Web UI)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager

log = get_logger(__name__)


from cognithor.channels.config_routes.config import _register_config_routes
from cognithor.channels.config_routes.governance import _register_governance_routes
from cognithor.channels.config_routes.monitoring import (
    _register_monitoring_routes,
    _register_prometheus_routes,
)
from cognithor.channels.config_routes.security import _register_security_routes
from cognithor.channels.config_routes.session import (
    _register_memory_routes,
    _register_session_routes,
)
from cognithor.channels.config_routes.skills import (
    _register_hermes_routes,
    _register_skill_registry_routes,
    _register_skill_routes,
)
from cognithor.channels.config_routes.system import _register_system_routes
from cognithor.channels.config_routes.ui import _register_ui_routes

# ======================================================================
# Public entry-point
# ======================================================================


def create_config_routes(
    app: Any,
    config_manager: ConfigManager,
    *,
    verify_token_dep: Any = None,
    gateway: Any = None,
) -> None:
    """Registriert Config-API-Endpoints auf einer FastAPI-App.

    Args:
        app: FastAPI-App-Instanz.
        config_manager: ConfigManager fuer Read/Write.
        verify_token_dep: Optional FastAPI Depends() fuer Auth.
        gateway: Optional Gateway-Instanz fuer Singleton-Zugriff.
    """
    deps = [verify_token_dep] if verify_token_dep else []

    # Shared MonitoringHub (singleton per app) -- created lazily and used
    # across monitoring, SSE, and audit routes.
    _hub_holder: dict[str, Any] = {"hub": None}

    def _get_hub() -> Any:
        if _hub_holder["hub"] is None:
            from cognithor.gateway.monitoring import MonitoringHub

            _hub_holder["hub"] = MonitoringHub()
        return _hub_holder["hub"]

    _register_system_routes(app, deps, config_manager, gateway)
    _register_config_routes(app, deps, config_manager, gateway)
    _register_session_routes(app, deps, gateway)
    _register_memory_routes(app, deps, gateway)
    _register_skill_routes(app, deps, gateway)
    _register_monitoring_routes(app, deps, _get_hub, config_manager)
    _register_prometheus_routes(app, _get_hub, gateway)
    _register_security_routes(app, deps, gateway)
    _register_governance_routes(app, deps, gateway)
    _register_prompt_evolution_routes(app, deps, gateway)
    _register_infrastructure_routes(app, deps, gateway)
    _register_portal_routes(app, deps, gateway)
    _register_ui_routes(app, deps, config_manager, gateway)
    _register_workflow_graph_routes(app, deps, gateway)
    _register_learning_routes(app, deps, gateway)
    _register_ingest_routes(app, deps, gateway)
    _register_hermes_routes(app, deps, gateway)
    _register_skill_registry_routes(app, deps, gateway)
    _register_self_improvement_routes(app, deps, gateway)
    _register_gepa_evolution_routes(app, deps, gateway)
    _register_backend_routes(app, deps, config_manager, gateway)
    _register_autonomous_routes(app, deps, gateway)
    _register_feedback_routes(app, deps, gateway)
    _register_social_routes(app, deps, gateway)


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
# Infrastructure routes (ecosystem, performance, portal)
# ======================================================================


def _register_infrastructure_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Ecosystem control, performance manager."""

    # -- Ecosystem-Kontrolle (Phase 28) -----------------------------------

    @app.get("/api/v1/ecosystem/stats", dependencies=deps)
    async def ecosystem_stats() -> dict[str, Any]:
        """Ecosystem-Controller Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"curator": {}, "fraud": {}}
        return ctrl.stats()

    @app.get("/api/v1/ecosystem/curator", dependencies=deps)
    async def ecosystem_curator() -> dict[str, Any]:
        """Kuration-Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_reviews": 0}
        return ctrl.curator.stats()

    @app.get("/api/v1/ecosystem/fraud", dependencies=deps)
    async def ecosystem_fraud() -> dict[str, Any]:
        """Fraud-Detection Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_signals": 0}
        return ctrl.fraud.stats()

    @app.get("/api/v1/ecosystem/training", dependencies=deps)
    async def ecosystem_training() -> dict[str, Any]:
        """Security-Training Status."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_modules": 0}
        return ctrl.trainer.stats()

    @app.get("/api/v1/ecosystem/trust", dependencies=deps)
    async def ecosystem_trust() -> dict[str, Any]:
        """Trust-Boundary Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_boundaries": 0}
        return ctrl.trust.stats()

    # -- Performance-Manager (Phase 37) -----------------------------------

    @app.get("/api/v1/performance/health", dependencies=deps)
    async def perf_health() -> dict[str, Any]:
        """Performance Health-Status."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"vector_store": {"entries": 0}}
        return pm.health()

    @app.get("/api/v1/performance/latency", dependencies=deps)
    async def perf_latency() -> dict[str, Any]:
        """Latenz-Statistiken."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"total_samples": 0}
        return pm.latency.stats()

    @app.get("/api/v1/performance/resources", dependencies=deps)
    async def perf_resources() -> dict[str, Any]:
        """Ressourcen-Auslastung."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"snapshots": 0}
        return pm.optimizer.stats()


# ======================================================================
# Portal routes (end-user portal)
# ======================================================================


def _register_portal_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """End-user portal, consent management."""

    @app.get("/api/v1/portal/stats", dependencies=deps)
    async def portal_stats() -> dict[str, Any]:
        """Endnutzer-Portal Statistiken."""
        up = getattr(gateway, "_user_portal", None)
        if up is None:
            return {"consents": {"total_users": 0}}
        return up.stats()

    @app.get("/api/v1/portal/consents", dependencies=deps)
    async def portal_consents() -> dict[str, Any]:
        """Consent-Management Status."""
        up = getattr(gateway, "_user_portal", None)
        if up is None:
            return {"total_users": 0}
        return up.consents.stats()


# ======================================================================
# Workflow Execution Graph API
# ======================================================================


def _register_workflow_graph_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for workflow execution graph visualization."""

    def _get_engines() -> tuple[Any, Any, Any]:
        """Return (simple_engine, dag_engine, template_library) from gateway."""
        simple = getattr(gateway, "_workflow_engine", None) if gateway else None
        dag = getattr(gateway, "_dag_workflow_engine", None) if gateway else None
        tmpl = getattr(gateway, "_template_library", None) if gateway else None
        return simple, dag, tmpl

    # -- Templates ---------------------------------------------------------

    @app.get("/api/v1/workflows/templates", dependencies=deps)
    async def wf_list_templates() -> dict[str, Any]:
        """List all available workflow templates."""
        _, _, tmpl = _get_engines()
        if not tmpl:
            return {"templates": [], "count": 0}
        return {"templates": tmpl.list_all(), "count": tmpl.template_count}

    @app.get("/api/v1/workflows/templates/{template_id}", dependencies=deps)
    async def wf_get_template(template_id: str) -> dict[str, Any]:
        _, _, tmpl = _get_engines()
        if not tmpl:
            return {"error": "Template library unavailable", "status": 503}
        t = tmpl.get(template_id)
        if not t:
            return {"error": "Template not found", "status": 404}
        return t.to_dict()

    # -- Simple workflow instances -----------------------------------------

    @app.get("/api/v1/workflows/instances", dependencies=deps)
    async def wf_list_instances() -> dict[str, Any]:
        """List all workflow instances (simple engine)."""
        simple, _, _ = _get_engines()
        if not simple:
            return {"instances": [], "stats": {}}
        all_inst = list(simple._instances.values())
        return {
            "instances": [i.to_dict() for i in all_inst],
            "stats": simple.stats(),
        }

    @app.get("/api/v1/workflows/instances/{instance_id}", dependencies=deps)
    async def wf_get_instance(instance_id: str) -> dict[str, Any]:
        simple, _, tmpl = _get_engines()
        if not simple:
            return {"error": "Workflow engine unavailable", "status": 503}
        inst = simple.get(instance_id)
        if not inst:
            return {"error": "Instance not found", "status": 404}
        result = inst.to_dict()
        result["step_results"] = inst.step_results
        if tmpl:
            t = tmpl.get(inst.template_id)
            if t:
                result["steps"] = [s.to_dict() for s in t.steps]
        return result

    @app.post("/api/v1/workflows/instances", dependencies=deps)
    async def wf_start_instance(request: Request) -> dict[str, Any]:
        """Start a new workflow from a template."""
        simple, _, tmpl = _get_engines()
        if not simple or not tmpl:
            return {"error": "Workflow engine unavailable", "status": 503}
        body = await request.json()
        template_id = body.get("template_id", "")
        t = tmpl.get(template_id)
        if not t:
            return {"error": f"Template '{template_id}' not found", "status": 404}
        inst = simple.start(t, created_by=body.get("created_by", "ui"))
        return {"status": "ok", "instance": inst.to_dict()}

    # -- DAG workflow runs -------------------------------------------------

    @app.get("/api/v1/workflows/dag/runs", dependencies=deps)
    async def wf_list_dag_runs() -> dict[str, Any]:
        """List DAG workflow runs (checkpoint-based)."""
        _, dag, _ = _get_engines()
        if not dag or not dag._checkpoint_dir:
            return {"runs": []}
        cp_dir = dag._checkpoint_dir
        runs = []
        if cp_dir.exists():
            for cp_file in sorted(cp_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(cp_file.read_text(encoding="utf-8"))
                    runs.append(
                        {
                            "id": data.get("id", ""),
                            "workflow_id": data.get("workflow_id", ""),
                            "workflow_name": data.get("workflow_name", ""),
                            "status": data.get("status", ""),
                            "started_at": data.get("started_at"),
                            "completed_at": data.get("completed_at"),
                            "node_count": len(data.get("node_results", {})),
                        }
                    )
                except Exception:
                    continue
        return {"runs": runs}

    @app.get("/api/v1/workflows/dag/runs/{run_id}", dependencies=deps)
    async def wf_get_dag_run(run_id: str) -> dict[str, Any]:
        """Get full DAG workflow run with node graph data."""
        _, dag, _ = _get_engines()
        if not dag or not dag._checkpoint_dir:
            return {"error": "DAG engine unavailable", "status": 503}
        cp_file = (dag._checkpoint_dir / f"{run_id}.json").resolve()
        try:
            cp_file.relative_to(dag._checkpoint_dir.resolve())
        except ValueError:
            return {"error": "Invalid run_id (Path-Traversal)", "status": 400}
        if not cp_file.exists():
            return {"error": "Run not found", "status": 404}
        try:
            return json.loads(cp_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("wf_dag_run_read_failed", run_id=run_id, error=str(exc))
            return {"error": "DAG-Run konnte nicht geladen werden", "status": 500}

    @app.get("/api/v1/workflows/dag/runs/{run_id}/nodes/{node_id}", dependencies=deps)
    async def wf_get_dag_node_detail(run_id: str, node_id: str) -> dict[str, Any]:
        """Get detailed execution data for a single DAG node."""
        _, dag, _ = _get_engines()
        if not dag or not dag._checkpoint_dir:
            return {"error": "DAG engine unavailable", "status": 503}
        cp_file = (dag._checkpoint_dir / f"{run_id}.json").resolve()
        try:
            cp_file.relative_to(dag._checkpoint_dir.resolve())
        except ValueError:
            return {"error": "Invalid run_id (Path-Traversal)", "status": 400}
        if not cp_file.exists():
            return {"error": "Run not found", "status": 404}
        try:
            data = json.loads(cp_file.read_text(encoding="utf-8"))
            node_results = data.get("node_results", {})
            if node_id not in node_results:
                return {"error": f"Node '{node_id}' not found in run", "status": 404}
            return {"node_id": node_id, "run_id": run_id, **node_results[node_id]}
        except json.JSONDecodeError:
            return {"error": "Invalid run data", "status": 500}

    # -- Combined stats ----------------------------------------------------

    @app.get("/api/v1/workflows/stats", dependencies=deps)
    async def wf_stats() -> dict[str, Any]:
        """Combined workflow stats."""
        simple, dag, tmpl = _get_engines()
        result: dict[str, Any] = {"templates": 0, "simple": {}, "dag_runs": 0}
        if tmpl:
            result["templates"] = tmpl.template_count
        if simple:
            result["simple"] = simple.stats()
        if dag and dag._checkpoint_dir and dag._checkpoint_dir.exists():
            result["dag_runs"] = len(list(dag._checkpoint_dir.glob("*.json")))
        return result


# ======================================================================
# Learning / Curiosity / Confidence routes
# ======================================================================


def _register_learning_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Active Learning, Curiosity Engine, and Confidence Manager."""

    def _get_learner() -> Any:
        return getattr(gateway, "_active_learner", None) if gateway else None

    def _get_curiosity() -> Any:
        return getattr(gateway, "_curiosity_engine", None) if gateway else None

    def _get_confidence() -> Any:
        return getattr(gateway, "_confidence_manager", None) if gateway else None

    # -- Stats -------------------------------------------------------------

    @app.get("/api/v1/learning/stats", dependencies=deps)
    async def learning_stats() -> dict[str, Any]:
        """Combined learning statistics."""
        result: dict[str, Any] = {}

        learner = _get_learner()
        if learner:
            result["active_learner"] = learner.stats()

        curiosity = _get_curiosity()
        if curiosity:
            result["curiosity"] = {
                "total_gaps": len(curiosity.gaps),
                "open_gaps": curiosity.open_gap_count,
            }

        confidence = _get_confidence()
        if confidence:
            result["confidence"] = confidence.stats()

        if not result:
            result["message"] = "Learning subsystem not initialized"

        return result

    # -- Knowledge gaps ----------------------------------------------------

    @app.get("/api/v1/learning/gaps", dependencies=deps)
    async def learning_gaps() -> dict[str, Any]:
        """List detected knowledge gaps."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"gaps": [], "count": 0}

        gaps = curiosity.gaps
        return {
            "gaps": [
                {
                    "id": g.id,
                    "question": g.question,
                    "topic": g.topic,
                    "importance": g.importance,
                    "curiosity": g.curiosity,
                    "status": g.status,
                    "created_at": g.created_at.isoformat(),
                    "suggested_sources": g.suggested_sources,
                }
                for g in gaps
            ],
            "count": len(gaps),
            "open": sum(1 for g in gaps if g.status == "open"),
        }

    @app.post("/api/v1/learning/gaps/{gap_id}/dismiss", dependencies=deps)
    async def learning_dismiss_gap(gap_id: str) -> dict[str, Any]:
        """Dismiss a knowledge gap."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"error": "Curiosity engine not initialized", "status": 503}

        found = curiosity.dismiss_gap(gap_id)
        if not found:
            return {"error": "Gap not found", "status": 404}
        return {"status": "dismissed", "gap_id": gap_id}

    # -- Confidence history ------------------------------------------------

    @app.get("/api/v1/learning/confidence/history", dependencies=deps)
    async def learning_confidence_history() -> dict[str, Any]:
        """Return recent confidence changes."""
        confidence = _get_confidence()
        if not confidence:
            return {"history": [], "stats": {}}

        history = confidence.history
        # Return last 100 entries
        recent = history[-100:]
        return {
            "history": [
                {
                    "entity_id": h.entity_id,
                    "old_confidence": round(h.old_confidence, 4),
                    "new_confidence": round(h.new_confidence, 4),
                    "reason": h.reason,
                    "timestamp": h.timestamp.isoformat(),
                }
                for h in recent
            ],
            "stats": confidence.stats(),
        }

    @app.post("/api/v1/learning/confidence/{entity_id}/feedback", dependencies=deps)
    async def learning_confidence_feedback(entity_id: str, request: Request) -> dict[str, Any]:
        """Apply feedback to an entity's confidence."""
        confidence = _get_confidence()
        if not confidence:
            return {"error": "Confidence manager not initialized", "status": 503}

        body = await request.json()
        feedback_type = body.get("type", "")
        if feedback_type not in ("positive", "negative", "correction"):
            return {"error": "Invalid feedback type. Must be: positive, negative, correction"}

        # Read current confidence from entity DB
        current = 0.5  # fallback
        mm = getattr(gateway, "_memory_manager", None)
        idx = getattr(mm, "_index", None) if mm else None
        if idx:
            try:
                ent = idx.get_entity_by_id(entity_id)
                if ent:
                    current = ent.confidence
            except Exception:
                log.debug("entity_confidence_read_failed", exc_info=True)

        new_conf = confidence.apply_feedback(entity_id, current, feedback_type)

        # Persist updated confidence to database
        if idx:
            with contextlib.suppress(Exception):
                idx.update_entity_confidence(entity_id, new_conf)

        return {
            "entity_id": entity_id,
            "old_confidence": round(current, 4),
            "new_confidence": round(new_conf, 4),
            "feedback_type": feedback_type,
        }

    # -- Exploration queue -------------------------------------------------

    @app.get("/api/v1/learning/queue", dependencies=deps)
    async def learning_queue() -> dict[str, Any]:
        """Return the exploration task queue."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"tasks": [], "count": 0}

        tasks = curiosity.propose_exploration()
        return {
            "tasks": [
                {
                    "gap_id": t.gap_id,
                    "query": t.query,
                    "sources": t.sources,
                    "priority": t.priority,
                    "max_depth": t.max_depth,
                }
                for t in tasks
            ],
            "count": len(tasks),
        }

    @app.post("/api/v1/learning/explore", dependencies=deps)
    async def learning_explore(request: Request) -> dict[str, Any]:
        """Trigger exploration of a specific gap."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"error": "Curiosity engine not initialized", "status": 503}

        body = await request.json()
        gap_id = body.get("gap_id", "")

        if not gap_id:
            return {"error": "gap_id is required"}

        found = curiosity.mark_exploring(gap_id)
        if not found:
            return {"error": "Gap not found", "status": 404}

        return {"status": "exploring", "gap_id": gap_id}

    # -- Watch directories -------------------------------------------------

    @app.get("/api/v1/learning/directories", dependencies=deps)
    async def learning_directories() -> dict[str, Any]:
        """Return watched directories configuration."""
        learner = _get_learner()
        if not learner:
            return {"directories": []}
        dirs = learner.stats().get("watch_dirs", [])
        return {"directories": dirs}

    @app.post("/api/v1/learning/directories", dependencies=deps)
    async def learning_update_directories(request: Request) -> dict[str, Any]:
        """Update watched directories (enable/disable, add new)."""
        learner = _get_learner()
        if not learner:
            return {"error": "Active learner not initialized", "status": 503}
        body = await request.json()
        dirs = body.get("directories", [])
        for d in dirs:
            path = d.get("path", "")
            enabled = d.get("enabled", True)
            if path:
                learner.add_directory(path, enabled=enabled)
        return {"status": "updated", "count": len(dirs)}

    # -- Q&A Knowledge Base ------------------------------------------------

    def _get_qa() -> Any:
        return getattr(gateway, "_knowledge_qa", None) if gateway else None

    @app.get("/api/v1/learning/qa", dependencies=deps)
    async def learning_qa_list(request: Request) -> dict[str, Any]:
        """List or search Q&A pairs."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        query = request.query_params.get("q", "")
        limit = int(request.query_params.get("limit", "50"))
        offset = int(request.query_params.get("offset", "0"))

        if query:
            pairs = qa_store.search(query, limit=limit)
        else:
            pairs = qa_store.list_all(limit=limit, offset=offset)

        return {
            "pairs": [
                {
                    "id": p.id,
                    "question": p.question,
                    "answer": p.answer,
                    "topic": p.topic,
                    "confidence": round(p.confidence, 4),
                    "source": p.source,
                    "entity_id": p.entity_id,
                    "created_at": p.created_at,
                    "last_verified": p.last_verified,
                    "verification_count": p.verification_count,
                }
                for p in pairs
            ],
            "count": len(pairs),
            "stats": qa_store.stats(),
        }

    @app.post("/api/v1/learning/qa", dependencies=deps)
    async def learning_qa_add(request: Request) -> dict[str, Any]:
        """Add a new Q&A pair."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        body = await request.json()
        question = body.get("question", "").strip()
        answer = body.get("answer", "").strip()
        if not question or not answer:
            return {"error": "question and answer are required"}

        pair = qa_store.add(
            question,
            answer,
            topic=body.get("topic", ""),
            confidence=float(body.get("confidence", 0.5)),
            source=body.get("source", ""),
            entity_id=body.get("entity_id", ""),
        )
        return {
            "id": pair.id,
            "question": pair.question,
            "answer": pair.answer,
            "topic": pair.topic,
            "confidence": pair.confidence,
            "created_at": pair.created_at,
        }

    @app.post(
        "/api/v1/learning/qa/{qa_id}/verify",
        dependencies=deps,
    )
    async def learning_qa_verify(qa_id: str) -> dict[str, Any]:
        """Verify a Q&A pair, boosting its confidence."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        found = qa_store.verify(qa_id)
        if not found:
            return {"error": "QA pair not found", "status": 404}
        return {"status": "verified", "id": qa_id}

    @app.delete(
        "/api/v1/learning/qa/{qa_id}",
        dependencies=deps,
    )
    async def learning_qa_delete(qa_id: str) -> dict[str, Any]:
        """Delete a Q&A pair."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        found = qa_store.delete(qa_id)
        if not found:
            return {"error": "QA pair not found", "status": 404}
        return {"status": "deleted", "id": qa_id}

    # -- Knowledge Lineage -------------------------------------------------

    def _get_lineage() -> Any:
        return getattr(gateway, "_knowledge_lineage", None) if gateway else None

    @app.get(
        "/api/v1/learning/lineage/{entity_id}",
        dependencies=deps,
    )
    async def learning_lineage_entity(
        entity_id: str,
        request: Request,
    ) -> dict[str, Any]:
        """Get lineage entries for a specific entity."""
        tracker = _get_lineage()
        if not tracker:
            return {"error": "Lineage tracker not initialized", "status": 503}

        limit = int(request.query_params.get("limit", "50"))
        entries = tracker.get_entity_lineage(
            entity_id,
            limit=limit,
        )
        return {
            "entity_id": entity_id,
            "entries": [
                {
                    "id": e.id,
                    "source_type": e.source_type,
                    "source_path": e.source_path,
                    "action": e.action,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "confidence_before": e.confidence_before,
                    "confidence_after": e.confidence_after,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            "count": len(entries),
        }

    @app.get("/api/v1/learning/lineage", dependencies=deps)
    async def learning_lineage_recent(
        request: Request,
    ) -> dict[str, Any]:
        """Get recent lineage entries."""
        tracker = _get_lineage()
        if not tracker:
            return {"error": "Lineage tracker not initialized", "status": 503}

        limit = int(request.query_params.get("limit", "100"))
        entries = tracker.get_recent(limit=limit)
        return {
            "entries": [
                {
                    "id": e.id,
                    "entity_id": e.entity_id,
                    "source_type": e.source_type,
                    "source_path": e.source_path,
                    "action": e.action,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "confidence_before": e.confidence_before,
                    "confidence_after": e.confidence_after,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            "count": len(entries),
            "stats": tracker.stats(),
        }

    # -- Batch Exploration -------------------------------------------------

    def _get_explorer() -> Any:
        return getattr(gateway, "_exploration_executor", None) if gateway else None

    @app.post(
        "/api/v1/learning/explore/run",
        dependencies=deps,
    )
    async def learning_explore_run(
        request: Request,
    ) -> dict[str, Any]:
        """Trigger a batch of exploration tasks."""
        explorer = _get_explorer()
        if not explorer:
            return {
                "error": "Exploration executor not initialized",
                "status": 503,
            }

        body = await request.json()
        max_tasks = int(body.get("max_tasks", 3))
        max_tasks = max(1, min(max_tasks, 10))

        results = await explorer.execute_batch(
            max_tasks=max_tasks,
        )
        return {
            "results": [
                {
                    "gap_id": r.gap_id,
                    "query": r.query,
                    "found_answer": r.found_answer,
                    "answer_summary": r.answer_summary,
                    "sources_checked": r.sources_checked,
                    "entities_updated": r.entities_updated,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results
            ],
            "count": len(results),
            "stats": explorer.stats(),
        }


# ======================================================================
# Knowledge Ingestion routes (file upload, URL, YouTube)
# ======================================================================


def _register_ingest_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for knowledge ingestion (files, URLs, YouTube)."""

    def _get_ingest() -> Any:
        return getattr(gateway, "_knowledge_ingest", None) if gateway else None

    # -- File upload --------------------------------------------------------

    @app.post("/api/v1/learn/file", dependencies=deps)
    async def learn_file(request: Request) -> dict[str, Any]:
        """Ingest a file upload (multipart/form-data).

        Expects field 'file' with the document/image.
        Optional field 'description' with context text.
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            form = await request.form()
            file_field = form.get("file")
            if file_field is None:
                return {"error": "Field 'file' is required", "code": "MISSING_FIELD"}

            file_bytes = await file_field.read()
            if not file_bytes:
                return {"error": "Empty file", "code": "EMPTY_FILE"}

            filename = "upload"
            if hasattr(file_field, "filename") and file_field.filename:
                filename = file_field.filename

            priority_str = str(form.get("priority", "normal") or "normal")
            from cognithor.learning.knowledge_ingest import Priority

            priority = Priority.from_string(priority_str)
            result = await svc.ingest_file(filename, file_bytes, priority=priority)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "chunks": result.chunks,
                "deep_learn_status": result.deep_learn_status,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_file_error", error=str(exc))
            return {"error": "File ingestion failed", "code": "INTERNAL_ERROR"}

    # -- URL ingestion ------------------------------------------------------

    @app.post("/api/v1/learn/url", dependencies=deps)
    async def learn_url(request: Request) -> dict[str, Any]:
        """Ingest a website URL.

        JSON body: {"url": "https://...", "description": "optional", "priority": "normal"}
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            body = await request.json()
            url = body.get("url", "").strip()
            if not url:
                return {"error": "Field 'url' is required", "code": "MISSING_FIELD"}

            priority_str = body.get("priority", "normal")
            from cognithor.learning.knowledge_ingest import Priority

            priority = Priority.from_string(priority_str)
            result = await svc.ingest_url(url, priority=priority)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "chunks": result.chunks,
                "deep_learn_status": result.deep_learn_status,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_url_error", error=str(exc))
            return {"error": "URL ingestion failed", "code": "INTERNAL_ERROR"}

    # -- YouTube ingestion --------------------------------------------------

    @app.post("/api/v1/learn/youtube", dependencies=deps)
    async def learn_youtube(request: Request) -> dict[str, Any]:
        """Ingest a YouTube video transcript.

        JSON body: {"url": "https://youtube.com/watch?v=...", "priority": "normal"}
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            body = await request.json()
            url = body.get("url", "").strip()
            if not url:
                return {"error": "Field 'url' is required", "code": "MISSING_FIELD"}

            priority_str = body.get("priority", "normal")
            from cognithor.learning.knowledge_ingest import Priority

            priority = Priority.from_string(priority_str)
            result = await svc.ingest_youtube(url, priority=priority)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "chunks": result.chunks,
                "deep_learn_status": result.deep_learn_status,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_youtube_error", error=str(exc))
            return {"error": "YouTube ingestion failed", "code": "INTERNAL_ERROR"}

    # -- Queue status -------------------------------------------------------

    @app.get("/api/v1/learn/queue", dependencies=deps)
    async def learn_queue() -> dict[str, Any]:
        """Show pending deep-learn tasks."""
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}
        return {"queue": svc._queue.pending(), "size": len(svc._queue)}

    # -- History & Stats ----------------------------------------------------

    @app.get("/api/v1/learn/history", dependencies=deps)
    async def learn_history(request: Request) -> dict[str, Any]:
        """List ingestion results."""
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        limit = int(request.query_params.get("limit", "50"))
        results = svc.results
        # Return most recent first
        recent = list(reversed(results))[:limit]

        return {
            "results": [
                {
                    "id": r.id,
                    "source_type": r.source_type,
                    "source_name": r.source_name,
                    "status": r.status,
                    "chunks_created": r.chunks_created,
                    "text_length": r.text_length,
                    "error": r.error,
                    "created_at": r.created_at.isoformat(),
                }
                for r in recent
            ],
            "count": len(results),
            "stats": svc.stats(),
        }

    @app.get("/api/v1/learn/stats", dependencies=deps)
    async def learn_stats() -> dict[str, Any]:
        """Return ingestion statistics."""
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        return svc.stats()


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


# ======================================================================
# Backend status / switch routes
# ======================================================================


def _register_backend_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any,
) -> None:
    """Endpoints for querying LLM backend availability and switching backends."""

    import shutil

    @app.get("/api/v1/backend/status", dependencies=deps)
    async def get_backend_status() -> dict[str, Any]:
        """Check which LLM backends are available and authenticated."""
        results: dict[str, Any] = {}

        # Claude Code CLI
        claude_path = shutil.which("claude")
        results["claude-code"] = {
            "installed": claude_path is not None,
            "path": claude_path or "",
            "authenticated": False,
            "models": [],
        }
        if claude_path:
            try:
                proc = await asyncio.create_subprocess_exec(
                    claude_path,
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                if proc.returncode == 0:
                    results["claude-code"]["authenticated"] = True
                    results["claude-code"]["version"] = stdout.decode().strip()
                    results["claude-code"]["models"] = [
                        "opus",
                        "sonnet",
                        "haiku",
                    ]
            except Exception:
                log.debug("claude_code_provider_check_failed", exc_info=True)

        # Ollama
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:11434/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    results["ollama"] = {
                        "installed": True,
                        "authenticated": True,
                        "models": models,
                    }
                else:
                    results["ollama"] = {
                        "installed": True,
                        "authenticated": False,
                        "models": [],
                    }
        except Exception:
            results["ollama"] = {
                "installed": False,
                "authenticated": False,
                "models": [],
            }

        # OpenAI (check if key is set)
        cfg = config_manager.config if config_manager else None
        if cfg:
            has_openai = bool(getattr(cfg, "openai_api_key", ""))
            results["openai"] = {
                "installed": True,
                "authenticated": has_openai,
                "models": [],
            }

            has_anthropic = bool(getattr(cfg, "anthropic_api_key", ""))
            results["anthropic"] = {
                "installed": True,
                "authenticated": has_anthropic,
                "models": [],
            }

            has_openrouter = bool(getattr(cfg, "openrouter_api_key", ""))
            results["openrouter"] = {
                "installed": True,
                "authenticated": has_openrouter,
                "models": [],
            }

        # Current backend
        current = getattr(cfg, "llm_backend_type", "ollama") if cfg else "ollama"

        return {"backends": results, "current": current}

    @app.post("/api/v1/backend/switch", dependencies=deps)
    async def switch_backend(request: Request) -> dict[str, Any]:
        """Switch the LLM backend type."""
        body = await request.json()
        new_backend = body.get("backend", "")

        # Derive valid backends from the config Literal type (single source of truth)
        from typing import get_args, get_type_hints

        from cognithor.config import CognithorConfig

        _hints = get_type_hints(CognithorConfig, include_extras=True)
        valid = list(get_args(_hints["llm_backend_type"]))
        if new_backend not in valid:
            raise HTTPException(400, f"Invalid backend: {new_backend}. Valid: {valid}")

        # Update config
        if config_manager:
            config_manager.config.llm_backend_type = new_backend
            # Save to config.yaml
            with contextlib.suppress(Exception):
                config_manager.save()

        return {
            "status": "switched",
            "backend": new_backend,
            "note": "Restart required for full effect",
        }


# ======================================================================
# Autonomous Task Orchestration routes
# ======================================================================


def _register_autonomous_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Endpoints for querying autonomous task execution status."""

    @app.get("/api/v1/autonomous/tasks", dependencies=deps)
    async def list_autonomous_tasks() -> dict[str, Any]:
        """List active autonomous tasks."""
        if not hasattr(gateway, "_autonomous_orchestrator"):
            return {"tasks": []}
        return {"tasks": gateway._autonomous_orchestrator.get_active_tasks()}


# ======================================================================
# Feedback routes (thumbs up/down)
# ======================================================================


def _register_feedback_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for user feedback (thumbs up/down)."""

    @app.post("/api/v1/feedback", dependencies=deps)
    async def submit_feedback(request: Request) -> dict[str, Any]:
        """Submit thumbs up/down feedback for a message."""
        body = await request.json()
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"error": "Feedback system not initialized"}

        rating = body.get("rating", 0)
        if rating not in (1, -1):
            return {"error": "rating must be 1 (thumbs up) or -1 (thumbs down)"}

        feedback_id = feedback_store.submit(
            session_id=body.get("session_id", ""),
            message_id=body.get("message_id", ""),
            rating=rating,
            comment=body.get("comment", ""),
            agent_name=body.get("agent_name", "jarvis"),
            channel=body.get("channel", "webui"),
            user_message=body.get("user_message", ""),
            assistant_response=body.get("assistant_response", ""),
            tool_calls=body.get("tool_calls", ""),
        )
        return {"status": "ok", "feedback_id": feedback_id}

    @app.patch("/api/v1/feedback/{feedback_id}", dependencies=deps)
    async def update_feedback_comment(feedback_id: str, request: Request) -> dict[str, Any]:
        """Add comment to existing feedback (after follow-up question)."""
        body = await request.json()
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"error": "Feedback system not initialized"}

        ok = feedback_store.add_comment(feedback_id, body.get("comment", ""))
        return {"status": "ok" if ok else "not_found"}

    @app.get("/api/v1/feedback/stats", dependencies=deps)
    async def feedback_stats(agent_name: str = "", hours: int = 0) -> dict[str, Any]:
        """Get feedback statistics."""
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0}
        return feedback_store.get_stats(agent_name=agent_name, hours=hours)

    @app.get("/api/v1/feedback/recent", dependencies=deps)
    async def recent_feedback(limit: int = 50) -> dict[str, Any]:
        """Get recent feedback entries."""
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"entries": []}
        return {"entries": feedback_store.get_recent(limit=limit)}

    # ── Chat Tree / Branching ────────────────────────────────────────

    @app.get("/api/v1/chat/tree/latest", dependencies=deps)
    async def get_latest_chat_tree(session_id: str = "") -> dict[str, Any]:
        """Get the most recent conversation tree.

        If session_id is provided, look up the session's persisted
        conversation_id first so the correct tree is returned.
        """
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"nodes": [], "conversation_id": None}

        conv_id = None

        # Try session-specific conversation first
        if session_id:
            store = getattr(gateway, "_session_store", None)
            if store:
                session = store.load_session_by_id(session_id)
                if session and getattr(session, "conversation_id", ""):
                    conv_id = session.conversation_id

        # Fallback: most recent conversation
        if not conv_id:
            with tree._conn() as conn:
                row = conn.execute(
                    "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                if row:
                    conv_id = row["id"]

        if not conv_id:
            return {"nodes": [], "conversation_id": None}
        return tree.get_tree_structure(conv_id)

    @app.get("/api/v1/chat/tree/{conversation_id}", dependencies=deps)
    async def get_chat_tree(conversation_id: str) -> dict[str, Any]:
        """Get full conversation tree structure."""
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        return tree.get_tree_structure(conversation_id)

    @app.get("/api/v1/chat/path/{conversation_id}/{leaf_id}", dependencies=deps)
    async def get_chat_path(conversation_id: str, leaf_id: str) -> dict[str, Any]:
        """Get path from root to a specific leaf."""
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        path = tree.get_path_to_root(leaf_id)
        return {"path": path, "count": len(path)}

    @app.post("/api/v1/chat/branch", dependencies=deps)
    async def create_chat_branch(request: Request) -> dict[str, Any]:
        """Create a branch at a specific node."""
        body = await request.json()
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        conv_id = body.get("conversation_id", "")
        parent_id = body.get("parent_id", "")
        text = body.get("text", "")
        role = body.get("role", "user")
        if not conv_id or not text:
            return {"error": "conversation_id and text required"}
        node_id = tree.add_node(conv_id, role=role, text=text, parent_id=parent_id or None)
        return {
            "node_id": node_id,
            "branch_index": tree.get_branch_index(node_id),
        }


# ======================================================================
# Social / Reddit Lead Hunter routes
# ======================================================================


def _register_social_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Reddit Lead Hunter."""

    def _get_service() -> Any:
        # Prefer the new source-agnostic LeadService; fall back to legacy alias.
        return (
            (
                getattr(gateway, "_leads_service", None)
                or getattr(gateway, "_reddit_lead_service", None)
            )
            if gateway
            else None
        )

    @app.get("/api/v1/leads/engine-status", dependencies=deps)
    async def leads_engine_status() -> dict[str, Any]:
        """Return which lead sources are enabled. Frontend uses this to gate the sidebar tab."""
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        if social_cfg is None:
            return {"enabled": False, "sources": {}}
        return {
            "enabled": bool(getattr(social_cfg, "leads_engine_enabled", False)),
            "sources": {
                "reddit": bool(getattr(social_cfg, "reddit_scan_enabled", False)),
                "hackernews": bool(getattr(social_cfg, "hn_enabled", False)),
                "discord": bool(getattr(social_cfg, "discord_scanner_enabled", False)),
                "rss": bool(getattr(social_cfg, "rss_enabled", False)),
            },
        }

    @app.get("/api/v1/packs/loaded", dependencies=deps)
    async def list_loaded_packs() -> dict[str, Any]:
        """Return currently loaded packs for Flutter tab gating."""
        loader = getattr(gateway, "_pack_loader", None)
        if loader is None:
            return {"packs": []}
        return {
            "packs": [
                {
                    "qualified_id": p.manifest.qualified_id,
                    "version": p.manifest.version,
                    "display_name": p.manifest.display_name,
                    "tools": p.manifest.tools,
                    "lead_sources": p.manifest.lead_sources,
                }
                for p in loader.loaded()
            ]
        }

    @app.get("/api/v1/leads/sources", dependencies=deps)
    async def list_lead_sources() -> dict[str, Any]:
        """Return registered LeadSource metadata.

        Feeds Flutter's LeadsScreen + locked-pack-card UX. An empty list
        means the backend has no lead sources and the sidebar tab should
        be hidden by the frontend.
        """
        svc = _get_service()
        if svc is None:
            return {"sources": []}
        # LeadService.list_sources() returns registered LeadSource instances.
        sources: list[dict[str, Any]] = []
        try:
            for source in svc.list_sources():
                sources.append(
                    {
                        "source_id": source.source_id,
                        "display_name": source.display_name,
                        "icon": getattr(source, "icon", ""),
                        "color": getattr(source, "color", ""),
                        "capabilities": sorted(getattr(source, "capabilities", [])),
                    }
                )
        except Exception:
            pass
        return {"sources": sources}

    @app.post("/api/v1/leads/scan/rss", dependencies=deps)
    async def scan_leads_rss(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        feeds = body.get("feeds") or (
            list(getattr(social_cfg, "rss_feeds", [])) if social_cfg else []
        )
        min_score = int(
            body.get("min_score")
            or (getattr(social_cfg, "rss_min_score", 60) if social_cfg else 60)
        )
        if not feeds:
            return {"error": "No RSS feeds configured", "leads_found": 0, "posts_checked": 0}
        # Delegate to the rss-lead-hunter pack source (source_id="rss") if registered.
        try:
            result = await svc.scan(
                source_id="rss",
                min_score=min_score,
                config={"rss": {"feeds": feeds}},
                trigger="ui",
            )
            return {
                "id": result.id,
                "summary": result.summary(),
                "posts_checked": result.posts_checked,
                "leads_found": result.leads_found,
            }
        except ValueError:
            return {
                "error": "RSS source not registered — install rss-lead-hunter pack",
                "leads_found": 0,
                "posts_checked": 0,
            }

    @app.post("/api/v1/leads/scan", dependencies=deps)
    async def scan_leads(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        min_score = int(body.get("min_score") or 0)
        source_id = body.get("source_id") or body.get("platform") or None
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        product = body.get("product") or getattr(social_cfg, "reddit_product_name", "") or ""
        result = await svc.scan(
            source_id=source_id,
            min_score=min_score,
            product=product,
            trigger="ui",
        )
        return {
            "id": result.id,
            "summary": result.summary(),
            "posts_checked": result.posts_checked,
            "leads_found": result.leads_found,
        }

    @app.get("/api/v1/leads", dependencies=deps)
    async def list_leads(
        status: str | None = None,
        min_score: int = 0,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        from cognithor.leads.models import LeadStatus

        status_filter = None
        if status and status in [s.value for s in LeadStatus]:
            status_filter = LeadStatus(status)
        leads = svc.get_leads(status=status_filter, min_score=min_score, limit=limit, offset=offset)
        return {
            "leads": [l.to_dict() for l in leads],
            "count": len(leads),
        }

    @app.get("/api/v1/leads/stats", dependencies=deps)
    async def lead_stats() -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        stats = svc.get_stats()
        history = svc.get_scan_history(limit=10)
        return {
            "stats": {
                "total": stats.total,
                "new": stats.new,
                "reviewed": stats.reviewed,
                "replied": stats.replied,
                "archived": stats.archived,
                "avg_score": stats.avg_score,
                "top_subreddits": stats.top_subreddits,
                "total_scans": stats.total_scans,
            },
            "recent_scans": history,
        }

    @app.post("/api/v1/leads/discover-subreddits", dependencies=deps)
    async def discover_subreddits(request: Request) -> dict[str, Any]:
        # Subreddit discovery is provided by the reddit-lead-hunter-pro pack.
        # Without the pack installed, this endpoint returns an empty suggestion list.
        return {
            "suggestions": [],
            "note": "Install reddit-lead-hunter-pro pack to enable subreddit discovery.",
        }

    @app.get("/api/v1/leads/templates", dependencies=deps)
    async def list_templates(subreddit: str = "") -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        return {"templates": svc.get_templates(subreddit)}

    @app.post("/api/v1/leads/templates", dependencies=deps)
    async def create_template(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        tid = svc.create_template(
            name=body.get("name", ""),
            text=body.get("text", ""),
            subreddit=body.get("subreddit", ""),
            style=body.get("style", ""),
        )
        return {"id": tid, "status": "created"}

    @app.delete("/api/v1/leads/templates/{template_id}", dependencies=deps)
    async def delete_template(template_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        svc.delete_template(template_id)
        return {"status": "deleted"}

    @app.get("/api/v1/leads/{lead_id}", dependencies=deps)
    async def get_lead(lead_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        lead = svc.get_lead(lead_id)
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return lead.to_dict()

    @app.patch("/api/v1/leads/{lead_id}", dependencies=deps)
    async def update_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        from cognithor.leads.models import LeadStatus

        status = LeadStatus(body["status"]) if "status" in body else None
        reply_final = body.get("reply_final")
        lead = svc.update_lead(lead_id, status=status, reply_final=reply_final)
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return lead.to_dict()

    @app.post("/api/v1/leads/{lead_id}/reply", dependencies=deps)
    async def reply_to_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = body.get("mode", "clipboard")
        result = svc.post_reply(lead_id, mode=mode)
        return {
            "success": result.success,
            "mode": result.mode.value if hasattr(result.mode, "value") else result.mode,
            "error": result.error,
        }

    @app.post("/api/v1/leads/{lead_id}/refine", dependencies=deps)
    async def refine_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        hint = body.get("hint", "")
        variants = body.get("variants", 0)
        result = await svc.refine_reply(lead_id, hint=hint, variants=variants)
        if result is None:
            raise HTTPException(404, "Lead not found")
        if isinstance(result, list):
            return {"variants": [{"text": r.text, "style": r.style} for r in result]}
        return {"text": result.text, "style": result.style, "changes": result.changes_summary}

    @app.get("/api/v1/leads/{lead_id}/performance", dependencies=deps)
    async def get_lead_performance(lead_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        perf = svc.get_performance(lead_id)
        if perf is None:
            return {"performance": None}

        def _engagement_score(upvotes: int, replies: int, author_replied: bool, tag: str) -> float:
            """Inline engagement score — simple weighted heuristic."""
            score = min(upvotes * 0.3 + replies * 0.5, 80.0)
            if author_replied:
                score += 15.0
            if tag == "good":
                score += 5.0
            elif tag == "bad":
                score -= 10.0
            return round(max(0.0, min(score, 100.0)), 1)

        perf["engagement_score"] = _engagement_score(
            perf.get("reply_upvotes", 0),
            perf.get("reply_replies", 0),
            bool(perf.get("author_replied", 0)),
            perf.get("feedback_tag", ""),
        )
        return {"performance": perf}

    @app.patch("/api/v1/leads/{lead_id}/feedback", dependencies=deps)
    async def set_lead_feedback(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        svc.set_feedback(lead_id, tag=body.get("tag", ""), note=body.get("note", ""))
        return {"status": "ok"}
