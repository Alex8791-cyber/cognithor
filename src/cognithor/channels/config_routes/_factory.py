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

import contextlib
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
from cognithor.channels.config_routes.evolution import (
    _register_gepa_evolution_routes,
    _register_prompt_evolution_routes,
    _register_self_improvement_routes,
)
from cognithor.channels.config_routes.governance import _register_governance_routes
from cognithor.channels.config_routes.infrastructure import (
    _register_backend_routes,
    _register_infrastructure_routes,
    _register_portal_routes,
)
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
from cognithor.channels.config_routes.workflows import _register_workflow_graph_routes

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
