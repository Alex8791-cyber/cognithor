"""Cognithor · Learning + Knowledge-Ingestion routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Bundle aus
zwei verwandten Helfern fuer aktives Lernen und Knowledge-Ingestion:

  - `_register_learning_routes()` — Active Learning, Curiosity-Engine,
    Confidence-Manager, Reflection-Engine.
  - `_register_ingest_routes()` — Knowledge-Ingestion (File-Upload,
    URL, YouTube, Job-Status, Queue).
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
