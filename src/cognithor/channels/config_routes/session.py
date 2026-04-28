"""Cognithor · Session + Memory routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_session_routes()` (Vault, Session-Isolation, Workspace-Isolation,
Multi-Tenancy) und `_register_memory_routes()` (Memory-Hygiene,
Integrity-Checker, Decision-Explainer / Explainability).
"""

from __future__ import annotations

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
# Session management routes (vault, session-isolation, isolation)
# ======================================================================


def _register_session_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Vault, session-isolation, workspace-isolation, multi-tenant."""

    # -- Vault & Session-Isolation ----------------------------------------

    @app.get("/api/v1/vault/stats", dependencies=deps)
    async def vault_stats() -> dict[str, Any]:
        """Vault-Manager Statistiken."""
        mgr = getattr(gateway, "_vault_manager", None)
        if mgr is None:
            return {"total_vaults": 0, "agents": [], "total_entries": 0}
        return mgr.stats()

    @app.get("/api/v1/vault/agents", dependencies=deps)
    async def vault_agents() -> dict[str, Any]:
        """Alle Agent-Vaults auflisten."""
        mgr = getattr(gateway, "_vault_manager", None)
        if mgr is None:
            return {"agents": []}
        return {"agents": [v.stats() for v in mgr._vaults.values()]}

    @app.get("/api/v1/sessions/stats", dependencies=deps)
    async def session_stats() -> dict[str, Any]:
        """Session-Store Statistiken."""
        store = getattr(gateway, "_isolated_sessions", None)
        if store is None:
            return {"total_agents": 0, "total_sessions": 0, "active_sessions": 0}
        return store.stats()

    @app.get("/api/v1/sessions/guard/violations", dependencies=deps)
    async def guard_violations() -> dict[str, Any]:
        """Session-Guard Violations."""
        guard = getattr(gateway, "_session_guard", None)
        if guard is None:
            return {"violations": [], "count": 0}
        v = guard.violations()
        return {"violations": v, "count": len(v)}

    # -- Multi-User Isolation (Phase 8) -----------------------------------

    @app.get("/api/v1/isolation/stats", dependencies=deps)
    async def isolation_stats_core() -> dict[str, Any]:
        """Isolation-Statistiken (core)."""
        try:
            from cognithor.core.isolation import MultiUserIsolation

            iso = MultiUserIsolation()
            return iso.stats()
        except Exception as exc:
            log.error("isolation_stats_failed", error=str(exc))
            return {"error": "Isolation-Statistiken nicht verfuegbar"}

    @app.get("/api/v1/isolation/quotas", dependencies=deps)
    async def isolation_quotas() -> dict[str, Any]:
        """Quota-Uebersicht aller Agents."""
        try:
            from cognithor.core.isolation import MultiUserIsolation

            iso = MultiUserIsolation()
            return {"quotas": iso.all_quota_summaries()}
        except Exception as exc:
            log.error("isolation_quotas_failed", error=str(exc))
            return {"error": "Quota-Uebersicht nicht verfuegbar"}

    @app.get("/api/v1/isolation/violations", dependencies=deps)
    async def isolation_violations() -> dict[str, Any]:
        """Workspace-Violations."""
        try:
            from cognithor.core.isolation import WorkspaceGuard

            guard = WorkspaceGuard()
            return {"violations": guard.violations, "count": guard.violation_count}
        except Exception as exc:
            log.error("isolation_violations_failed", error=str(exc))
            return {"error": "Violations konnten nicht geladen werden"}

    # -- Sandbox-Isolierung + Multi-Tenant (Phase 25) ---------------------

    @app.get("/api/v1/isolation/sandboxes", dependencies=deps)
    async def isolation_sandboxes() -> dict[str, Any]:
        """Laufende Sandboxes."""
        enforcer = getattr(gateway, "_isolation_enforcer", None)
        if enforcer is None:
            return {"sandboxes": []}
        return {"sandboxes": [sb.to_dict() for sb in enforcer.sandboxes.running()]}

    @app.get("/api/v1/isolation/tenants", dependencies=deps)
    async def isolation_tenants() -> dict[str, Any]:
        """Tenant-Uebersicht."""
        enforcer = getattr(gateway, "_isolation_enforcer", None)
        if enforcer is None:
            return {"total_tenants": 0}
        return enforcer.tenants.stats()

    @app.get("/api/v1/isolation/secrets", dependencies=deps)
    async def isolation_secrets() -> dict[str, Any]:
        """Secret-Vault Statistiken."""
        enforcer = getattr(gateway, "_isolation_enforcer", None)
        if enforcer is None:
            return {"total_secrets": 0}
        return enforcer.secrets.stats()

    # -- Per-Agent Vault & Session-Isolation (Phase 29) -------------------

    @app.get("/api/v1/vaults/stats", dependencies=deps)
    async def vaults_stats() -> dict[str, Any]:
        """Agent-Vault Statistiken."""
        vm = getattr(gateway, "_vault_manager", None)
        if vm is None:
            return {"total_vaults": 0}
        return vm.stats()

    @app.get("/api/v1/vaults/sessions", dependencies=deps)
    async def vaults_sessions() -> dict[str, Any]:
        """Session-Isolation Status."""
        vm = getattr(gateway, "_vault_manager", None)
        if vm is None:
            return {"agent_stores": 0}
        return vm.sessions.stats()

    @app.get("/api/v1/vaults/firewall", dependencies=deps)
    async def vaults_firewall() -> dict[str, Any]:
        """Session-Firewall Status."""
        vm = getattr(gateway, "_vault_manager", None)
        if vm is None:
            return {"total_violations": 0}
        return vm.firewall.stats()

    # -- Chat-History API (fuer WebUI Sidebar) ------------------------------

    def _get_session_store() -> Any:
        """Zugriff auf den SessionStore des Gateways."""
        return getattr(gateway, "_session_store", None)

    @app.get("/api/v1/sessions/list", dependencies=deps)
    async def list_sessions(channel: str = "webui", limit: int = 50) -> dict[str, Any]:
        """Aktive Sessions fuer die Chat-History-Sidebar auflisten."""
        store = _get_session_store()
        if not store:
            return {"sessions": []}
        sessions = store.list_sessions_for_channel(channel=channel, limit=limit)
        return {"sessions": sessions}

    @app.get("/api/v1/sessions/{session_id}/history", dependencies=deps)
    async def get_session_history(session_id: str, limit: int = 100) -> dict[str, Any]:
        """Chat-Messages einer bestimmten Session abrufen."""
        store = _get_session_store()
        if not store:
            return {"messages": [], "session_id": session_id}
        messages = store.get_session_history(session_id, limit=limit)
        return {"messages": messages, "session_id": session_id}

    @app.get("/api/v1/sessions/{session_id}/lineage", dependencies=deps)
    async def get_session_lineage(session_id: str) -> dict[str, Any]:
        """Reconstruct session fork chain to root (Phase 2 Provenance)."""
        store = _get_session_store()
        if not store:
            return {"lineage": [session_id], "session_id": session_id}
        chain = [session_id]
        current_id = session_id
        for _ in range(20):  # Max depth guard
            meta = (
                store.get_session_metadata(current_id)
                if hasattr(store, "get_session_metadata")
                else None
            )
            if meta is None:
                break
            parent = (
                meta.get("parent_session_id", "")
                if isinstance(meta, dict)
                else getattr(meta, "parent_session_id", "")
            )
            if not parent:
                break
            chain.append(parent)
            current_id = parent
        chain.reverse()
        return {"lineage": chain, "session_id": session_id}

    @app.get("/api/v1/sessions/{session_id}/export", dependencies=deps)
    async def export_session(session_id: str) -> dict[str, Any]:
        """Export session chat history as JSON."""
        store = _get_session_store()
        if not store:
            return {"error": "Store not available"}
        return store.export_session(session_id)

    @app.get("/api/v1/sessions/folders", dependencies=deps)
    async def list_folders(channel: str = "webui") -> dict[str, Any]:
        """Eindeutige Ordnernamen auflisten."""
        store = _get_session_store()
        if not store:
            return {"folders": []}
        folders = store.list_folders(channel=channel)
        return {"folders": folders}

    @app.get("/api/v1/sessions/by-folder/{folder}", dependencies=deps)
    async def list_sessions_by_folder(folder: str, limit: int = 50) -> dict[str, Any]:
        """List sessions filtered by project/folder."""
        store = _get_session_store()
        if not store:
            return {"sessions": []}
        sessions = store.list_sessions_by_folder(folder, limit=limit)
        return {"sessions": sessions}

    @app.patch("/api/v1/sessions/{session_id}", dependencies=deps)
    async def update_session(session_id: str, request: Request) -> dict[str, Any]:
        """Session-Metadaten aktualisieren (Titel, Ordner)."""
        body = await request.json()
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        title = body.get("title")
        if title is not None:
            store.update_session_title(session_id, title)
        folder = body.get("folder")
        if folder is not None:
            store.update_session_folder(session_id, folder)
        return {"status": "updated", "session_id": session_id}

    @app.delete("/api/v1/sessions/{session_id}", dependencies=deps)
    async def delete_session(session_id: str) -> dict[str, Any]:
        """Soft-Delete einer Session (active=0)."""
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        store.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}

    @app.post("/api/v1/sessions/new", dependencies=deps)
    async def create_new_session() -> dict[str, Any]:
        """Neue leere Session erstellen und ID zurueckgeben."""
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        import uuid

        from cognithor.models import SessionContext

        session_id = uuid.uuid4().hex[:16]
        store.save_session(
            SessionContext(
                session_id=session_id,
                channel="webui",
                user_id="web_user",
                agent_name="jarvis",
            )
        )
        return {"session_id": session_id}

    @app.post("/api/v1/sessions/new-incognito", dependencies=deps)
    async def create_incognito_session() -> dict[str, Any]:
        """Neue Inkognito-Session erstellen (kein Memory, keine Persistierung)."""
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        import uuid

        from cognithor.models import SessionContext

        session_id = uuid.uuid4().hex[:16]
        store.save_session(
            SessionContext(
                session_id=session_id,
                channel="webui",
                user_id="web_user",
                agent_name="jarvis",
                incognito=True,
            )
        )
        return {"session_id": session_id, "incognito": True}

    @app.get("/api/v1/sessions/search", dependencies=deps)
    async def search_sessions(q: str = "", limit: int = 20) -> dict[str, Any]:
        """Full-text search across all chat sessions."""
        store = _get_session_store()
        if not store or not q.strip():
            return {"results": [], "query": q}
        results = store.search_chat_history(q.strip(), limit=limit)
        return {"results": results, "query": q}

    @app.get("/api/v1/sessions/should-new", dependencies=deps)
    async def should_new_session(
        channel: str = "webui",
        timeout_minutes: int = 30,
    ) -> dict[str, Any]:
        """Check if client should start a new session due to inactivity."""
        store = _get_session_store()
        if not store:
            return {"should_new": True}
        should_new = store.should_create_new_session(
            channel=channel,
            user_id="web_user",
            inactivity_timeout_minutes=timeout_minutes,
        )
        return {"should_new": should_new}


# ======================================================================
# Memory / search routes
# ======================================================================


def _register_memory_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Memory-hygiene, memory-integrity, explainability."""

    # -- Memory-Hygiene ---------------------------------------------------

    @app.post("/api/v1/memory/hygiene/scan", dependencies=deps)
    async def memory_hygiene_scan(request: Request) -> dict[str, Any]:
        """Memory-Eintraege auf Injection/Credentials/Widersprueche scannen."""
        try:
            from cognithor.memory.hygiene import MemoryHygieneEngine

            engine = getattr(gateway, "_memory_hygiene", None) or MemoryHygieneEngine()
            body = await request.json()
            entries = body.get("entries", [])
            auto_quarantine = body.get("auto_quarantine", True)
            report = engine.scan_batch(entries, auto_quarantine=auto_quarantine)
            return report.to_dict()
        except Exception:
            log.exception("Error during memory hygiene scan")
            return {"error": "Internal error during memory hygiene scan"}

    @app.get("/api/v1/memory/hygiene/stats", dependencies=deps)
    async def memory_hygiene_stats() -> dict[str, Any]:
        """Memory-Hygiene Statistiken."""
        engine = getattr(gateway, "_memory_hygiene", None)
        if engine is None:
            return {
                "total_scans": 0,
                "total_scanned": 0,
                "total_threats": 0,
                "quarantined": 0,
                "threat_rate": 0.0,
            }
        return engine.stats()

    @app.get("/api/v1/memory/hygiene/quarantine", dependencies=deps)
    async def memory_quarantine() -> dict[str, Any]:
        """Quarantaene-Liste."""
        engine = getattr(gateway, "_memory_hygiene", None)
        if engine is None:
            return {"quarantined": []}
        return {"quarantined": engine.quarantine()}

    # -- Memory-Integritaet (Phase 26) ------------------------------------

    @app.get("/api/v1/memory/integrity", dependencies=deps)
    async def memory_integrity() -> dict[str, Any]:
        """Memory-Integritaets-Status."""
        checker = getattr(gateway, "_integrity_checker", None)
        if checker is None:
            return {"total_checks": 0, "last_score": 100}
        return checker.stats()

    @app.get("/api/v1/memory/explainability", dependencies=deps)
    async def memory_explainability() -> dict[str, Any]:
        """Decision-Explainer Statistiken."""
        explainer = getattr(gateway, "_decision_explainer", None)
        if explainer is None:
            return {"total_explanations": 0}
        return explainer.stats()

    # -- Explainability ---------------------------------------------------

    @app.get("/api/v1/explainability/trails", dependencies=deps)
    async def explainability_trails() -> dict[str, Any]:
        """Letzte Decision-Trails."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {"trails": [], "count": 0}
        trails = engine.recent_trails(limit=20)
        return {"trails": [t.to_dict() for t in trails], "count": len(trails)}

    @app.get("/api/v1/explainability/stats", dependencies=deps)
    async def explainability_stats() -> dict[str, Any]:
        """Explainability-Engine Statistiken."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {
                "total_requests": 0,
                "active_trails": 0,
                "completed_trails": 0,
                "avg_confidence": 0.0,
            }
        return engine.stats()

    @app.get("/api/v1/explainability/low-trust", dependencies=deps)
    async def explainability_low_trust() -> dict[str, Any]:
        """Trails mit niedrigem Trust-Score."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {"low_trust_trails": [], "count": 0}
        trails = engine.low_trust_trails(threshold=0.5)
        return {"low_trust_trails": [t.to_dict() for t in trails], "count": len(trails)}

    # -- Knowledge Graph --------------------------------------------------

    @app.get("/api/v1/memory/graph/stats", dependencies=deps)
    async def knowledge_graph_stats() -> dict[str, Any]:
        """Wissensgraph-Statistiken."""
        semantic = getattr(gateway, "_semantic_memory", None)
        if semantic is None:
            return {"entities": 0, "relations": 0, "entity_types": {}}
        try:
            entities = getattr(semantic, "entities", None) or {}
            relations = getattr(semantic, "relations", None) or []
            type_counts: dict[str, int] = {}
            for e in entities.values() if isinstance(entities, dict) else entities:
                etype = getattr(e, "type", None) or getattr(e, "entity_type", "unknown")
                type_counts[etype] = type_counts.get(etype, 0) + 1
            return {
                "entities": len(entities),
                "relations": len(relations),
                "entity_types": type_counts,
            }
        except Exception:
            return {"entities": 0, "relations": 0, "entity_types": {}}

    @app.get("/api/v1/memory/graph/entities", dependencies=deps)
    async def knowledge_graph_entities() -> dict[str, Any]:
        """Alle Entitaeten und Beziehungen im Wissensgraph."""
        semantic = getattr(gateway, "_semantic_memory", None)
        if semantic is None:
            return {"entities": [], "relations": []}
        try:
            raw_entities = getattr(semantic, "entities", None) or {}
            raw_relations = getattr(semantic, "relations", None) or []

            entities = []
            for eid, e in (
                raw_entities.items() if isinstance(raw_entities, dict) else enumerate(raw_entities)
            ):
                entity_id = str(getattr(e, "id", eid))
                entities.append(
                    {
                        "id": entity_id,
                        "name": getattr(e, "name", str(e)),
                        "type": getattr(e, "type", None) or getattr(e, "entity_type", "unknown"),
                        "confidence": getattr(e, "confidence", 0.5),
                        "attributes": getattr(e, "attributes", {}),
                    }
                )

            relations = []
            for r in raw_relations:
                relations.append(
                    {
                        "source_entity": str(
                            getattr(r, "source_entity", getattr(r, "source_name", ""))
                        ),
                        "target_entity": str(
                            getattr(r, "target_entity", getattr(r, "target_name", ""))
                        ),
                        "relation_type": str(getattr(r, "relation_type", "related_to")),
                        "confidence": getattr(r, "confidence", 0.5),
                    }
                )

            return {"entities": entities, "relations": relations}
        except Exception:
            return {"entities": [], "relations": []}

    @app.get("/api/v1/memory/graph/entities/{entity_id}/relations", dependencies=deps)
    async def knowledge_graph_entity_relations(entity_id: str) -> dict[str, Any]:
        """Beziehungen einer bestimmten Entitaet."""
        semantic = getattr(gateway, "_semantic_memory", None)
        if semantic is None:
            return {"relations": []}
        try:
            raw_relations = getattr(semantic, "relations", None) or []
            entity_rels = []
            for r in raw_relations:
                src = str(getattr(r, "source_entity", getattr(r, "source_name", "")))
                tgt = str(getattr(r, "target_entity", getattr(r, "target_name", "")))
                if src == entity_id or tgt == entity_id:
                    entity_rels.append(
                        {
                            "source_entity": src,
                            "target_entity": tgt,
                            "target_name": tgt,
                            "relation_type": str(getattr(r, "relation_type", "related_to")),
                            "confidence": getattr(r, "confidence", 0.5),
                        }
                    )
            return {"relations": entity_rels}
        except Exception:
            return {"relations": []}
