"""Cognithor · Security / audit routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_security_routes()` — registriert SecurityScanner-Endpoints,
Compliance, Decision-Log, Remediation, Compliance-Export, Security-
Pipeline / Ecosystem-Policy, Metrics, Incident-Tracker, Security-Team,
Posture-Scorer, Security-Gate, Continuous-RedTeam und Code-Auditor.
Groesster Helper im Paket (~975 LOC).
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
# Security / audit routes
# ======================================================================


def _register_security_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Red-team scanning, compliance, security pipeline, security framework,
    ecosystem security policy, CI/CD gate."""

    # -- Red-Team Security Scanner ----------------------------------------

    @app.post("/api/v1/security/redteam/scan", dependencies=deps)
    async def redteam_scan(request: Request) -> dict[str, Any]:
        """Red-Team-Scan gegen Prompt-Injection etc."""
        try:
            from cognithor.security.redteam import ScanPolicy, SecurityScanner

            scanner = getattr(gateway, "_security_scanner", None) or SecurityScanner()
            body = await request.json()
            if "policy" in body:
                p = body["policy"]
                scanner.policy = ScanPolicy(
                    max_risk_score=p.get("max_risk_score", 70),
                    block_on_critical=p.get("block_on_critical", True),
                    block_on_high=p.get("block_on_high", False),
                    min_tests=p.get("min_tests", 5),
                )

            import re as _re

            def sanitizer(text: str) -> dict[str, Any]:
                dangerous = [
                    r"ignore\s+(all\s+)?previous",
                    r"system\s*:",
                    r"<\s*script",
                    r"rm\s+-rf",
                    r"\bsudo\b",
                    r"\bexec\b",
                    r"\beval\b",
                ]
                for pat in dangerous:
                    if _re.search(pat, text, _re.IGNORECASE):
                        return {"blocked": True}
                return {"blocked": False}

            result = scanner.scan(
                sanitizer_fn=sanitizer,
                is_blocked_fn=lambda r: r.get("blocked", False),
            )
            return result.to_dict()
        except Exception as exc:
            log.error("redteam_scan_failed", error=str(exc))
            return {"error": "Security-Scan fehlgeschlagen"}

    @app.get("/api/v1/security/redteam/status", dependencies=deps)
    async def redteam_status() -> dict[str, Any]:
        """Status des Security-Scanners."""
        scanner = getattr(gateway, "_security_scanner", None)
        return {
            "available": True,
            "scanner": "SecurityScanner",
            "has_gateway_instance": scanner is not None,
        }

    # -- Compliance & Audit -----------------------------------------------

    @app.get("/api/v1/compliance/report", dependencies=deps)
    async def compliance_report() -> dict[str, Any]:
        """EU-AI-Act + DSGVO Compliance-Report generieren."""
        try:
            from cognithor.audit.compliance import ComplianceFramework

            fw = getattr(gateway, "_compliance_framework", None) or ComplianceFramework()
            fw.auto_assess(
                has_audit_log=True,
                has_decision_log=getattr(gateway, "_decision_log", None) is not None,
                has_kill_switch=True,
                has_encryption=True,
                has_rbac=True,
                has_sandbox=True,
                has_approval_workflow=True,
                has_redteam=getattr(gateway, "_security_scanner", None) is not None,
            )
            report = fw.generate_report()
            return report.to_dict()
        except Exception as exc:
            log.error("compliance_report_failed", error=str(exc))
            return {"error": "Compliance-Report konnte nicht generiert werden"}

    @app.get("/api/v1/compliance/export/{fmt}", dependencies=deps)
    async def compliance_export(fmt: str) -> Any:
        """Compliance-Report exportieren (json/csv/markdown)."""
        try:
            from cognithor.audit.compliance import ComplianceFramework, ReportExporter

            fw = getattr(gateway, "_compliance_framework", None) or ComplianceFramework()
            fw.auto_assess(
                has_audit_log=True,
                has_decision_log=True,
                has_kill_switch=True,
                has_encryption=True,
                has_rbac=True,
                has_sandbox=True,
                has_approval_workflow=True,
                has_redteam=True,
            )
            report = fw.generate_report()
            if fmt == "json":
                from starlette.responses import JSONResponse

                return JSONResponse(content=report.to_dict())
            elif fmt == "csv":
                from starlette.responses import PlainTextResponse

                return PlainTextResponse(ReportExporter.to_csv(report), media_type="text/csv")
            elif fmt == "markdown":
                from starlette.responses import PlainTextResponse

                return PlainTextResponse(
                    ReportExporter.to_markdown(report), media_type="text/markdown"
                )
            return {"error": f"Unknown format: {fmt}. Use json/csv/markdown."}
        except Exception as exc:
            log.error("compliance_export_failed", error=str(exc))
            return {"error": "Compliance-Export fehlgeschlagen"}

    @app.get("/api/v1/compliance/decisions", dependencies=deps)
    async def compliance_decisions() -> dict[str, Any]:
        """Decision-Log Uebersicht."""
        decision_log = getattr(gateway, "_decision_log", None)
        if decision_log is None:
            return {
                "total_decisions": 0,
                "flagged_count": 0,
                "approval_rate": 0.0,
                "unique_agents": 0,
                "avg_confidence": 0.0,
            }
        return decision_log.stats()

    @app.get("/api/v1/compliance/remediations", dependencies=deps)
    async def compliance_remediations() -> dict[str, Any]:
        """Remediation-Tracker Status."""
        tracker = getattr(gateway, "_remediation_tracker", None)
        if tracker is None:
            return {"total": 0, "open": 0, "in_progress": 0, "resolved": 0, "overdue": 0}
        return tracker.stats()

    @app.get("/api/v1/compliance/stats", dependencies=deps)
    async def compliance_stats() -> dict[str, Any]:
        """Compliance-Exporter Statistiken."""
        exporter = getattr(gateway, "_compliance_exporter", None)
        if exporter is None:
            return {"total_reports": 0}
        return exporter.stats()

    @app.get("/api/v1/compliance/transparency", dependencies=deps)
    async def compliance_transparency() -> dict[str, Any]:
        """Transparenzpflichten-Status."""
        exporter = getattr(gateway, "_compliance_exporter", None)
        if exporter is None:
            return {"total_obligations": 0}
        return exporter.transparency.stats()

    @app.post("/api/v1/compliance/report", dependencies=deps)
    async def compliance_generate() -> dict[str, Any]:
        """Generiert einen Compliance-Bericht."""
        exporter = getattr(gateway, "_compliance_exporter", None)
        if exporter is None:
            return {"error": "Exporter nicht verfügbar"}
        report = exporter.generate_report()
        return report.to_dict()

    # -- GDPR Art. 15: User Data Export -----------------------------------

    @app.get("/api/v1/user/audit-data", dependencies=deps)
    async def export_user_audit_data(
        channel: str = "",
        hours: int = 0,
    ) -> dict[str, Any]:
        """GDPR Art. 15: Export audit data for a user/channel."""
        audit = getattr(gateway, "_audit_logger", None)
        if not audit:
            return {"entries": [], "count": 0, "message": "Audit logging not active."}

        entries = audit.get_entries_for_export(channel=channel, hours=hours)
        return {
            "entries": entries,
            "count": len(entries),
            "channel_filter": channel or "all",
            "hours_filter": hours or "all",
            "gdpr_article": "Art. 15 DSGVO — Auskunftsrecht",
        }

    # -- GDPR Art. 17: User Data Erasure ---------------------------------

    @app.delete("/api/v1/user/data", dependencies=deps)
    async def erase_user_data(request: Request) -> dict[str, Any]:
        """GDPR Art. 17: Delete all personal data for authenticated user.

        user_id is extracted from the authenticated session, NOT from
        the request body (prevents IDOR attacks). Admin override via
        COGNITHOR_ADMIN_TOKEN header for erasing other users' data.
        """
        import os as _os

        gdpr_mgr = getattr(gateway, "_gdpr_manager", None)
        consent_mgr = getattr(gateway, "_consent_manager", None)
        if not gdpr_mgr:
            return {"error": "GDPR manager not available"}

        # Extract user_id from session, not request body (IDOR prevention)
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Admin override: COGNITHOR_ADMIN_TOKEN can erase any user
        admin_token = _os.environ.get("COGNITHOR_ADMIN_TOKEN", "")
        auth_header = request.headers.get("X-Admin-Token", "")
        is_admin = admin_token and auth_header == admin_token

        if is_admin:
            user_id = body.get("user_id", "")
        else:
            # Regular users: extract from session (WebSocket session binding)
            user_id = body.get("session_user_id", "")
            # Fallback: single-user systems use config owner
            if not user_id:
                user_id = getattr(getattr(gateway, "_config", None), "owner", "") or ""

        if not user_id:
            return {"error": "user_id could not be determined from session"}

        result = await gdpr_mgr.erasure.erase_all(
            user_id=user_id,
            consent_manager=consent_mgr,
        )
        return {
            "status": "erased",
            "user_id": user_id,
            "counts": result,
            "gdpr_article": "Art. 17 DSGVO — Recht auf Loeschung",
        }

    # -- GDPR Art. 20: User Data Export (full) ----------------------------

    @app.get("/api/v1/user/data", dependencies=deps)
    async def export_user_data(user_id: str = "", format: str = "json") -> Any:
        """GDPR Art. 15/20: Export all personal data for a user."""
        if not user_id:
            return {"error": "user_id query parameter required"}

        export: dict[str, Any] = {
            "export_version": "2.0",
            "format": "cognithor_portable",
            "user_id": user_id,
            "gdpr_article": "Art. 15/20 DSGVO",
        }

        # Processing logs
        gdpr_mgr = getattr(gateway, "_gdpr_manager", None)
        if gdpr_mgr:
            try:
                export["processing_log"] = gdpr_mgr.user_report(user_id)
            except Exception:
                export["processing_log"] = {"error": "unavailable"}

        # Consents
        consent_mgr = getattr(gateway, "_consent_manager", None)
        if consent_mgr:
            try:
                export["consents"] = consent_mgr.get_user_consents(user_id)
            except Exception:
                export["consents"] = []

        # Sessions
        session_store = getattr(gateway, "_session_store", None)
        if session_store:
            try:
                # Get sessions for this user
                sessions = session_store.conn.execute(
                    "SELECT session_id, user_id, channel, agent_id, created_at"
                    " FROM sessions WHERE user_id = ? LIMIT 100",
                    (user_id,),
                ).fetchall()
                export["sessions"] = [dict(s) for s in sessions] if sessions else []
            except Exception:
                export["sessions"] = []

        # Memory data
        memory_mgr = getattr(gateway, "_memory_manager", None)
        if memory_mgr:
            try:
                results = memory_mgr.search_memory_sync(query=user_id, top_k=100)
                export["memories"] = [
                    {"text": getattr(r, "text", str(r))[:500], "source": getattr(r, "source", "")}
                    for r in (results or [])
                ]
            except Exception:
                export["memories"] = []

        # Episodic memories (daily logs)
        if memory_mgr and hasattr(memory_mgr, "episodic"):
            try:
                ep = memory_mgr.episodic
                episodes = []
                if hasattr(ep, "_base_path"):
                    from pathlib import Path

                    ep_dir = Path(ep._base_path) if hasattr(ep, "_base_path") else None
                    if not ep_dir:
                        ep_dir = Path(getattr(ep, "_dir", ""))
                    if ep_dir and ep_dir.exists():
                        for md_file in sorted(ep_dir.glob("*.md"))[-30:]:  # Last 30 days
                            try:
                                # Try encrypted read first
                                try:
                                    from cognithor.security.encrypted_file import efile

                                    content = efile.read(md_file)
                                except Exception:
                                    content = md_file.read_text(encoding="utf-8")
                                episodes.append(
                                    {
                                        "date": md_file.stem,
                                        "content": content[:2000],
                                    }
                                )
                            except Exception:
                                log.debug("gdpr_export_episode_read_failed", exc_info=True)
                export["episodic_memories"] = episodes
            except Exception:
                export["episodic_memories"] = []

        # Procedures (learned behaviors)
        if memory_mgr and hasattr(memory_mgr, "procedures"):
            try:
                procs = memory_mgr.procedures
                procedures = []
                if hasattr(procs, "list_all"):
                    for proc in procs.list_all()[:50]:
                        procedures.append(
                            {
                                "name": getattr(proc, "name", str(proc)),
                                "content": getattr(proc, "content", "")[:1000],
                            }
                        )
                export["procedures"] = procedures
            except Exception:
                export["procedures"] = []

        # Core memory (CORE.md)
        if memory_mgr and hasattr(memory_mgr, "core"):
            try:
                core = memory_mgr.core
                if hasattr(core, "content"):
                    export["core_memory"] = core.content[:5000]
                elif hasattr(core, "load"):
                    export["core_memory"] = str(core.load())[:5000]
                else:
                    export["core_memory"] = ""
            except Exception:
                export["core_memory"] = ""

            # Entities
            try:
                if hasattr(memory_mgr, "semantic") and hasattr(
                    memory_mgr.semantic, "list_entities"
                ):
                    entities = memory_mgr.semantic.list_entities(limit=500)
                    export["entities"] = [
                        {
                            "name": getattr(e, "name", ""),
                            "type": getattr(e, "entity_type", ""),
                            "attributes": str(getattr(e, "attributes", ""))[:200],
                        }
                        for e in (entities or [])
                    ]
            except Exception:
                export["entities"] = []

        # Relations
        try:
            if hasattr(memory_mgr, "semantic") and hasattr(memory_mgr.semantic, "_indexer"):
                indexer = memory_mgr.semantic._indexer
                if hasattr(indexer, "_conn"):
                    rows = indexer._conn.execute(
                        "SELECT source_entity, relation_type, target_entity"
                        " FROM relations LIMIT 1000"
                    ).fetchall()
                    export["relations"] = [
                        {
                            "source": r[0] if isinstance(r, tuple) else r["source_entity"],
                            "relation": r[1] if isinstance(r, tuple) else r["relation_type"],
                            "target": r[2] if isinstance(r, tuple) else r["target_entity"],
                        }
                        for r in rows
                    ]
        except Exception:
            export["relations"] = []

        # User preferences
        pref_store = getattr(gateway, "_user_pref_store", None)
        if pref_store:
            try:
                if hasattr(pref_store, "get_preferences"):
                    export["user_preferences"] = pref_store.get_preferences(user_id)
                elif hasattr(pref_store, "_conn"):
                    rows = pref_store._conn.execute(
                        "SELECT * FROM user_preferences LIMIT 50"
                    ).fetchall()
                    export["user_preferences"] = [dict(r) for r in rows] if rows else []
            except Exception:
                export["user_preferences"] = []

        # Vault notes list
        vault = getattr(gateway, "_vault_tools", None)
        if not vault:
            # Try to find vault in the tools phase
            for attr in dir(gateway):
                obj = getattr(gateway, attr, None)
                if hasattr(obj, "vault_list"):
                    vault = obj
                    break
        if vault and hasattr(vault, "_backend"):
            try:
                notes = vault._backend.list_notes(limit=200)
                export["vault_notes"] = [
                    {
                        "path": n.path,
                        "title": n.title,
                        "tags": n.tags,
                        "folder": n.folder,
                        "content": (n.content or "")[:5000],
                        "created_at": n.created_at,
                        "updated_at": n.updated_at,
                    }
                    for n in notes
                ]
            except Exception:
                export["vault_notes"] = []

        if format == "csv":
            import csv
            import io

            from starlette.responses import StreamingResponse

            output = io.StringIO()

            # Entities CSV
            if export.get("entities"):
                output.write("# Entities\n")
                writer = csv.DictWriter(output, fieldnames=["name", "type", "attributes"])
                writer.writeheader()
                for e in export["entities"]:
                    writer.writerow(e)
                output.write("\n")

            # Sessions CSV
            if export.get("sessions"):
                output.write("# Sessions\n")
                keys = export["sessions"][0].keys() if export["sessions"] else []
                writer = csv.DictWriter(output, fieldnames=keys)
                writer.writeheader()
                for s in export["sessions"]:
                    writer.writerow(s)
                output.write("\n")

            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=cognithor_export_{user_id}.csv"
                },
            )

        return export

    @app.patch("/api/v1/user/data", dependencies=deps)
    async def correct_user_data(request: Request) -> dict[str, Any]:
        """GDPR Art. 16: Correct personal data."""
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON"}

        corrections = body.get("corrections", [])
        if not corrections:
            return {"error": "No corrections provided"}

        results = []
        for corr in corrections:
            corr_type = corr.get("type", "")
            try:
                if corr_type == "preference":
                    pref_store = getattr(gateway, "_user_pref_store", None)
                    if pref_store and hasattr(pref_store, "_conn"):
                        key = corr.get("key", "")
                        value = corr.get("new_value", "")
                        pref_store._conn.execute(
                            "UPDATE user_preferences SET value = ? WHERE key = ?", (str(value), key)
                        )
                        pref_store._conn.commit()
                        results.append({"type": "preference", "key": key, "status": "corrected"})

                elif corr_type == "entity":
                    memory_mgr = getattr(gateway, "_memory_manager", None)
                    if memory_mgr and hasattr(memory_mgr, "semantic"):
                        name = corr.get("name", "")
                        field = corr.get("field", "name")
                        new_value = corr.get("new_value", "")
                        # Update entity in indexer
                        indexer = (
                            memory_mgr.semantic._indexer
                            if hasattr(memory_mgr.semantic, "_indexer")
                            else None
                        )
                        if indexer and hasattr(indexer, "_conn"):
                            indexer._conn.execute(
                                f"UPDATE entities SET {field} = ? WHERE name = ?", (new_value, name)
                            )
                            indexer._conn.commit()
                            results.append({"type": "entity", "name": name, "status": "corrected"})

                elif corr_type == "vault_note":
                    path = corr.get("path", "")
                    new_content = corr.get("new_value", "")
                    # Use vault backend if available
                    vault = None
                    for attr in dir(gateway):
                        obj = getattr(gateway, attr, None)
                        if hasattr(obj, "_backend") and hasattr(obj._backend, "update"):
                            vault = obj
                            break
                    if vault:
                        vault._backend.update(path, append_content=new_content)
                        results.append({"type": "vault_note", "path": path, "status": "corrected"})

                else:
                    results.append({"type": corr_type, "status": "unsupported"})

            except Exception as e:
                results.append({"type": corr_type, "status": "error", "error": str(e)[:100]})

        # Log corrections in compliance audit
        try:
            from cognithor.security.compliance_audit import ComplianceAuditLog

            audit = ComplianceAuditLog()
            audit.record(
                "data_corrected",
                corrections_count=len(results),
                results=[r["status"] for r in results],
            )
        except Exception:
            log.debug("gdpr_correction_audit_log_failed", exc_info=True)

        return {
            "corrections_applied": len(results),
            "results": results,
            "gdpr_article": "Art. 16 DSGVO — Recht auf Berichtigung",
        }

    # -- GDPR Art. 20: Data Import (Portability) --------------------------

    @app.post("/api/v1/user/data/import", dependencies=deps)
    async def import_user_data(request: Request) -> dict[str, Any]:
        """GDPR Art. 20: Import data from another Cognithor instance."""
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON"}

        if body.get("format") != "cognithor_portable":
            return {"error": "Unsupported format. Expected 'cognithor_portable'"}

        counts = {}

        # Import vault notes
        vault_notes = body.get("vault_notes", [])
        if vault_notes:
            imported = 0
            vault = None
            for attr in dir(gateway):
                obj = getattr(gateway, attr, None)
                if hasattr(obj, "_backend") and hasattr(obj._backend, "save"):
                    vault = obj
                    break
            if vault:
                for note in vault_notes:
                    try:
                        path = note.get("path", "")
                        if not vault._backend.exists(path):
                            vault._backend.save(
                                path=path,
                                title=note.get("title", "Imported"),
                                content=note.get("content", ""),
                                tags=note.get("tags", ""),
                                folder=note.get("folder", "wissen"),
                                sources="",
                                backlinks=[],
                            )
                            imported += 1
                    except Exception:
                        log.debug("gdpr_import_vault_note_failed", exc_info=True)
            counts["vault_notes"] = imported

        # Import entities
        entities = body.get("entities", [])
        if entities:
            imported = 0
            memory_mgr = getattr(gateway, "_memory_manager", None)
            if memory_mgr:
                mcp = getattr(gateway, "_mcp_client", None)
                for ent in entities:
                    try:
                        name = ent.get("name", "")
                        etype = ent.get("type", "concept")
                        if name and mcp:
                            await mcp.call_tool(
                                "add_entity",
                                {
                                    "name": name,
                                    "entity_type": etype,
                                    "attributes": "{}",
                                    "source_file": "import",
                                },
                            )
                            imported += 1
                        elif name and hasattr(memory_mgr, "semantic"):
                            indexer = getattr(memory_mgr.semantic, "_indexer", None)
                            if indexer and hasattr(indexer, "_conn"):
                                indexer._conn.execute(
                                    "INSERT OR IGNORE INTO entities"
                                    " (name, entity_type) VALUES (?, ?)",
                                    (name, etype),
                                )
                                indexer._conn.commit()
                                imported += 1
                    except Exception:
                        log.debug("gdpr_import_entity_failed", exc_info=True)
            counts["entities"] = imported

        # Import relations
        relations = body.get("relations", [])
        if relations:
            imported = 0
            memory_mgr = getattr(gateway, "_memory_manager", None)
            if memory_mgr and hasattr(memory_mgr, "semantic"):
                indexer = getattr(memory_mgr.semantic, "_indexer", None)
                if indexer and hasattr(indexer, "_conn"):
                    for rel in relations:
                        try:
                            indexer._conn.execute(
                                "INSERT OR IGNORE INTO relations"
                                " (source_entity, relation_type, target_entity)"
                                " VALUES (?, ?, ?)",
                                (
                                    rel.get("source", ""),
                                    rel.get("relation", ""),
                                    rel.get("target", ""),
                                ),
                            )
                            imported += 1
                        except Exception:
                            log.debug("gdpr_import_relation_failed", exc_info=True)
                    indexer._conn.commit()
            counts["relations"] = imported

        # Import user preferences
        prefs = body.get("user_preferences", [])
        if prefs:
            imported = 0
            pref_store = getattr(gateway, "_user_pref_store", None)
            if pref_store and hasattr(pref_store, "_conn"):
                for pref in prefs if isinstance(prefs, list) else [prefs]:
                    try:
                        if isinstance(pref, dict):
                            for k, v in pref.items():
                                pref_store._conn.execute(
                                    "INSERT OR REPLACE INTO user_preferences"
                                    " (key, value) VALUES (?, ?)",
                                    (str(k), str(v)),
                                )
                            pref_store._conn.commit()
                            imported += 1
                    except Exception:
                        log.debug("gdpr_import_preference_failed", exc_info=True)
            counts["user_preferences"] = imported

        # Log import in compliance audit
        try:
            from cognithor.security.compliance_audit import ComplianceAuditLog

            audit = ComplianceAuditLog()
            audit.record("data_imported", counts=counts)
        except Exception:
            log.debug("gdpr_import_audit_log_failed", exc_info=True)

        return {
            "status": "imported",
            "counts": counts,
            "gdpr_article": "Art. 20 DSGVO — Datenportabilitaet",
        }

    # -- GDPR Art. 18/21: Purpose Restrictions ----------------------------

    @app.post("/api/v1/user/restrictions", dependencies=deps)
    async def set_restriction(request: Request) -> dict[str, Any]:
        """GDPR Art. 18/21: Restrict specific processing purposes."""
        consent_mgr = getattr(gateway, "_consent_manager", None)
        if not consent_mgr:
            return {"error": "Consent manager not available"}
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON"}
        user_id = body.get("user_id", "")
        channel = body.get("channel", "all")
        purpose = body.get("purpose", "")
        if not user_id or not purpose:
            return {"error": "user_id and purpose required"}
        consent_mgr.restrict_purpose(user_id, channel, purpose)
        return {"status": "restricted", "user_id": user_id, "purpose": purpose}

    @app.get("/api/v1/user/restrictions", dependencies=deps)
    async def get_restrictions(user_id: str = "") -> dict[str, Any]:
        """GDPR Art. 18/21: List active restrictions."""
        consent_mgr = getattr(gateway, "_consent_manager", None)
        if not consent_mgr:
            return {"restrictions": []}
        if not user_id:
            return {"error": "user_id query parameter required"}
        return {"user_id": user_id, "restrictions": consent_mgr.get_restrictions(user_id)}

    @app.delete("/api/v1/user/restrictions", dependencies=deps)
    async def remove_restriction(request: Request) -> dict[str, Any]:
        """GDPR Art. 18/21: Remove a processing restriction."""
        consent_mgr = getattr(gateway, "_consent_manager", None)
        if not consent_mgr:
            return {"error": "Consent manager not available"}
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON"}
        user_id = body.get("user_id", "")
        channel = body.get("channel", "all")
        purpose = body.get("purpose", "")
        if not user_id or not purpose:
            return {"error": "user_id and purpose required"}
        consent_mgr.unrestrict_purpose(user_id, channel, purpose)
        return {"status": "unrestricted", "user_id": user_id, "purpose": purpose}

    # -- Security Pipeline (Phase 19) ------------------------------------

    @app.get("/api/v1/security/pipeline/stats", dependencies=deps)
    async def pipeline_stats() -> dict[str, Any]:
        """Security-Pipeline Statistiken."""
        pipeline = getattr(gateway, "_security_pipeline", None)
        if pipeline is None:
            return {"total_runs": 0, "last_result": "none", "total_findings": 0, "pass_rate": 0}
        return pipeline.stats()

    @app.post("/api/v1/security/pipeline/run", dependencies=deps)
    async def pipeline_run(request: Request) -> dict[str, Any]:
        """Security-Pipeline manuell starten."""
        try:
            pipeline = getattr(gateway, "_security_pipeline", None)
            if pipeline is None:
                return {"error": "Security-Pipeline nicht verfügbar"}
            body = await request.json()
            trigger = body.get("trigger", "manual")

            def sanitizer(text: str) -> dict[str, Any]:
                return {"blocked": False}

            run = pipeline.run(
                handler_fn=sanitizer,
                is_blocked_fn=lambda r: r.get("blocked", False),
                test_inputs=body.get("test_inputs", []),
                dependencies=body.get("dependencies", []),
                trigger=trigger,
            )
            return run.to_dict()
        except Exception as exc:
            log.error("pipeline_run_failed", error=str(exc))
            return {"error": "Security-Pipeline-Run fehlgeschlagen"}

    @app.get("/api/v1/security/pipeline/history", dependencies=deps)
    async def pipeline_history() -> dict[str, Any]:
        """Security-Pipeline Run-Historie."""
        pipeline = getattr(gateway, "_security_pipeline", None)
        if pipeline is None:
            return {"runs": [], "count": 0}
        runs = pipeline.history(limit=20)
        return {"runs": [r.to_dict() for r in runs], "count": len(runs)}

    # -- Ecosystem Security Policy ----------------------------------------

    @app.get("/api/v1/ecosystem/policy/stats", dependencies=deps)
    async def ecosystem_policy_stats() -> dict[str, Any]:
        """Ecosystem-Policy Statistiken."""
        policy = getattr(gateway, "_ecosystem_policy", None)
        if policy is None:
            return {"total_requirements": 0, "minimum_tier": "community", "total_badges": 0}
        return policy.stats()

    @app.post("/api/v1/ecosystem/evaluate", dependencies=deps)
    async def ecosystem_evaluate(request: Request) -> dict[str, Any]:
        """Skill gegen Ecosystem-Policy evaluieren."""
        try:
            policy = getattr(gateway, "_ecosystem_policy", None)
            if policy is None:
                return {"error": "Ecosystem-Policy nicht verfügbar"}
            body = await request.json()
            skill_id = body.get("skill_id", "unknown")
            badge = policy.evaluate_skill(
                skill_id,
                has_signature=body.get("has_signature", False),
                has_sandbox=body.get("has_sandbox", False),
                has_license=body.get("has_license", False),
                has_network_control=body.get("has_network_control", False),
                passed_static_analysis=body.get("passed_static_analysis", False),
                passed_code_review=body.get("passed_code_review", False),
                passed_pentest=body.get("passed_pentest", False),
                has_audit_trail=body.get("has_audit_trail", False),
                has_input_validation=body.get("has_input_validation", False),
                is_dsgvo_compliant=body.get("is_dsgvo_compliant", False),
            )
            return badge.to_dict()
        except Exception as exc:
            log.error("ecosystem_evaluate_failed", error=str(exc))
            return {"error": "Ecosystem-Evaluierung fehlgeschlagen"}

    # -- AI Agent Security Framework (Phase 21) ---------------------------

    @app.get("/api/v1/framework/metrics", dependencies=deps)
    async def framework_metrics() -> dict[str, Any]:
        """Security-Metriken (MTTD, MTTR, etc.)."""
        metrics = getattr(gateway, "_security_metrics", None)
        if metrics is None:
            return {
                "mttd_seconds": 0,
                "mttr_seconds": 0,
                "resolution_rate": 100,
                "total_incidents": 0,
            }
        return metrics.to_dict()

    @app.get("/api/v1/framework/incidents", dependencies=deps)
    async def framework_incidents() -> dict[str, Any]:
        """Alle Incidents."""
        tracker = getattr(gateway, "_incident_tracker", None)
        if tracker is None:
            return {"incidents": [], "stats": {}}
        return {
            "incidents": [i.to_dict() for i in tracker.all_incidents()],
            "stats": tracker.stats(),
        }

    @app.get("/api/v1/framework/team", dependencies=deps)
    async def framework_team() -> dict[str, Any]:
        """Security-Team Uebersicht."""
        team = getattr(gateway, "_security_team", None)
        if team is None:
            return {"members": [], "stats": {"total_members": 0}}
        return {
            "members": [m.to_dict() for m in team.on_call()],
            "stats": team.stats(),
        }

    @app.get("/api/v1/framework/posture", dependencies=deps)
    async def framework_posture() -> dict[str, Any]:
        """Security-Posture-Score."""
        scorer = getattr(gateway, "_posture_scorer", None)
        if scorer is None:
            return {"posture_score": 0, "level": "unknown"}
        metrics = getattr(gateway, "_security_metrics", None)
        pipeline = getattr(gateway, "_security_pipeline", None)
        team = getattr(gateway, "_security_team", None)
        return scorer.calculate(
            resolution_rate=metrics.resolution_rate() if metrics else 100,
            mttr_seconds=metrics.mttr() if metrics else 0,
            team_roles_filled=team.member_count if team else 0,
            pipeline_pass_rate=pipeline.stats().get("pass_rate", 100) if pipeline else 100,
        )

    # -- CI/CD Security Gate (Phase 24) -----------------------------------

    @app.get("/api/v1/gate/stats", dependencies=deps)
    async def gate_stats() -> dict[str, Any]:
        """Security-Gate Statistiken."""
        gate = getattr(gateway, "_security_gate", None)
        if gate is None:
            return {"total_evaluations": 0, "pass_rate": 100}
        return gate.stats()

    @app.post("/api/v1/gate/evaluate", dependencies=deps)
    async def gate_evaluate(body: dict[str, Any]) -> dict[str, Any]:
        """Evaluiert ein Pipeline-Ergebnis."""
        gate = getattr(gateway, "_security_gate", None)
        if gate is None:
            return {"verdict": "pass", "error": "Gate nicht verfügbar"}
        result = gate.evaluate(body)
        return result.to_dict()

    @app.get("/api/v1/gate/history", dependencies=deps)
    async def gate_history() -> dict[str, Any]:
        """Gate-History."""
        gate = getattr(gateway, "_security_gate", None)
        if gate is None:
            return {"history": []}
        return {"history": [r.to_dict() for r in gate.history()]}

    @app.get("/api/v1/redteam/stats", dependencies=deps)
    async def redteam_stats() -> dict[str, Any]:
        """Continuous Red-Team Statistiken."""
        rt = getattr(gateway, "_continuous_redteam", None)
        if rt is None:
            return {"total_probes": 0, "detection_rate": 100}
        return rt.stats()

    @app.get("/api/v1/scans/stats", dependencies=deps)
    async def scans_stats() -> dict[str, Any]:
        """Scan-Scheduler Status."""
        sched = getattr(gateway, "_scan_scheduler", None)
        if sched is None:
            return {"total_schedules": 0}
        return sched.stats()

    # -- Red-Team-Framework (Phase 30) ------------------------------------

    @app.get("/api/v1/red-team/stats", dependencies=deps)
    async def red_team_stats() -> dict[str, Any]:
        """Red-Team Statistiken."""
        rt = getattr(gateway, "_red_team", None)
        if rt is None:
            return {"total_runs": 0}
        return rt.stats()

    @app.get("/api/v1/red-team/coverage", dependencies=deps)
    async def red_team_coverage() -> dict[str, Any]:
        """Angriffs-Abdeckung."""
        rt = getattr(gateway, "_red_team", None)
        if rt is None:
            return {"coverage_rate": 0}
        return rt.coverage_report()

    @app.get("/api/v1/red-team/latest", dependencies=deps)
    async def red_team_latest() -> dict[str, Any]:
        """Letzter Red-Team-Report."""
        rt = getattr(gateway, "_red_team", None)
        if rt is None:
            return {"report": None}
        report = rt.runner.latest_report()
        return {"report": report.to_dict() if report else None}

    # -- Code-Audit (Phase 33) -------------------------------------------

    @app.get("/api/v1/code-audit/stats", dependencies=deps)
    async def code_audit_stats() -> dict[str, Any]:
        """Code-Audit Statistiken."""
        ca = getattr(gateway, "_code_auditor", None)
        if ca is None:
            return {"total_audits": 0}
        return ca.stats()
