"""Cognithor · Workflow Execution Graph routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_workflow_graph_routes()` — REST-Endpoints fuer die
Visualisierung der DAG-Workflow-Ausfuehrung.
"""

from __future__ import annotations

import json
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
