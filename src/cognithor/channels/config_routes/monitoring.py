"""Cognithor · Monitoring + Prometheus routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_monitoring_routes()` (Live-Metriken via MonitoringHub, JSON
und SSE-Stream) sowie `_register_prometheus_routes()` (`/metrics`
Prometheus-Exposition Format). Beide bekommen den `get_hub`-Callable
als Parameter und teilen sich damit denselben MonitoringHub-Singleton,
der in `_factory.py` gehalten wird.
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
# Monitoring / metrics routes
# ======================================================================


def _register_monitoring_routes(
    app: Any,
    deps: list[Any],
    get_hub: Any,
    config_manager: Any = None,
) -> None:
    """Metrics, events, audit-trail, heartbeat, SSE streaming, performance."""

    @app.get("/api/v1/monitoring/dashboard", dependencies=deps)
    async def monitoring_dashboard() -> dict[str, Any]:
        """Komplett-Snapshot fuer das Live-Dashboard."""
        return get_hub().dashboard_snapshot()

    @app.get("/api/v1/monitoring/metrics", dependencies=deps)
    async def monitoring_metrics() -> dict[str, Any]:
        """Aktuelle Metriken."""
        hub = get_hub()
        return {"snapshot": hub.metrics.snapshot(), "names": hub.metrics.all_metric_names()}

    @app.get("/api/v1/monitoring/metrics/{name}", dependencies=deps)
    async def monitoring_metric_history(name: str, n: int = 60) -> dict[str, Any]:
        """Zeitreihe einer einzelnen Metrik."""
        return {"name": name, "history": get_hub().metrics.get_history(name, last_n=n)}

    @app.get("/api/v1/monitoring/events", dependencies=deps)
    async def monitoring_events(n: int = 50, severity: str = "") -> dict[str, Any]:
        """Letzte System-Events."""
        hub = get_hub()
        events = hub.events.recent_events(n=n, severity=severity or "")
        return {"events": [e.to_dict() for e in events], "total": hub.events.event_count}

    @app.get("/api/v1/monitoring/audit", dependencies=deps)
    async def audit_trail(
        action: str = "",
        actor: str = "",
        severity: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Durchsucht den Audit-Trail."""
        hub = get_hub()
        entries = hub.audit.search(action=action, actor=actor, severity=severity, limit=limit)
        return {
            "entries": [e.to_dict() for e in entries],
            "total": hub.audit.entry_count,
            "severity_counts": hub.audit.severity_counts(),
        }

    @app.get("/api/v1/audit/verify", dependencies=deps)
    async def verify_audit_integrity() -> dict[str, Any]:
        """Verify the integrity of the gatekeeper audit hash-chain."""
        import json as json_mod

        if config_manager is None:
            raise HTTPException(500, "Config manager not available")
        _cfg = config_manager.config
        gk_log = _cfg.cognithor_home / "logs" / "gatekeeper.jsonl"
        if not gk_log.exists():
            return {"status": "no_log", "message": "No gatekeeper audit log found."}

        total = 0
        valid = 0
        broken_at = None
        prev_hash = "genesis"

        try:
            with open(gk_log, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        entry = json_mod.loads(line)
                    except json_mod.JSONDecodeError:
                        broken_at = line_no
                        break

                    stored_prev = entry.get("prev_hash", "")
                    if stored_prev != prev_hash and broken_at is None:
                        broken_at = line_no

                    if broken_at is None:
                        valid += 1

                    prev_hash = entry.get("hash", "")
        except OSError as exc:
            return {"status": "error", "message": str(exc)}

        return {
            "status": "intact" if broken_at is None else "broken",
            "total_entries": total,
            "valid_entries": valid,
            "broken_at_line": broken_at,
            "log_file": str(gk_log),
        }

    @app.get("/api/v1/audit/timestamps", dependencies=deps)
    async def list_audit_timestamps() -> dict[str, Any]:
        """List all RFC 3161 TSA timestamps for audit logs."""
        try:
            from cognithor.security.tsa import TSAClient

            if config_manager is None:
                raise HTTPException(500, "Config manager not available")
            _cfg = config_manager.config
            tsa_dir = _cfg.cognithor_home / "tsa"
            client = TSAClient(storage_dir=tsa_dir)
            timestamps = client.list_timestamps()
            return {
                "timestamps": timestamps,
                "count": len(timestamps),
                "tsa_url": getattr(
                    getattr(_cfg, "audit", None), "tsa_url", "https://freetsa.org/tsr"
                ),
                "tsa_enabled": getattr(getattr(_cfg, "audit", None), "tsa_enabled", False),
            }
        except Exception as exc:
            return {"timestamps": [], "count": 0, "error": str(exc)}

    @app.get("/api/v1/monitoring/heartbeat", dependencies=deps)
    async def heartbeat_status() -> dict[str, Any]:
        """Heartbeat-Status und Historie."""
        hub = get_hub()
        return {
            "stats": hub.heartbeat.stats(),
            "recent_runs": [r.to_dict() for r in hub.heartbeat.recent_runs(20)],
        }

    # -- SSE Live-Event-Streaming -----------------------------------------

    @app.get("/api/v1/monitoring/stream", dependencies=deps)
    async def monitoring_sse_stream() -> Any:
        """Server-Sent-Events Stream fuer Live-Monitoring."""
        from starlette.responses import StreamingResponse

        hub = get_hub()
        queue = hub.events.create_sse_stream()

        async def event_generator():
            import asyncio

            try:
                while True:
                    try:
                        event = queue.get_nowait()
                        yield event.to_sse()
                    except Exception:
                        yield ": keepalive\n\n"
                        await asyncio.sleep(1)
            finally:
                hub.events.remove_sse_stream(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )


# ======================================================================
# Prometheus metrics endpoint
# ======================================================================


def _register_prometheus_routes(
    app: Any,
    get_hub: Any,
    gateway: Any,
) -> None:
    """Prometheus /metrics endpoint -- no auth required (standard practice)."""

    @app.get("/metrics")
    async def prometheus_metrics() -> Any:
        """Prometheus-Metriken im Text Exposition Format."""
        from starlette.responses import Response

        from cognithor.telemetry.prometheus import PrometheusExporter

        # Collect sources: MetricsProvider from TelemetryHub, MetricCollector from MonitoringHub
        metrics_provider = None
        metric_collector = None

        # TelemetryHub -> MetricsProvider (telemetry/metrics.py)
        if gateway is not None:
            telemetry_hub = getattr(gateway, "_telemetry_hub", None)
            if telemetry_hub is not None:
                metrics_provider = getattr(telemetry_hub, "metrics", None)

        # MonitoringHub -> MetricCollector (gateway/monitoring.py)
        try:
            hub = get_hub()
            if hub is not None:
                metric_collector = getattr(hub, "metrics", None)
        except Exception:
            pass  # Cleanup — metric collector lookup failure is non-critical

        exporter = PrometheusExporter(
            metrics_provider=metrics_provider,
            metric_collector=metric_collector,
        )
        content = exporter.export()
        return Response(
            content=content,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
