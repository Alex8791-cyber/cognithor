"""Runtime helpers for Crew.kickoff() / kickoff_async()."""

from __future__ import annotations

import threading
from typing import Any

_registry_lock = threading.Lock()
_registry_singleton: Any = None


def get_default_tool_registry() -> Any:
    """Return a process-wide default ToolRegistryDB instance.

    Builds from `cognithor.config.load_config().cognithor_home / 'db' /
    'tool_registry.db'`. If config loading fails (e.g. standalone test without
    ~/.cognithor/ present), fall back to a temp-dir DB — never silently return
    None.
    """
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is not None:
            return _registry_singleton
        from pathlib import Path

        from cognithor.config import load_config
        from cognithor.mcp.tool_registry_db import ToolRegistryDB

        try:
            cfg = load_config()
            db_path = Path(cfg.cognithor_home) / "db" / "tool_registry.db"
        except Exception as exc:
            import tempfile
            import warnings

            warnings.warn(
                f"cognithor config load failed ({exc!r}); using temp-dir tool "
                "registry. State will not persist across restarts.",
                RuntimeWarning,
                # warn -> get_default_tool_registry -> kickoff -> USER  (4 frames)
                stacklevel=3,
            )
            db_path = Path(tempfile.gettempdir()) / "cognithor_crew_registry.db"
        _registry_singleton = ToolRegistryDB(db_path=db_path)
        return _registry_singleton


_planner_lock = threading.Lock()
_planner_singleton: Any = None


def get_default_planner() -> Any:
    """Return a process-wide default ``Planner`` instance.

    No auto-discovery: always built from config for standalone Crew scripts.
    Embedded callers (Gateway, tests) pass a live Planner to
    ``Crew(planner=...)`` so this factory is never invoked for them.

    Async-safe: construction happens OUTSIDE the ``threading.Lock`` so async
    event loops aren't blocked for tens of milliseconds while the Planner
    wires up Ollama + router. The lock guards only the final sentinel swap,
    and the fast-path early-return keeps hot calls lock-free.
    """
    global _planner_singleton
    if _planner_singleton is not None:
        return _planner_singleton

    from cognithor.config import load_config
    from cognithor.core.model_router import ModelRouter, OllamaClient
    from cognithor.core.planner import Planner

    cfg = load_config()
    ollama = OllamaClient(cfg)
    router = ModelRouter(cfg, ollama)
    candidate = Planner(cfg, ollama, router)

    with _planner_lock:
        if _planner_singleton is None:
            _planner_singleton = candidate
        return _planner_singleton
