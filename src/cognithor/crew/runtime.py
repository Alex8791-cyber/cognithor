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
