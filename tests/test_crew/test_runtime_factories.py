"""Unit tests for ``cognithor.crew.runtime`` singleton factories.

The crew test-suite's conftest autouse-patches ``get_default_tool_registry``
and ``get_default_planner`` with mocks so no kickoff test touches
``~/.cognithor/`` or wires a real Ollama client. That keeps ~70 tests
hermetic but leaves the factory bodies themselves uncovered.

This module overrides the autouse fixtures with no-ops so we can exercise
the real factory functions with inner imports stubbed via monkeypatch.
The tests verify:

* singleton-ness: two calls return the same instance,
* ``cognithor.config.load_config`` failure falls back to the tempdir DB,
* the planner factory constructs OllamaClient / ModelRouter / Planner
  in order and caches the result across calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _patched_tool_registry() -> None:  # pragma: no cover - override
    """Override conftest autouse so factories run unmocked in this module."""
    return None


@pytest.fixture(autouse=True)
def _patched_default_planner() -> None:  # pragma: no cover - override
    """Override conftest autouse so factories run unmocked in this module."""
    return None


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    import cognithor.crew.runtime as runtime

    monkeypatch.setattr(runtime, "_registry_singleton", None)
    monkeypatch.setattr(runtime, "_planner_singleton", None)


class TestGetDefaultToolRegistry:
    def test_builds_from_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """Happy path: config loads, ToolRegistryDB is built with db_path."""
        import cognithor.crew.runtime as runtime

        fake_cfg = MagicMock()
        fake_cfg.cognithor_home = tmp_path
        monkeypatch.setattr("cognithor.config.load_config", lambda: fake_cfg)

        mock_db_cls = MagicMock(name="ToolRegistryDB")
        monkeypatch.setattr("cognithor.mcp.tool_registry_db.ToolRegistryDB", mock_db_cls)

        result = runtime.get_default_tool_registry()
        assert result is mock_db_cls.return_value
        mock_db_cls.assert_called_once()
        db_path_arg = mock_db_cls.call_args.kwargs["db_path"]
        assert str(db_path_arg).endswith("tool_registry.db")

    def test_falls_back_to_tempdir_on_config_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If load_config raises, emit a RuntimeWarning and use tempdir path."""
        import cognithor.crew.runtime as runtime

        def _boom() -> Any:
            raise RuntimeError("no config on this host")

        monkeypatch.setattr("cognithor.config.load_config", _boom)

        mock_db_cls = MagicMock(name="ToolRegistryDB")
        monkeypatch.setattr("cognithor.mcp.tool_registry_db.ToolRegistryDB", mock_db_cls)

        with pytest.warns(RuntimeWarning, match="cognithor config load failed"):
            result = runtime.get_default_tool_registry()

        assert result is mock_db_cls.return_value
        db_path_arg = mock_db_cls.call_args.kwargs["db_path"]
        assert "cognithor_crew_registry.db" in str(db_path_arg)

    def test_singleton_across_calls(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """Second call returns the cached instance, no second DB construction."""
        import cognithor.crew.runtime as runtime

        fake_cfg = MagicMock()
        fake_cfg.cognithor_home = tmp_path
        monkeypatch.setattr("cognithor.config.load_config", lambda: fake_cfg)

        mock_db_cls = MagicMock(name="ToolRegistryDB")
        monkeypatch.setattr("cognithor.mcp.tool_registry_db.ToolRegistryDB", mock_db_cls)

        first = runtime.get_default_tool_registry()
        second = runtime.get_default_tool_registry()
        assert first is second
        assert mock_db_cls.call_count == 1


class TestGetDefaultPlanner:
    def test_builds_ollama_router_planner_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory wires OllamaClient -> ModelRouter -> Planner once."""
        import cognithor.crew.runtime as runtime

        fake_cfg = MagicMock()
        monkeypatch.setattr("cognithor.config.load_config", lambda: fake_cfg)

        mock_ollama_cls = MagicMock(name="OllamaClient")
        mock_router_cls = MagicMock(name="ModelRouter")
        mock_planner_cls = MagicMock(name="Planner")
        monkeypatch.setattr("cognithor.core.model_router.OllamaClient", mock_ollama_cls)
        monkeypatch.setattr("cognithor.core.model_router.ModelRouter", mock_router_cls)
        monkeypatch.setattr("cognithor.core.planner.Planner", mock_planner_cls)

        result = runtime.get_default_planner()
        assert result is mock_planner_cls.return_value
        mock_ollama_cls.assert_called_once_with(fake_cfg)
        mock_router_cls.assert_called_once_with(fake_cfg, mock_ollama_cls.return_value)
        mock_planner_cls.assert_called_once_with(
            fake_cfg, mock_ollama_cls.return_value, mock_router_cls.return_value
        )

    def test_singleton_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Second call returns the cached instance; no second wiring happens."""
        import cognithor.crew.runtime as runtime

        fake_cfg = MagicMock()
        monkeypatch.setattr("cognithor.config.load_config", lambda: fake_cfg)

        mock_ollama_cls = MagicMock(name="OllamaClient")
        mock_router_cls = MagicMock(name="ModelRouter")
        mock_planner_cls = MagicMock(name="Planner")
        monkeypatch.setattr("cognithor.core.model_router.OllamaClient", mock_ollama_cls)
        monkeypatch.setattr("cognithor.core.model_router.ModelRouter", mock_router_cls)
        monkeypatch.setattr("cognithor.core.planner.Planner", mock_planner_cls)

        first = runtime.get_default_planner()
        second = runtime.get_default_planner()
        assert first is second
        # Planner constructed at most twice (candidate + cache swap) but only
        # one result cached; however the factory builds "candidate" outside
        # the lock so a second call does NOT re-enter construction because
        # the early-return fires before the inner imports.
        assert mock_planner_cls.call_count == 1
