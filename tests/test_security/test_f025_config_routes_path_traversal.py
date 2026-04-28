"""Tests fuer F-025: config_routes run_id Path Traversal.

Prueft dass:
  - run_id mit '../' in wf_get_dag_run abgelehnt wird (Status 400)
  - run_id mit '../../etc/passwd' abgelehnt wird
  - run_id mit absoluten Pfaden abgelehnt wird
  - Normaler run_id weiterhin funktioniert
  - Path-Traversal nicht zu Dateilesen ausserhalb fuehrt
  - .resolve() + .relative_to() Pattern im Source vorhanden ist
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from cognithor.config import CognithorConfig
from cognithor.config_manager import ConfigManager

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# FakeApp (gleicher Pattern wie test_workflow_graph_routes.py)
# ============================================================================


class FakeApp:
    def __init__(self) -> None:
        self.routes: dict[str, Any] = {}

    def _register(self, method: str, path: str, **kwargs: Any):
        def decorator(fn):
            self.routes[f"{method} {path}"] = fn
            return fn

        return decorator

    def get(self, path: str, **kwargs: Any):
        return self._register("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any):
        return self._register("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any):
        return self._register("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs: Any):
        return self._register("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any):
        return self._register("DELETE", path, **kwargs)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / ".cognithor"
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture
def config(tmp_home: Path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_home)


@pytest.fixture
def config_manager(config: CognithorConfig) -> ConfigManager:
    return ConfigManager(config=config)


def _setup_dag_app(
    app: FakeApp,
    config_manager: ConfigManager,
    cp_dir: Path,
) -> Any:
    """Registriert Routes mit DAG-Engine und gibt Handler zurueck."""
    dag_engine = MagicMock()
    dag_engine._checkpoint_dir = cp_dir

    gw = MagicMock()
    gw._workflow_engine = None
    gw._template_library = None
    gw._dag_workflow_engine = dag_engine

    from cognithor.channels.config_routes import create_config_routes

    create_config_routes(app, config_manager, gateway=gw)

    return app.routes["GET /api/v1/workflows/dag/runs/{run_id}"]


# ============================================================================
# Path-Traversal Tests
# ============================================================================


class TestPathTraversalRejection:
    """Prueft dass Path-Traversal in run_id abgelehnt wird."""

    @pytest.mark.asyncio
    async def test_dotdot_rejected(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)

        result = await handler("../../etc/passwd")
        assert result.get("status") == 400
        assert "Path-Traversal" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_single_dotdot_rejected(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)

        result = await handler("../secret")
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_nested_traversal_rejected(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)

        result = await handler("foo/../../bar")
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_dotdot_slash_prefix_rejected(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)

        result = await handler("../../../etc/shadow")
        assert result.get("status") == 400


class TestPathTraversalNoFileLeak:
    """Prueft dass Path-Traversal nicht zu Datei-Leak fuehrt."""

    @pytest.mark.asyncio
    async def test_traversal_does_not_read_outside(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        """Sensitive Datei ausserhalb von checkpoint_dir wird nicht gelesen."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        # Erstelle sensitive Datei neben checkpoint_dir
        secret = tmp_path / "secret.json"
        secret.write_text('{"password": "123"}', encoding="utf-8")

        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)

        # Versuche die Datei via Traversal zu lesen
        result = await handler("../secret")
        # Darf nicht den Inhalt der Datei zurueckgeben
        assert "password" not in str(result)
        assert result.get("status") == 400


# ============================================================================
# Normale Funktion
# ============================================================================


class TestNormalRunIdWorks:
    """Prueft dass normale run_ids weiterhin funktionieren."""

    @pytest.mark.asyncio
    async def test_valid_run_id_returns_data(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        run_data = {
            "id": "abc123",
            "workflow_name": "Test",
            "status": "success",
            "node_results": {},
        }
        (cp_dir / "abc123.json").write_text(
            json.dumps(run_data),
            encoding="utf-8",
        )

        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)
        result = await handler("abc123")
        assert result["id"] == "abc123"
        assert result["workflow_name"] == "Test"

    @pytest.mark.asyncio
    async def test_nonexistent_run_returns_404(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)
        result = await handler("nonexistent")
        assert result.get("status") == 404

    @pytest.mark.asyncio
    async def test_uuid_style_run_id(
        self,
        tmp_path: Path,
        config_manager: ConfigManager,
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        run_id = "550e8400-e29b-41d4-a716-446655440000"
        run_data = {"id": run_id, "status": "ok"}
        (cp_dir / f"{run_id}.json").write_text(
            json.dumps(run_data),
            encoding="utf-8",
        )

        handler = _setup_dag_app(FakeApp(), config_manager, cp_dir)
        result = await handler(run_id)
        assert result["id"] == run_id


# ============================================================================
# Source-Level-Checks
# ============================================================================


class TestSourceLevelChecks:
    """Prueft den Source-Code auf den Fix."""

    def _get_source(self) -> str:
        # `_register_workflow_graph_routes` lebt aktuell in `_factory.py` und
        # zieht im Rahmen des Splits in `config_routes/workflows.py` um
        # (siehe `docs/superpowers/plans/2026-04-29-config-routes-split.md`).
        # Solange das Symbol im Paket existiert, finden wir es per Lookup.
        from cognithor.channels import config_routes as _cr_pkg
        from cognithor.channels.config_routes import _factory

        for module in (_factory, _cr_pkg):
            fn = getattr(module, "_register_workflow_graph_routes", None)
            if fn is not None:
                return inspect.getsource(fn)
        raise AttributeError("_register_workflow_graph_routes nicht gefunden")

    def test_uses_resolve(self) -> None:
        source = self._get_source()
        assert ".resolve()" in source

    def test_uses_relative_to(self) -> None:
        source = self._get_source()
        assert ".relative_to(" in source

    def test_returns_400_on_traversal(self) -> None:
        source = self._get_source()
        assert "Path-Traversal" in source
        assert "400" in source
