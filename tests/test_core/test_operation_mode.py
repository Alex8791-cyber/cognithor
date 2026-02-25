"""Tests fuer OperationMode Foundation (Feature 1).

Testet:
  - Auto-Detection aus API-Keys
  - Expliziter Mode-Override
  - Gatekeeper-Enforcement im OFFLINE-Modus
  - Recherche-Tools im OFFLINE-Modus erlaubt
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.core.gatekeeper import Gatekeeper
from jarvis.models import (
    GateStatus,
    OperationMode,
    PlannedAction,
    RiskLevel,
    SessionContext,
    ToolCapability,
    ToolCapabilitySpec,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Config-Tests ──────────────────────────────────────────────


class TestOperationModeAutoDetect:
    """Tests fuer die automatische OperationMode-Erkennung."""

    def test_auto_detect_offline(self, tmp_path: Path) -> None:
        """Kein API-Key → OFFLINE."""
        config = JarvisConfig(jarvis_home=tmp_path)
        assert config.resolved_operation_mode == OperationMode.OFFLINE

    def test_auto_detect_online_openai(self, tmp_path: Path) -> None:
        """OpenAI-Key vorhanden → ONLINE."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            openai_api_key="sk-test1234567890abcdefg",
        )
        assert config.resolved_operation_mode == OperationMode.ONLINE

    def test_auto_detect_online_anthropic(self, tmp_path: Path) -> None:
        """Anthropic-Key vorhanden → ONLINE."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            anthropic_api_key="sk-ant-test1234567890abcdefg",
        )
        assert config.resolved_operation_mode == OperationMode.ONLINE

    def test_explicit_mode_override_hybrid(self, tmp_path: Path) -> None:
        """Explizit 'hybrid' → HYBRID (unabhaengig von API-Keys)."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            operation_mode="hybrid",
        )
        assert config.resolved_operation_mode == OperationMode.HYBRID

    def test_explicit_mode_override_offline(self, tmp_path: Path) -> None:
        """Explizit 'offline' → OFFLINE auch mit API-Key."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            operation_mode="offline",
            openai_api_key="sk-test1234567890abcdefg",
        )
        assert config.resolved_operation_mode == OperationMode.OFFLINE

    def test_explicit_mode_override_online(self, tmp_path: Path) -> None:
        """Explizit 'online' → ONLINE auch ohne API-Key."""
        config = JarvisConfig(
            jarvis_home=tmp_path,
            operation_mode="online",
        )
        assert config.resolved_operation_mode == OperationMode.ONLINE


# ── Gatekeeper-Tests ──────────────────────────────────────────


def _make_capability_matrix(tool_specs: dict[str, frozenset[ToolCapability]]):
    """Erzeugt ein Mock-CapabilityMatrix-Objekt."""
    matrix = MagicMock()
    specs = {}
    for tool_name, caps in tool_specs.items():
        spec = ToolCapabilitySpec(tool_name=tool_name, capabilities=caps)
        specs[tool_name] = spec

    def get_spec(name: str):
        return specs.get(name)

    matrix.get_spec = get_spec
    return matrix


@pytest.fixture()
def gk_config_offline(tmp_path: Path) -> JarvisConfig:
    """Config im OFFLINE-Modus."""
    config = JarvisConfig(
        jarvis_home=tmp_path,
        operation_mode="offline",
        security=SecurityConfig(allowed_paths=[str(tmp_path), "/tmp/jarvis/"]),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def gk_config_online(tmp_path: Path) -> JarvisConfig:
    """Config im ONLINE-Modus."""
    config = JarvisConfig(
        jarvis_home=tmp_path,
        operation_mode="online",
        security=SecurityConfig(allowed_paths=[str(tmp_path), "/tmp/jarvis/"]),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def session() -> SessionContext:
    """Standard-Session fuer Tests."""
    return SessionContext(user_id="test_user", channel="test")


class TestGatekeeperOfflineEnforcement:
    """Tests fuer OperationMode-Enforcement im Gatekeeper."""

    def test_offline_blocks_network_tool(
        self, gk_config_offline: JarvisConfig, session: SessionContext,
    ) -> None:
        """HTTP-Tool wird im OFFLINE-Modus blockiert."""
        gk = Gatekeeper(gk_config_offline, operation_mode=OperationMode.OFFLINE)
        gk._capability_matrix = _make_capability_matrix({
            "cloud_api_call": frozenset({ToolCapability.NETWORK_HTTP}),
        })
        gk.initialize()

        action = PlannedAction(tool="cloud_api_call", params={"url": "https://api.example.com"})
        decision = gk.evaluate(action, session)

        assert decision.status == GateStatus.BLOCK
        assert "OFFLINE" in decision.reason
        assert decision.policy_name == "operation_mode_offline"

    def test_offline_blocks_websocket_tool(
        self, gk_config_offline: JarvisConfig, session: SessionContext,
    ) -> None:
        """WebSocket-Tool wird im OFFLINE-Modus blockiert."""
        gk = Gatekeeper(gk_config_offline, operation_mode=OperationMode.OFFLINE)
        gk._capability_matrix = _make_capability_matrix({
            "ws_connect": frozenset({ToolCapability.NETWORK_WS}),
        })
        gk.initialize()

        action = PlannedAction(tool="ws_connect", params={"url": "wss://stream.example.com"})
        decision = gk.evaluate(action, session)

        assert decision.status == GateStatus.BLOCK
        assert decision.policy_name == "operation_mode_offline"

    def test_online_allows_network_tool(
        self, gk_config_online: JarvisConfig, session: SessionContext,
    ) -> None:
        """HTTP-Tool wird im ONLINE-Modus durchgelassen."""
        gk = Gatekeeper(gk_config_online, operation_mode=OperationMode.ONLINE)
        gk._capability_matrix = _make_capability_matrix({
            "cloud_api_call": frozenset({ToolCapability.NETWORK_HTTP}),
        })
        gk.initialize()

        action = PlannedAction(tool="cloud_api_call", params={"url": "https://api.example.com"})
        decision = gk.evaluate(action, session)

        # Sollte NICHT wegen OperationMode blockiert werden
        assert decision.policy_name != "operation_mode_offline"

    def test_offline_allows_local_tool(
        self, gk_config_offline: JarvisConfig, session: SessionContext,
    ) -> None:
        """FS/exec-Tools gehen im OFFLINE-Modus durch."""
        gk = Gatekeeper(gk_config_offline, operation_mode=OperationMode.OFFLINE)
        gk._capability_matrix = _make_capability_matrix({
            "read_file": frozenset({ToolCapability.FS_READ}),
        })
        gk.initialize()

        action = PlannedAction(
            tool="read_file",
            params={"path": str(gk_config_offline.jarvis_home / "test.txt")},
        )
        decision = gk.evaluate(action, session)

        assert decision.status != GateStatus.BLOCK or decision.policy_name != "operation_mode_offline"

    def test_offline_allows_web_search(
        self, gk_config_offline: JarvisConfig, session: SessionContext,
    ) -> None:
        """Web-Recherche bleibt im OFFLINE-Modus erlaubt."""
        gk = Gatekeeper(gk_config_offline, operation_mode=OperationMode.OFFLINE)
        gk._capability_matrix = _make_capability_matrix({
            "web_search": frozenset({ToolCapability.NETWORK_HTTP}),
        })
        gk.initialize()

        action = PlannedAction(tool="web_search", params={"query": "Python tutorials"})
        decision = gk.evaluate(action, session)

        # web_search ist in _OFFLINE_ALLOWED_NETWORK_TOOLS → nicht blockiert
        assert decision.policy_name != "operation_mode_offline"

    def test_offline_allows_web_fetch(
        self, gk_config_offline: JarvisConfig, session: SessionContext,
    ) -> None:
        """web_fetch bleibt im OFFLINE-Modus erlaubt (Recherche)."""
        gk = Gatekeeper(gk_config_offline, operation_mode=OperationMode.OFFLINE)
        gk._capability_matrix = _make_capability_matrix({
            "web_fetch": frozenset({ToolCapability.NETWORK_HTTP}),
        })
        gk.initialize()

        action = PlannedAction(tool="web_fetch", params={"url": "https://example.com"})
        decision = gk.evaluate(action, session)

        assert decision.policy_name != "operation_mode_offline"

    def test_offline_no_capability_matrix_no_crash(
        self, gk_config_offline: JarvisConfig, session: SessionContext,
    ) -> None:
        """Ohne CapabilityMatrix → kein Crash, kein Block."""
        gk = Gatekeeper(gk_config_offline, operation_mode=OperationMode.OFFLINE)
        gk._capability_matrix = None
        gk.initialize()

        action = PlannedAction(tool="some_tool", params={"key": "val"})
        decision = gk.evaluate(action, session)

        # Kein OperationMode-Block (Matrix fehlt → Schritt wird uebersprungen)
        assert decision.policy_name != "operation_mode_offline"


# ── Enum-Tests ────────────────────────────────────────────────


class TestOperationModeEnum:
    """Tests fuer das OperationMode-Enum."""

    def test_values(self) -> None:
        assert OperationMode.OFFLINE == "offline"
        assert OperationMode.ONLINE == "online"
        assert OperationMode.HYBRID == "hybrid"

    def test_from_string(self) -> None:
        assert OperationMode("offline") == OperationMode.OFFLINE
        assert OperationMode("online") == OperationMode.ONLINE
        assert OperationMode("hybrid") == OperationMode.HYBRID

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            OperationMode("invalid_mode")
