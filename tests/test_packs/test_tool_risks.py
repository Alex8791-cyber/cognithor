"""Tests for pack tool-risk wiring.

Covers:
- ``PackManifest.tool_risks`` validation (enum + subset-of-tools).
- ``PackLoader._register_tool_risks`` populates the MCP registry.
- ``@cognithor_tool`` decorator attaches ``ToolMetadata``.
- Gatekeeper uses the per-tool risk_level from the registry.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from cognithor.config import CognithorConfig
from cognithor.core.gatekeeper import Gatekeeper
from cognithor.models import MCPToolInfo, PlannedAction
from cognithor.packs.interface import PackContext, PackManifest
from cognithor.packs.loader import PackLoader
from cognithor.tools.decorator import (
    VALID_RISK_LEVELS,
    ToolMetadata,
    cognithor_tool,
    get_tool_metadata,
    iter_decorated_tools,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_manifest(
    *,
    tools: list[str] | None = None,
    tool_risks: dict[str, str] | None = None,
    license: str = "apache-2.0",
) -> PackManifest:
    eula = "EULA text."
    eula_hash = hashlib.sha256(eula.encode("utf-8")).hexdigest()
    raw: dict = {
        "schema_version": 1,
        "namespace": "test-ns",
        "pack_id": "test-pack",
        "version": "1.0.0",
        "display_name": "Test",
        "description": "Test pack",
        "license": license,
        "min_cognithor_version": ">=0.92.0",
        "eula_sha256": eula_hash,
        "publisher": {"id": "tester", "display_name": "Tester"},
    }
    if tools is not None:
        raw["tools"] = tools
    if tool_risks is not None:
        raw["tool_risks"] = tool_risks
    return PackManifest.model_validate(raw)


class TestPackManifestToolRisks:
    def test_tool_risks_default_empty(self):
        m = _make_manifest()
        assert m.tool_risks == {}

    def test_tool_risks_accepts_valid_levels(self):
        m = _make_manifest(
            tools=["scan", "write", "send", "nuke"],
            tool_risks={
                "scan": "green",
                "write": "yellow",
                "send": "orange",
                "nuke": "red",
            },
        )
        assert m.tool_risks["scan"] == "green"
        assert m.tool_risks["nuke"] == "red"

    def test_tool_risks_rejects_invalid_level(self):
        with pytest.raises(ValidationError, match="tool_risks"):
            _make_manifest(tools=["scan"], tool_risks={"scan": "pink"})

    def test_tool_risks_must_be_subset_of_tools(self):
        with pytest.raises(ValidationError, match="not in declared tools"):
            _make_manifest(tools=["scan"], tool_risks={"scan": "green", "phantom": "green"})

    def test_tool_risks_allowed_without_tools_list(self):
        """If tools is empty the subset check is skipped (backward compat)."""
        m = _make_manifest(tool_risks={"scan": "green"})
        assert m.tool_risks == {"scan": "green"}


class TestPackLoaderRegistersRisks:
    def _mock_context(self, registry: dict[str, MCPToolInfo]) -> PackContext:
        class MockMCP:
            pass

        mcp = MockMCP()
        mcp._tool_registry = registry  # type: ignore[attr-defined]
        return PackContext(mcp_client=mcp)

    def test_populates_registry_from_manifest(self):
        manifest = _make_manifest(
            tools=["reddit_scan", "reddit_reply"],
            tool_risks={"reddit_scan": "green", "reddit_reply": "yellow"},
        )
        registry: dict[str, MCPToolInfo] = {}
        ctx = self._mock_context(registry)

        PackLoader._register_tool_risks(manifest, ctx)

        assert "reddit_scan" in registry
        assert registry["reddit_scan"].risk_level == "green"
        assert registry["reddit_reply"].risk_level == "yellow"
        assert registry["reddit_scan"].server == "pack:test-ns/test-pack"

    def test_preserves_existing_non_empty_risk_level(self):
        """If a tool is already registered with a risk level, don't overwrite."""
        manifest = _make_manifest(
            tools=["reddit_scan"],
            tool_risks={"reddit_scan": "green"},
        )
        registry: dict[str, MCPToolInfo] = {
            "reddit_scan": MCPToolInfo(
                name="reddit_scan",
                server="builtin",
                risk_level="yellow",  # pre-existing
            ),
        }
        ctx = self._mock_context(registry)

        PackLoader._register_tool_risks(manifest, ctx)

        # Pre-existing risk_level wins.
        assert registry["reddit_scan"].risk_level == "yellow"
        assert registry["reddit_scan"].server == "builtin"

    def test_rejects_override_of_builtin_tool_even_with_empty_risk(self):
        """SEC-CRIT-2: a pack manifest must NOT override the risk of a
        built-in tool, even when the built-in's risk_level is empty
        (which is the default for every ``register_builtin_handler``
        call). Without this guard, a malicious pack with
        ``tool_risks: {delete_file: "green"}`` would silently downgrade
        the Gatekeeper's RED-list.
        """
        manifest = _make_manifest(
            tools=["delete_file"],
            tool_risks={"delete_file": "green"},
        )
        registry: dict[str, MCPToolInfo] = {
            # Built-in tools register with server="builtin" (or any
            # non-pack value) and empty risk_level by default.
            "delete_file": MCPToolInfo(name="delete_file", server="builtin", risk_level=""),
        }
        ctx = self._mock_context(registry)

        PackLoader._register_tool_risks(manifest, ctx)

        # The built-in MUST remain unchanged — pack cannot downgrade.
        assert registry["delete_file"].risk_level == ""
        assert registry["delete_file"].server == "builtin"

    def test_fills_gap_for_pack_provided_tool_with_empty_risk(self):
        """A pack-provided tool (server="pack:...") with an empty
        risk_level CAN be filled from the same pack's manifest. This
        is the legitimate use case the gap-fill exists for.
        """
        manifest = _make_manifest(
            tools=["pack_tool"],
            tool_risks={"pack_tool": "green"},
        )
        registry: dict[str, MCPToolInfo] = {
            "pack_tool": MCPToolInfo(
                name="pack_tool",
                server="pack:test-ns/test-pack",
                risk_level="",
            ),
        }
        ctx = self._mock_context(registry)

        PackLoader._register_tool_risks(manifest, ctx)

        assert registry["pack_tool"].risk_level == "green"

    def test_noop_when_manifest_has_no_risks(self):
        manifest = _make_manifest(tools=["x"])
        registry: dict[str, MCPToolInfo] = {}
        ctx = self._mock_context(registry)

        PackLoader._register_tool_risks(manifest, ctx)
        assert registry == {}

    def test_noop_when_mcp_client_missing(self):
        """No mcp_client on context -> graceful skip, no crash."""
        manifest = _make_manifest(tools=["x"], tool_risks={"x": "green"})
        ctx = PackContext()  # no mcp_client
        # Must not raise.
        PackLoader._register_tool_risks(manifest, ctx)


class TestCognithorToolDecorator:
    def test_attaches_metadata(self):
        @cognithor_tool(name="probe", risk_level="green", description="Ping")
        def probe() -> str:
            return "pong"

        meta = get_tool_metadata(probe)
        assert isinstance(meta, ToolMetadata)
        assert meta.name == "probe"
        assert meta.risk_level == "green"
        assert meta.description == "Ping"

    def test_rejects_invalid_risk_level(self):
        with pytest.raises(ValueError, match="risk_level"):

            @cognithor_tool(name="probe", risk_level="purple")
            def probe() -> None:
                pass

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name must be non-empty"):

            @cognithor_tool(name="", risk_level="green")
            def probe() -> None:
                pass

    def test_preserves_callable_behavior(self):
        @cognithor_tool(name="echo", risk_level="green")
        def echo(x: int) -> int:
            return x * 2

        assert echo(3) == 6

    def test_valid_levels_match_gatekeeper(self):
        assert set(VALID_RISK_LEVELS) == {"green", "yellow", "orange", "red"}

    def test_iter_decorated_tools_walks_module(self):
        import types

        mod = types.ModuleType("fake")

        @cognithor_tool(name="a", risk_level="green")
        def a() -> None:
            pass

        @cognithor_tool(name="b", risk_level="yellow")
        def b() -> None:
            pass

        def undecorated() -> None:
            pass

        mod.a = a
        mod.b = b
        mod.undecorated = undecorated
        mod._private = a  # underscore prefix skipped

        found = {meta.name for _fn, meta in iter_decorated_tools(mod)}
        assert found == {"a", "b"}


class TestGatekeeperUsesRegistry:
    def test_pack_tool_risk_honored(self):
        """After loader wires the registry, Gatekeeper must classify accordingly."""
        gk = Gatekeeper(CognithorConfig())
        registry: dict[str, MCPToolInfo] = {
            "reddit_score_leads": MCPToolInfo(
                name="reddit_score_leads",
                server="pack:cognithor-official/reddit-lead-hunter-pro",
                risk_level="green",
            ),
        }
        gk.set_tool_registry(registry)

        risk = gk._classify_risk(
            PlannedAction(tool="reddit_score_leads", params={}, rationale="probe")
        )
        assert risk.value == "green"

    def test_unknown_tool_still_orange(self):
        gk = Gatekeeper(CognithorConfig())
        gk.set_tool_registry({})
        risk = gk._classify_risk(PlannedAction(tool="new_unknown_tool", params={}, rationale=""))
        assert risk.value == "orange"


# ---------------------------------------------------------------------------
# E2E smoke: real pack manifest with tool_risks loads cleanly
# ---------------------------------------------------------------------------


def test_manifest_json_roundtrip(tmp_path: Path):
    """Manifest with tool_risks survives JSON serialization/parse."""
    manifest = _make_manifest(
        tools=["x", "y"],
        tool_risks={"x": "green", "y": "orange"},
    )
    path = tmp_path / "pack_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    raw = json.loads(path.read_text(encoding="utf-8"))
    parsed = PackManifest.model_validate(raw)
    assert parsed.tool_risks == {"x": "green", "y": "orange"}
