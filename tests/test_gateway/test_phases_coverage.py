"""Coverage-Tests fuer gateway/phases/ -- alle 8 Phasen-Module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.config import CognithorConfig, ensure_directory_structure


@pytest.fixture()
def config(tmp_path) -> CognithorConfig:
    cfg = CognithorConfig(cognithor_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


# ============================================================================
# phases/core.py
# ============================================================================


class TestCorePhase:
    def test_declare_core_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.core import declare_core_attrs

        result = declare_core_attrs(config)
        assert "ollama" in result
        assert "llm" in result
        assert "model_router" in result
        assert "session_store" in result
        assert all(v is None for v in result.values())

    @pytest.mark.asyncio
    async def test_init_core_llm_available(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.core import init_core

        mock_llm = MagicMock()
        mock_llm._ollama = MagicMock()
        mock_llm._backend = None
        mock_llm.is_available = AsyncMock(return_value=True)
        mock_llm.backend_type = "ollama"

        mock_router = MagicMock()
        mock_router.initialize = AsyncMock()

        with (
            patch("cognithor.core.unified_llm.UnifiedLLMClient.create", return_value=mock_llm),
            patch("cognithor.core.model_router.ModelRouter", return_value=mock_router),
            patch("cognithor.gateway.session_store.SessionStore") as MockStore,
        ):
            MockStore.return_value = MagicMock(count_sessions=MagicMock(return_value=0))

            result = await init_core(config)
            assert result["__llm_ok"] is True
            assert result["llm"] is mock_llm

    @pytest.mark.asyncio
    async def test_init_core_llm_not_available_ollama(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.core import init_core

        mock_llm = MagicMock()
        mock_llm._ollama = MagicMock()
        mock_llm._backend = None
        mock_llm.is_available = AsyncMock(return_value=False)
        mock_llm.backend_type = "ollama"

        with (
            patch("cognithor.core.unified_llm.UnifiedLLMClient.create", return_value=mock_llm),
            patch("cognithor.core.model_router.ModelRouter") as MockRouter,
            patch("cognithor.gateway.session_store.SessionStore") as MockStore,
        ):
            MockRouter.return_value = MagicMock()
            MockStore.return_value = MagicMock(count_sessions=MagicMock(return_value=0))

            result = await init_core(config)
            assert result["__llm_ok"] is False

    @pytest.mark.asyncio
    async def test_init_core_llm_not_available_lmstudio(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.core import init_core

        mock_llm = MagicMock()
        mock_llm._ollama = MagicMock()
        mock_llm._backend = None
        mock_llm.is_available = AsyncMock(return_value=False)
        mock_llm.backend_type = "lmstudio"

        with (
            patch("cognithor.core.unified_llm.UnifiedLLMClient.create", return_value=mock_llm),
            patch("cognithor.core.model_router.ModelRouter") as MockRouter,
            patch("cognithor.gateway.session_store.SessionStore") as MockStore,
        ):
            MockRouter.return_value = MagicMock()
            MockStore.return_value = MagicMock(count_sessions=MagicMock(return_value=0))

            result = await init_core(config)
            assert result["__llm_ok"] is False

    @pytest.mark.asyncio
    async def test_init_core_llm_not_available_other(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.core import init_core

        mock_llm = MagicMock()
        mock_llm._ollama = MagicMock()
        mock_llm._backend = MagicMock()  # has backend
        mock_llm.is_available = AsyncMock(return_value=False)
        mock_llm.backend_type = "openai"

        mock_router = MagicMock()

        with (
            patch("cognithor.core.unified_llm.UnifiedLLMClient.create", return_value=mock_llm),
            patch("cognithor.core.model_router.ModelRouter") as MockRouter,
            patch("cognithor.gateway.session_store.SessionStore") as MockStore,
        ):
            MockRouter.from_backend.return_value = mock_router
            MockStore.return_value = MagicMock(count_sessions=MagicMock(return_value=0))

            result = await init_core(config)
            assert result["__llm_ok"] is False

    @pytest.mark.asyncio
    async def test_init_core_with_backend(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.core import init_core

        mock_llm = MagicMock()
        mock_llm._ollama = None
        mock_llm._backend = MagicMock()
        mock_llm.is_available = AsyncMock(return_value=True)
        mock_llm.backend_type = "openai"

        mock_router = MagicMock()
        mock_router.initialize = AsyncMock()

        with (
            patch("cognithor.core.unified_llm.UnifiedLLMClient.create", return_value=mock_llm),
            patch("cognithor.core.model_router.ModelRouter") as MockRouter,
            patch("cognithor.gateway.session_store.SessionStore") as MockStore,
        ):
            MockRouter.from_backend.return_value = mock_router
            MockStore.return_value = MagicMock(count_sessions=MagicMock(return_value=0))

            result = await init_core(config)
            assert result["__llm_ok"] is True


# ============================================================================
# phases/security.py
# ============================================================================


class TestSecurityPhase:
    def test_declare_security_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.security import declare_security_attrs

        result = declare_security_attrs(config)
        assert "audit_logger" in result
        assert "gatekeeper" in result

    @pytest.mark.asyncio
    async def test_init_security(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.security import init_security

        result = await init_security(config)
        assert "audit_logger" in result
        assert "gatekeeper" in result


# ============================================================================
# phases/memory.py
# ============================================================================


class TestMemoryPhase:
    def test_declare_memory_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.memory import declare_memory_attrs

        result = declare_memory_attrs(config)
        assert "memory_manager" in result

    @pytest.mark.asyncio
    async def test_init_memory(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.memory import init_memory

        mock_audit = MagicMock()
        mock_mm = MagicMock()
        mock_mm.initialize = AsyncMock(return_value={"chunks": 0, "entities": 0})

        with patch("cognithor.memory.manager.MemoryManager", return_value=mock_mm):
            result = await init_memory(config, audit_logger=mock_audit)
            assert "memory_manager" in result

    @pytest.mark.asyncio
    async def test_init_memory_failure(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.memory import init_memory

        mock_mm = MagicMock()
        mock_mm.initialize = AsyncMock(side_effect=Exception("DB error"))

        with patch("cognithor.memory.manager.MemoryManager", return_value=mock_mm):
            result = await init_memory(config, audit_logger=MagicMock())
            assert "memory_manager" in result


# ============================================================================
# phases/tools.py
# ============================================================================


class TestToolsPhase:
    def test_declare_tools_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.tools import declare_tools_attrs

        result = declare_tools_attrs(config)
        assert "mcp_client" in result

    @pytest.mark.asyncio
    async def test_init_tools(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.tools import init_tools

        mock_mcp = MagicMock()
        mock_mm = MagicMock()
        result = await init_tools(config, mcp_client=mock_mcp, memory_manager=mock_mm)
        assert isinstance(result, dict)


# ============================================================================
# phases/tools.py — Sprint-26 PSE domain registration wiring
#
# These tests cover the integration point added in Sprint-26 where
# init_tools centralises registration of the seven Domain-Expansion
# catalogs (sql, json, datetime, ast, bytes, float, image_v2) into
# the canonical DOMAIN_REGISTRY. Without this wiring the registry
# stays empty at runtime and every cross-domain synthesis query falls
# back to the free-form path.
#
# Each test uses a fresh in-process DomainRegistry monkey-patched
# into BOTH the module that exports it AND the registry submodule, so
# the function-local ``from … import DOMAIN_REGISTRY`` inside
# init_tools picks up the test's instance instead of the global
# singleton (which other tests in the session may have populated).
# ============================================================================


class TestToolsPhaseSprint26Registration:
    @pytest.fixture
    def fresh_domain_registry(self, monkeypatch):
        """Inject a clean DomainRegistry for the duration of this test."""
        from cognithor.channels.program_synthesis.domains.registry import (
            DomainRegistry,
        )

        fresh = DomainRegistry()
        # Patch BOTH locations — the package re-exports DOMAIN_REGISTRY
        # at the package level for ergonomics, and init_tools imports
        # via the package re-export.
        monkeypatch.setattr(
            "cognithor.channels.program_synthesis.domains.registry.DOMAIN_REGISTRY",
            fresh,
        )
        monkeypatch.setattr(
            "cognithor.channels.program_synthesis.domains.DOMAIN_REGISTRY",
            fresh,
        )
        return fresh

    @pytest.mark.asyncio
    async def test_init_tools_registers_all_seven_sprint26_domains(
        self, config: CognithorConfig, fresh_domain_registry
    ) -> None:
        """After init_tools runs against an empty registry, all 7
        Sprint-26 domains must be present (Owner-D3 catalog complete)."""
        from cognithor.channels.program_synthesis.domains import (
            SPRINT26_DOMAIN_NAMES,
        )
        from cognithor.gateway.phases.tools import init_tools

        assert len(fresh_domain_registry) == 0, "fixture must start empty"

        await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())

        for name in SPRINT26_DOMAIN_NAMES:
            assert name in fresh_domain_registry, (
                f"Sprint-26 domain {name!r} missing after init_tools"
            )
        assert len(fresh_domain_registry) == 7

    @pytest.mark.asyncio
    async def test_init_tools_idempotent_on_double_boot(
        self, config: CognithorConfig, fresh_domain_registry
    ) -> None:
        """Running init_tools twice in the same process (test fixtures,
        hot-reload) must NOT raise DomainAlreadyRegisteredError. The
        idempotent path is what register_missing_sprint26_domains
        guarantees."""
        from cognithor.gateway.phases.tools import init_tools

        await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())
        # Second boot must not raise
        await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())
        # Registry stayed at 7 — no duplicates
        assert len(fresh_domain_registry) == 7

    @pytest.mark.asyncio
    async def test_init_tools_partial_registry_fills_gaps(
        self, config: CognithorConfig, fresh_domain_registry
    ) -> None:
        """If the registry was partially pre-seeded (rare but possible
        with multi-process test runners or hand-crafted fixtures),
        init_tools must fill the gaps without error and reach the
        expected 7-domain steady state."""
        from cognithor.channels.program_synthesis.domains.sql import (
            register_sql_domain,
        )
        from cognithor.gateway.phases.tools import init_tools

        register_sql_domain(fresh_domain_registry)
        assert len(fresh_domain_registry) == 1

        await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())

        from cognithor.channels.program_synthesis.domains import (
            SPRINT26_DOMAIN_NAMES,
        )

        for name in SPRINT26_DOMAIN_NAMES:
            assert name in fresh_domain_registry
        assert len(fresh_domain_registry) == 7

    @pytest.mark.asyncio
    async def test_init_tools_preserves_owner_d3_priority_order(
        self, config: CognithorConfig, fresh_domain_registry
    ) -> None:
        """Owner-Decision D3 fixes the registration order: SQL → JSON
        → Datetime → AST → BinaryData → Float → Image-Boost. The
        public scorecard depends on this ordering, so the gateway-boot
        path must preserve it (even though name-lookup wouldn't care)."""
        from cognithor.channels.program_synthesis.domains import (
            SPRINT26_DOMAIN_NAMES,
        )
        from cognithor.gateway.phases.tools import init_tools

        await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())

        # Iterate over the registry in registration order — registry's
        # internal _metadata dict preserves insertion order (PEP 468).
        registered_in_order = list(fresh_domain_registry._metadata.keys())
        # Filter to just the Sprint-26 names (in case other things land
        # in the registry alongside)
        sprint26_only = [n for n in registered_in_order if n in set(SPRINT26_DOMAIN_NAMES)]
        assert tuple(sprint26_only) == SPRINT26_DOMAIN_NAMES

    @pytest.mark.asyncio
    async def test_init_tools_logs_newly_registered_names(
        self,
        config: CognithorConfig,
        fresh_domain_registry,
    ) -> None:
        """First-boot must emit ``pse_sprint26_domains_registered`` with
        the names actually registered. Reviewers can then tell first-
        boot from re-boot at a glance in audit logs.

        The project uses structlog, which does NOT propagate to
        pytest's caplog. We patch the module-level ``log.info`` instead
        to capture structured events directly.
        """
        from cognithor.channels.program_synthesis.domains import (
            SPRINT26_DOMAIN_NAMES,
        )
        from cognithor.gateway.phases import tools as tools_module

        captured: list[tuple[str, dict]] = []

        original_info = tools_module.log.info

        def capture_info(event: str, **kwargs):  # type: ignore[no-untyped-def]
            captured.append((event, kwargs))
            return original_info(event, **kwargs)

        with patch.object(tools_module.log, "info", side_effect=capture_info):
            await tools_module.init_tools(
                config, mcp_client=MagicMock(), memory_manager=MagicMock()
            )

        # Find the Sprint-26 registered event among all captured info events
        sprint26_events = [
            (event, kwargs)
            for event, kwargs in captured
            if event == "pse_sprint26_domains_registered"
        ]
        assert sprint26_events, (
            f"expected ``pse_sprint26_domains_registered`` event, got events: "
            f"{[e for e, _ in captured]}"
        )
        event, kwargs = sprint26_events[0]
        # The wiring forwards the helper's return value (list of newly
        # registered names) under the ``domains`` key — that's the
        # contract the audit-log readers depend on.
        assert "domains" in kwargs, f"event {event!r} missing ``domains`` kwarg: {kwargs}"
        assert set(kwargs["domains"]) == set(SPRINT26_DOMAIN_NAMES)
        # And the ``total`` field must reflect the registry size after
        # this call so log readers don't have to count.
        assert kwargs.get("total") == 7

    @pytest.mark.asyncio
    async def test_init_tools_swallows_pse_import_error(
        self, config: CognithorConfig, monkeypatch
    ) -> None:
        """If the program_synthesis channel import path raises (e.g.
        the package is shipped without optional deps in a constrained
        deploy), init_tools must continue and complete — Sprint-26
        registration is best-effort, not gateway-fatal."""
        import sys

        from cognithor.gateway.phases.tools import init_tools

        # Block the import that the inner try/except depends on
        if isinstance(__builtins__, dict):
            original_import = __builtins__["__import__"]
        else:
            original_import = __builtins__.__import__

        def raising_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "cognithor.channels.program_synthesis.domains" or name.startswith(
                "cognithor.channels.program_synthesis.domains."
            ):
                raise ImportError(f"forced-fail: {name}")
            return original_import(name, *args, **kwargs)

        # Drop any cached imports first so the next import attempt
        # actually goes through __import__
        for mod_name in list(sys.modules):
            if mod_name.startswith("cognithor.channels.program_synthesis.domains"):
                del sys.modules[mod_name]

        if isinstance(__builtins__, dict):
            monkeypatch.setitem(__builtins__, "__import__", raising_import)
        else:
            monkeypatch.setattr(__builtins__, "__import__", raising_import)

        # Must NOT raise — the inner try/except absorbs ImportError
        result = await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())
        # init_tools still returns the result dict for the rest of the
        # tool subsystems (mcp_bridge, telemetry_hub, etc.)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_init_tools_does_not_register_duplicate_on_repeat_full_boot(
        self, config: CognithorConfig, fresh_domain_registry
    ) -> None:
        """Repeated boots leave the registry at exactly 7 domains —
        no duplicate factories, no spurious metadata entries."""
        from cognithor.gateway.phases.tools import init_tools

        for _ in range(3):
            await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())

        assert len(fresh_domain_registry) == 7
        # Each metadata block should be the canonical singleton — id()
        # comparison after multiple boots confirms no replacement happened.
        from cognithor.channels.program_synthesis.domains import (
            SPRINT26_DOMAIN_NAMES,
        )

        first_ids = {n: id(fresh_domain_registry.metadata(n)) for n in SPRINT26_DOMAIN_NAMES}
        # Trigger another boot
        await init_tools(config, mcp_client=MagicMock(), memory_manager=MagicMock())
        for n in SPRINT26_DOMAIN_NAMES:
            assert id(fresh_domain_registry.metadata(n)) == first_ids[n], (
                f"metadata for {n!r} was replaced — registration is not idempotent"
            )


# ============================================================================
# phases/pge.py
# ============================================================================


class TestPGEPhase:
    def test_declare_pge_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.pge import declare_pge_attrs

        result = declare_pge_attrs(config)
        assert "planner" in result
        assert "executor" in result
        assert "reflector" in result

    @pytest.mark.asyncio
    async def test_init_pge_with_llm(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.pge import init_pge

        mock_llm = MagicMock()
        mock_llm._ollama = MagicMock()
        mock_mcp = MagicMock()
        mock_router = MagicMock()

        result = await init_pge(
            config,
            llm=mock_llm,
            mcp_client=mock_mcp,
            model_router=mock_router,
            runtime_monitor=MagicMock(),
            audit_logger=MagicMock(),
        )
        assert result["planner"] is not None
        assert result["executor"] is not None

    @pytest.mark.asyncio
    async def test_init_pge_no_llm(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.pge import init_pge

        result = await init_pge(
            config,
            llm=None,
            mcp_client=None,
            model_router=None,
            runtime_monitor=None,
            audit_logger=None,
        )
        # PGE always creates Planner/Executor/Reflector (even without LLM)
        assert "planner" in result
        assert "executor" in result
        assert "reflector" in result


# ============================================================================
# phases/agents.py
# ============================================================================


class TestAgentsPhase:
    def test_declare_agents_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.agents import declare_agents_attrs

        result = declare_agents_attrs(config)
        assert "agent_router" in result

    @pytest.mark.asyncio
    async def test_init_agents(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.agents import init_agents

        result = await init_agents(
            config,
            memory_manager=MagicMock(),
            mcp_client=MagicMock(),
            audit_logger=MagicMock(),
            cognithor_home=config.cognithor_home,
        )
        assert "agent_router" in result


# ============================================================================
# phases/advanced.py
# ============================================================================


class TestAdvancedPhase:
    def test_declare_advanced_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.advanced import declare_advanced_attrs

        result = declare_advanced_attrs(config)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_init_advanced(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.advanced import init_advanced

        result = await init_advanced(config)
        assert isinstance(result, dict)


# ============================================================================
# phases/compliance.py
# ============================================================================


class TestCompliancePhase:
    def test_declare_compliance_attrs(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.compliance import declare_compliance_attrs

        result = declare_compliance_attrs(config)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_init_compliance(self, config: CognithorConfig) -> None:
        from cognithor.gateway.phases.compliance import init_compliance

        result = await init_compliance(config)
        assert isinstance(result, dict)
