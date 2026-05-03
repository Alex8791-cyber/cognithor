# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tests for Sprint-23 PR#C — ``router.context_profile_scope`` helper."""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from cognithor.config import CognithorConfig
from cognithor.core.model_router import (
    ModelRouter,
    OllamaClient,
    _context_profile_var,
)


@pytest.fixture()
def config(tmp_path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_path)


@pytest.fixture()
def client(config: CognithorConfig) -> OllamaClient:
    return OllamaClient(config)


@pytest.fixture()
def router(config: CognithorConfig, client: OllamaClient) -> ModelRouter:
    return ModelRouter(config, client)


@pytest.fixture(autouse=True)
def _reset():
    _context_profile_var.set(None)
    yield
    _context_profile_var.set(None)


# ---------------------------------------------------------------------------
# Basic scope semantics
# ---------------------------------------------------------------------------


class TestContextProfileScopeBasics:
    def test_enters_and_exits_cleanly(self, router: ModelRouter) -> None:
        assert router.get_context_profile() is None
        with router.context_profile_scope("deep"):
            assert router.get_context_profile() == "deep"
        assert router.get_context_profile() is None

    def test_get_model_config_sees_scoped_profile(
        self, router: ModelRouter, config: CognithorConfig
    ) -> None:
        with router.context_profile_scope("arc_agi3"):
            cfg = router.get_model_config(config.models.planner.name)
            assert cfg["context_window"] == 131072
        cfg_after = router.get_model_config(config.models.planner.name)
        assert cfg_after["context_window"] == config.models.planner.context_window

    def test_unknown_profile_raises_before_yield(self, router: ModelRouter) -> None:
        # The validation must happen at *enter*, not somewhere inside
        # the user's with-body — otherwise a typoed profile name would
        # silently pass through and only blow up later.
        with pytest.raises(ValueError, match="Unknown context profile"):
            with router.context_profile_scope("nonexistent"):
                pytest.fail("with-body should not have executed")


# ---------------------------------------------------------------------------
# Nested scopes restore the previous value, not None
# ---------------------------------------------------------------------------


class TestNestedScopes:
    def test_nested_scopes_restore_outer_profile(self, router: ModelRouter) -> None:
        # Outer ``deep`` must come back when the inner ``quick`` scope
        # exits — *not* fall through to None. The previous code-path
        # always wrote None on exit, which would silently drop a
        # caller's earlier profile selection.
        with router.context_profile_scope("deep"):
            assert router.get_context_profile() == "deep"
            with router.context_profile_scope("quick"):
                assert router.get_context_profile() == "quick"
            assert router.get_context_profile() == "deep"
        assert router.get_context_profile() is None

    def test_set_then_scope_then_exit_restores_set_value(self, router: ModelRouter) -> None:
        # Caller already set ``arc_agi3`` via the imperative API; a
        # nested scope flip and exit must put it back, not clear it.
        router.set_context_profile("arc_agi3")
        with router.context_profile_scope("default"):
            assert router.get_context_profile() == "default"
        assert router.get_context_profile() == "arc_agi3"


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


class TestExceptionSafety:
    def test_exception_inside_scope_still_restores(self, router: ModelRouter) -> None:
        # Without the try/finally, an exception inside the with-body
        # would leak the wider window into the next request and trigger
        # GPU OOM. This is the core reason the helper exists.
        with pytest.raises(RuntimeError):
            with router.context_profile_scope("arc_agi3"):
                assert router.get_context_profile() == "arc_agi3"
                raise RuntimeError("boom")
        assert router.get_context_profile() is None

    def test_exception_inside_nested_scope_restores_outer(self, router: ModelRouter) -> None:
        with router.context_profile_scope("deep"):
            with pytest.raises(RuntimeError):
                with router.context_profile_scope("quick"):
                    raise RuntimeError("boom")
            # Outer scope is intact even though inner raised.
            assert router.get_context_profile() == "deep"
        assert router.get_context_profile() is None


# ---------------------------------------------------------------------------
# Concurrency — two asyncio tasks each get their own scope
# ---------------------------------------------------------------------------


class TestConcurrentScopes:
    def test_two_tasks_dont_leak_through_scope(self, router: ModelRouter) -> None:
        observed: dict[str, str | None] = {}

        async def _task(profile_name: str) -> None:
            ctx = contextvars.copy_context()

            def _inside() -> None:
                with router.context_profile_scope(profile_name):
                    observed[profile_name] = router.get_context_profile()

            ctx.run(_inside)

        async def _drive() -> None:
            await asyncio.gather(
                _task("quick"),
                _task("arc_agi3"),
                _task("deep"),
            )

        asyncio.run(_drive())
        assert observed == {
            "quick": "quick",
            "arc_agi3": "arc_agi3",
            "deep": "deep",
        }
        # Outer test scope is untouched.
        assert router.get_context_profile() is None
