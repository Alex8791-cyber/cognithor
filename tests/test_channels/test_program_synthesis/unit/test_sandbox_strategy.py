# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sandbox strategy + platform-detection tests (spec §11.6)."""

from __future__ import annotations

import logging

import numpy as np

from cognithor.channels.program_synthesis.integration.capability_tokens import (
    PSECapability,
)
from cognithor.channels.program_synthesis.sandbox import (
    DEFAULT_LIMITS,
    RESEARCH_MODE_LIMITS,
    RESEARCH_MODE_WARNING,
    LinuxSubprocessStrategy,
    SandboxLimits,
    WindowsResearchStrategy,
    WSL2WorkerStrategy,
    capabilities_for_strategy,
    select_sandbox_strategy,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Limits constants
# ---------------------------------------------------------------------------


class TestLimits:
    def test_default_limits_match_spec(self) -> None:
        # Spec §11.2.
        assert DEFAULT_LIMITS.wall_clock_seconds == 30.0
        assert DEFAULT_LIMITS.memory_mb == 256
        assert DEFAULT_LIMITS.per_candidate_ms == 100

    def test_research_mode_limits_reduced(self) -> None:
        # Spec §11.6: 10s / 256 MB / 100 ms.
        assert RESEARCH_MODE_LIMITS.wall_clock_seconds == 10.0
        assert RESEARCH_MODE_LIMITS.memory_mb == 256
        assert RESEARCH_MODE_LIMITS.per_candidate_ms == 100
        # Strictly tighter wall-clock than DEFAULT_LIMITS.
        assert RESEARCH_MODE_LIMITS.wall_clock_seconds < DEFAULT_LIMITS.wall_clock_seconds

    def test_sandbox_limits_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        lim = SandboxLimits(wall_clock_seconds=1.0, memory_mb=1, per_candidate_ms=1)
        with pytest.raises(FrozenInstanceError):
            lim.memory_mb = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


class TestLinuxSubprocessStrategy:
    def test_info(self) -> None:
        s = LinuxSubprocessStrategy()
        assert s.info.name == "linux-subprocess"
        assert s.info.allows_production_capability
        assert not s.info.research_mode

    def test_executes_via_in_process_fallback(self) -> None:
        s = LinuxSubprocessStrategy()
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = s.execute(prog, _g([[1, 2], [3, 4]]))
        assert result.ok
        assert np.array_equal(result.value, _g([[3, 1], [4, 2]]))

    def test_limits_default(self) -> None:
        s = LinuxSubprocessStrategy()
        assert s.limits == DEFAULT_LIMITS


class TestWSL2WorkerStrategy:
    def test_info(self) -> None:
        s = WSL2WorkerStrategy()
        assert s.info.name == "wsl2-worker"
        assert s.info.allows_production_capability

    def test_limits_default(self) -> None:
        s = WSL2WorkerStrategy()
        assert s.limits == DEFAULT_LIMITS


class TestWindowsResearchStrategy:
    def test_info(self) -> None:
        # Suppress the warning during the test.
        s = WindowsResearchStrategy(emit_warning=False)
        assert s.info.name == "windows-research"
        assert not s.info.allows_production_capability
        assert s.info.research_mode

    def test_emits_warning_on_construction(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            WindowsResearchStrategy()
        # The warning text must appear at least once.
        assert any(RESEARCH_MODE_WARNING in r.message for r in caplog.records)

    def test_no_warning_when_suppressed(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            WindowsResearchStrategy(emit_warning=False)
        assert not any(RESEARCH_MODE_WARNING in r.message for r in caplog.records)

    def test_limits_reduced(self) -> None:
        s = WindowsResearchStrategy(emit_warning=False)
        assert s.limits == RESEARCH_MODE_LIMITS


# ---------------------------------------------------------------------------
# Platform-aware selection
# ---------------------------------------------------------------------------


class TestSelectSandboxStrategy:
    def test_linux_uses_subprocess_strategy(self) -> None:
        s = select_sandbox_strategy(platform_override="linux")
        assert isinstance(s, LinuxSubprocessStrategy)

    def test_windows_with_wsl2_uses_wsl_strategy(self) -> None:
        s = select_sandbox_strategy(platform_override="win32", wsl2_check=True)
        assert isinstance(s, WSL2WorkerStrategy)

    def test_windows_without_wsl2_uses_research_strategy(self) -> None:
        s = select_sandbox_strategy(
            platform_override="win32",
            wsl2_check=False,
            emit_warning=False,
        )
        assert isinstance(s, WindowsResearchStrategy)

    def test_darwin_uses_subprocess_strategy(self) -> None:
        s = select_sandbox_strategy(platform_override="darwin")
        assert isinstance(s, LinuxSubprocessStrategy)

    def test_unknown_platform_falls_back_to_linux(self) -> None:
        s = select_sandbox_strategy(platform_override="exotic-os-9")
        assert isinstance(s, LinuxSubprocessStrategy)


# ---------------------------------------------------------------------------
# Capability allow-list
# ---------------------------------------------------------------------------


class TestCapabilitiesForStrategy:
    def test_linux_grants_production_capability(self) -> None:
        caps = capabilities_for_strategy(LinuxSubprocessStrategy())
        assert PSECapability.SYNTHESIZE_PRODUCTION in caps
        assert PSECapability.SYNTHESIZE in caps

    def test_wsl2_grants_production_capability(self) -> None:
        caps = capabilities_for_strategy(WSL2WorkerStrategy())
        assert PSECapability.SYNTHESIZE_PRODUCTION in caps

    def test_research_mode_strips_production_capability(self) -> None:
        caps = capabilities_for_strategy(WindowsResearchStrategy(emit_warning=False))
        assert PSECapability.SYNTHESIZE_PRODUCTION not in caps
        # Base set still present.
        assert PSECapability.SYNTHESIZE in caps
        assert PSECapability.EXECUTE in caps
        assert PSECapability.CACHE_READ in caps
        assert PSECapability.CACHE_WRITE in caps

    def test_no_admin_capabilities_granted_by_default(self) -> None:
        # Admin capabilities (DSL_EXTEND/DSL_TUNE) are never auto-granted
        # by the strategy router — they require explicit operator action.
        for strat in (
            LinuxSubprocessStrategy(),
            WSL2WorkerStrategy(),
            WindowsResearchStrategy(emit_warning=False),
        ):
            caps = capabilities_for_strategy(strat)
            assert PSECapability.DSL_EXTEND not in caps
            assert PSECapability.DSL_TUNE not in caps
