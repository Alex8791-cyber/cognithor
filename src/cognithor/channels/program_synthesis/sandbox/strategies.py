# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sandbox strategy implementations + platform-aware selection (spec §11.6).

Three concrete strategies, all satisfying the
:class:`~cognithor.channels.program_synthesis.search.executor.Executor`
protocol so the search engine and the equivalence pruner stay
oblivious to which one is wired in:

* :class:`LinuxSubprocessStrategy` — production-grade isolation on
  Linux native (full setrlimit + namespace-drop subprocess worker; the
  setrlimit / namespace plumbing lives in a follow-up PR).
* :class:`WSL2WorkerStrategy` — production-grade isolation on Windows
  via a ``wsl.exe`` worker subprocess. The Linux kernel inside WSL2
  provides the security primitives Windows lacks.
* :class:`WindowsResearchStrategy` — Windows native fallback with
  reduced wall-clock and memory caps and the
  ``pse:synthesize:production`` capability **disabled**.

Phase 1 ships the routing + detection + capability gating; the real
subprocess execution layer is a separate PR (the protocol is the same,
so the swap is transparent).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.integration.capability_tokens import (
    PSECapability,
)
from cognithor.channels.program_synthesis.sandbox.policy import (
    DEFAULT_LIMITS,
    RESEARCH_MODE_LIMITS,
    RESEARCH_MODE_WARNING,
    SandboxLimits,
)
from cognithor.channels.program_synthesis.search.executor import (
    ExecutionResult,
    InProcessExecutor,
)

if TYPE_CHECKING:
    from typing import Any

    from cognithor.channels.program_synthesis.search.candidate import ProgramNode

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyInfo:
    """Static description of a sandbox strategy.

    ``allows_production_capability`` is the spec's gate on
    ``pse:synthesize:production`` — only Linux native and WSL2 grant
    it. The Strategy Router refuses to issue that capability when the
    selected strategy returns False.
    """

    name: str
    description: str
    limits: SandboxLimits
    allows_production_capability: bool
    research_mode: bool = False


# ---------------------------------------------------------------------------
# Strategy implementations.
# ---------------------------------------------------------------------------


class _BaseStrategy:
    """Common boilerplate. Subclasses set ``info`` at construction.

    Phase 1 implementations all delegate to :class:`InProcessExecutor`;
    the production-grade subprocess plumbing slots in here later via
    overriding ``execute``.
    """

    info: StrategyInfo

    def __init__(self) -> None:
        self._inner = InProcessExecutor()

    def execute(self, program: ProgramNode, input_grid: Any) -> ExecutionResult:
        return self._inner.execute(program, input_grid)

    @property
    def limits(self) -> SandboxLimits:
        return self.info.limits

    @property
    def allows_production_capability(self) -> bool:
        return self.info.allows_production_capability


class LinuxSubprocessStrategy(_BaseStrategy):
    """Native Linux subprocess worker with full setrlimit + namespace drop."""

    info = StrategyInfo(
        name="linux-subprocess",
        description="Native Linux subprocess with setrlimit + namespace drop.",
        limits=DEFAULT_LIMITS,
        allows_production_capability=True,
    )


class WSL2WorkerStrategy(_BaseStrategy):
    """Windows + WSL2 worker. Production-grade isolation via the Linux kernel."""

    info = StrategyInfo(
        name="wsl2-worker",
        description="WSL2 subprocess worker invoked via wsl.exe.",
        limits=DEFAULT_LIMITS,
        allows_production_capability=True,
    )


class WindowsResearchStrategy(_BaseStrategy):
    """Windows native fallback with reduced limits + production cap disabled.

    On construction emits a warning to ``logging`` and (for development
    visibility) ``stderr``. The reduced limits are a hard ceiling — any
    Budget the search engine receives is clamped to these values when
    routed through this strategy.
    """

    info = StrategyInfo(
        name="windows-research",
        description=(
            "Native-Windows fallback with reduced wall-clock + memory; "
            "pse:synthesize:production disabled."
        ),
        limits=RESEARCH_MODE_LIMITS,
        allows_production_capability=False,
        research_mode=True,
    )

    def __init__(self, *, emit_warning: bool = True) -> None:
        super().__init__()
        if emit_warning:
            LOG.warning(RESEARCH_MODE_WARNING)


# ---------------------------------------------------------------------------
# Detection.
# ---------------------------------------------------------------------------


def _wsl2_available() -> bool:
    """Heuristic: look for wsl.exe on PATH and verify it can list distros.

    Returns False on platforms other than Windows (so a Linux test run
    doesn't trip the Windows path).
    """
    if sys.platform != "win32":
        return False
    if shutil.which("wsl") is None and shutil.which("wsl.exe") is None:
        return False
    try:
        r = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            timeout=5.0,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and bool((r.stdout or "").strip())


def select_sandbox_strategy(
    *,
    platform_override: str | None = None,
    wsl2_check: bool | None = None,
    emit_warning: bool = True,
) -> _BaseStrategy:
    """Pick the appropriate strategy for the current platform.

    ``platform_override`` and ``wsl2_check`` are dependency-injectable
    for tests — they default to live ``sys.platform`` /
    :func:`_wsl2_available` lookups.

    Resolution table::

        Linux       → LinuxSubprocessStrategy
        Windows+WSL → WSL2WorkerStrategy
        Windows-WSL → WindowsResearchStrategy (with warning)
        Mac / other → LinuxSubprocessStrategy (best-effort POSIX path)
    """
    plat = platform_override if platform_override is not None else sys.platform
    if plat == "linux":
        return LinuxSubprocessStrategy()
    if plat == "win32":
        wsl_present = wsl2_check if wsl2_check is not None else _wsl2_available()
        if wsl_present:
            return WSL2WorkerStrategy()
        return WindowsResearchStrategy(emit_warning=emit_warning)
    if plat == "darwin":
        # Mac is technically a POSIX subset of Linux for our purposes —
        # setrlimit + the subprocess pipe both work. Fall through to
        # the Linux strategy; spec §11.6 doesn't require Mac-specific
        # tracking.
        return LinuxSubprocessStrategy()
    LOG.warning(
        "PSE: unknown platform %r — falling back to Linux subprocess strategy",
        plat,
    )
    return LinuxSubprocessStrategy()


# ---------------------------------------------------------------------------
# Capability allow-list helper.
# ---------------------------------------------------------------------------


def capabilities_for_strategy(strategy: _BaseStrategy) -> tuple[PSECapability, ...]:
    """Return the capability set that *strategy* is allowed to grant.

    Research mode strips ``pse:synthesize:production`` per spec §11.6.
    """
    base: tuple[PSECapability, ...] = (
        PSECapability.SYNTHESIZE,
        PSECapability.EXECUTE,
        PSECapability.CACHE_READ,
        PSECapability.CACHE_WRITE,
    )
    if strategy.allows_production_capability:
        return (PSECapability.SYNTHESIZE_PRODUCTION, *base)
    return base


# Tests sometimes override env to influence detection without touching
# the real wsl.exe — surface the env knob explicitly so test setUp can
# clear it.
WSL_OVERRIDE_ENV = "PSE_FORCE_WSL2_AVAILABLE"


def _wsl2_available_for_tests() -> bool | None:
    """Returns True/False if the env override is set, otherwise None.

    Kept as a separate helper so test code reads naturally; production
    callers should use :func:`_wsl2_available`.
    """
    val = os.environ.get(WSL_OVERRIDE_ENV)
    if val is None:
        return None
    return val.strip().lower() in {"1", "true", "yes"}


__all__ = [
    "WSL_OVERRIDE_ENV",
    "LinuxSubprocessStrategy",
    "StrategyInfo",
    "WSL2WorkerStrategy",
    "WindowsResearchStrategy",
    "capabilities_for_strategy",
    "select_sandbox_strategy",
]
