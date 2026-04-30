# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Unit tests for the subprocess sandbox runner (spec §11.5 + §11.6).

The runner is the worker behind K4. The full adversarial-payload
matrix lives in ``security/test_adversarial.py``; these tests cover
the *runner itself* — happy path, decode errors, and the typed-error
contract on the unsupported-platform branch.
"""

from __future__ import annotations

import sys

import pytest

# Import an integration symbol first so the integration package fully
# loads before the sandbox package — avoids the existing
# integration ⇄ sandbox.strategies circular at import time.
from cognithor.channels.program_synthesis.integration.capability_tokens import (
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.sandbox.policy import SandboxLimits
from cognithor.channels.program_synthesis.sandbox.runner import (
    RunResult,
    run_in_sandbox,
)

_ = _PSECapability  # keep the import non-dead

_skip_on_windows_native = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")),
    reason="subprocess sandbox is POSIX-only (Windows uses WSL2 worker).",
)


class TestRunResult:
    def test_ok_factory(self) -> None:
        r = RunResult(ok=True, error="ok", value=42)
        assert r.ok is True
        assert r.value == 42

    def test_error_factory(self) -> None:
        r = RunResult(ok=False, error="WallClockExceeded")
        assert r.ok is False
        assert r.error == "WallClockExceeded"
        assert r.value is None


@_skip_on_windows_native
class TestHappyPath:
    def test_simple_callable_runs_to_completion(self) -> None:
        # ``json:dumps`` is a stdlib function — safe target for the
        # smoke check. The worker imports it, calls dumps([1,2,3]),
        # returns the value as JSON.
        result = run_in_sandbox("json:dumps", [1, 2, 3])
        assert result.ok is True
        assert result.error == "ok"
        # json.dumps returns a string '"[1, 2, 3]"' — the worker
        # marshals strings as-is.
        assert result.value == "[1, 2, 3]"

    def test_returns_typed_error_on_unknown_module(self) -> None:
        result = run_in_sandbox("nonexistent_module_xyz:fn", None)
        assert result.ok is False
        # Import inside the worker fails before limits apply.
        assert result.error in {"WorkerCrashed", "DecodeError"}

    def test_target_must_have_module_colon_attr_shape(self) -> None:
        result = run_in_sandbox("not_a_target", None)
        assert result.ok is False
        assert result.error == "DecodeError"


class TestUnsupportedPlatform:
    def test_windows_native_returns_typed_error(self) -> None:
        # If we're actually on Windows native, the runner refuses with
        # UnsupportedPlatform. On Linux/macOS this test is a no-op.
        if sys.platform != "win32":
            pytest.skip("only meaningful on Windows native")
        result = run_in_sandbox("json:dumps", None)
        assert result.ok is False
        assert result.error == "UnsupportedPlatform"


class TestPayloadEncodingErrors:
    def test_non_json_arg_rejected_before_spawn(self) -> None:
        class _NotJsonable:
            pass

        result = run_in_sandbox("json:dumps", _NotJsonable())
        assert result.ok is False
        assert result.error == "DecodeError"


@_skip_on_windows_native
class TestLimitsArePropagated:
    def test_short_wall_clock_terminates_busy_loop(self) -> None:
        # Sanity for the wall-clock path without going through the
        # adversarial payload module — calls a 1-second sleep with a
        # 0.2s cap.
        result = run_in_sandbox(
            "time:sleep",
            5.0,
            limits=SandboxLimits(wall_clock_seconds=0.2, memory_mb=256, per_candidate_ms=100),
        )
        assert result.ok is False
        assert result.error == "WallClockExceeded"
