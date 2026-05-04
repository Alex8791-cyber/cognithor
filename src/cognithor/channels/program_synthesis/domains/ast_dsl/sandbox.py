"""Subprocess sandbox for AST-domain function execution.

Runs synthesised Python functions in a *fresh subprocess* with:

* Hard wall-time limit (default 2 s) — uses ``subprocess.run`` timeout
* Memory limit on POSIX via ``resource.setrlimit`` (best-effort on
  Windows; the wall-time gate still applies)
* No network (the spawned process has no network stack from us, but
  we cannot enforce that without a container — we document this).
* No filesystem writes outside ``tempfile.gettempdir()`` (advisory).

The sandbox is intentionally minimal — it's *not* a security boundary
against malicious code (a hostile LLM output could escape this
trivially). It IS a safety net against synthesised functions that
infinite-loop, exhaust memory, or block on stdin. For real
adversarial isolation we'd need bwrap / firejail / containers — that
work is the existing ``cognithor.security.sandbox`` module's
responsibility, not Sprint-26.3's.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SandboxConfig:
    """Resource limits for one sandbox run."""

    timeout_seconds: float = 2.0
    memory_mb: int = 128
    enforce_memory: bool = True


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of a sandbox execution."""

    ok: bool
    value: Any = None
    error_kind: str = ""  # "timeout" | "exception" | "memory" | "" (success)
    error_message: str = ""
    duration_ms: float = 0.0


_RUNNER_TEMPLATE = textwrap.dedent(
    """
    import json, sys, time
    {memory_setup}

    {function_source}

    args = json.loads(sys.argv[1])
    kwargs = json.loads(sys.argv[2])
    fn_name = sys.argv[3]
    fn = globals()[fn_name]
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout.write(json.dumps({{"ok": True, "value": result, "duration_ms": elapsed_ms}}))
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        msg = f"{{type(exc).__name__}}: {{exc}}"
        sys.stdout.write(json.dumps({{
            "ok": False,
            "error_kind": "exception",
            "error_message": msg,
            "duration_ms": elapsed_ms,
        }}))
    """
).strip()


def _memory_setup_snippet(config: SandboxConfig) -> str:
    if not config.enforce_memory:
        return ""
    # POSIX-only: setrlimit. Windows fallback = no-op (timeout still enforced).
    bytes_limit = config.memory_mb * 1024 * 1024
    return textwrap.dedent(
        f"""
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, ({bytes_limit}, {bytes_limit}))
        except (ImportError, ValueError, OSError):
            pass
        """
    ).strip()


def run_in_sandbox(
    function_source: str,
    function_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute ``function_source`` in a subprocess sandbox.

    Parameters
    ----------
    function_source:
        Complete Python source containing the target function. Must
        define ``function_name``.
    function_name:
        The name of the function to invoke.
    args:
        Positional arguments — must be JSON-serialisable.
    kwargs:
        Keyword arguments — must be JSON-serialisable.
    config:
        Sandbox config (timeout + memory). Defaults to 2 s / 128 MB.

    Returns
    -------
    :class:`SandboxResult` with ``ok=True`` + ``value`` on success, or
    ``ok=False`` + ``error_kind`` ("timeout"/"exception"/"memory") on
    failure.
    """
    cfg = config or SandboxConfig()
    runner = _RUNNER_TEMPLATE.format(
        memory_setup=_memory_setup_snippet(cfg),
        function_source=function_source,
    )
    payload_args = json.dumps(list(args))
    payload_kwargs = json.dumps(dict(kwargs or {}))
    try:
        completed = subprocess.run(
            [sys.executable, "-c", runner, payload_args, payload_kwargs, function_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False,
            error_kind="timeout",
            error_message=f"exceeded {cfg.timeout_seconds}s wall-time",
            duration_ms=cfg.timeout_seconds * 1000.0,
        )

    if completed.returncode != 0:
        # Memory-limit kills typically yield SIGKILL; surface as
        # "memory" so the caller can route accordingly.
        kind = "memory" if completed.returncode in (-9, 137) else "exception"
        return SandboxResult(
            ok=False,
            error_kind=kind,
            error_message=(completed.stderr or "subprocess failed").strip()[:500],
        )

    try:
        body = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return SandboxResult(
            ok=False,
            error_kind="exception",
            error_message=f"sandbox JSON decode failed: {exc}",
        )

    if body.get("ok"):
        return SandboxResult(
            ok=True,
            value=body.get("value"),
            duration_ms=float(body.get("duration_ms", 0.0)),
        )
    return SandboxResult(
        ok=False,
        error_kind=str(body.get("error_kind", "exception")),
        error_message=str(body.get("error_message", ""))[:500],
        duration_ms=float(body.get("duration_ms", 0.0)),
    )
