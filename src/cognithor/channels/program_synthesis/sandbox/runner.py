# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Subprocess sandbox runner — K4 hard gate (spec §11.5 + §11.6).

Runs an arbitrary callable in a *fresh subprocess* with POSIX
``setrlimit`` resource caps applied before the callable executes:

* wall-clock — enforced by the parent via ``subprocess.communicate(timeout=...)``,
  with ``proc.kill()`` if the worker hangs.
* memory — ``RLIMIT_AS`` clamp.
* file descriptors — ``RLIMIT_NOFILE`` clamp (also blocks new sockets,
  because ``socket()`` consumes an FD).
* child processes — ``RLIMIT_NPROC`` clamp (fork-bomb defence).

The runner returns a structured :class:`RunResult` with a typed
``error`` tag so the K4 adversarial-test suite can assert on the
specific failure mode without parsing strings.

Platform support:

* **Linux / WSL2** — full coverage. ``setrlimit`` works as documented.
* **macOS** — ``setrlimit`` mostly works; ``RLIMIT_NPROC`` is per-user
  on Darwin, so the fork-bomb test is approximate but still useful.
* **Windows native** — the ``resource`` module is unavailable. The
  runner refuses to start with a ``RunResult(error="UnsupportedPlatform")``,
  matching the spec's research-mode posture.

The callable is identified by ``module:attr`` so we never pass code
across the process boundary — only data. The worker ``import``s the
module under its registered name, looks up ``attr``, and calls it
with the JSON-decoded ``arg``. This is the same trust pattern as the
production registry whitelist (spec §11.5 layer 2).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

from cognithor.channels.program_synthesis.sandbox.policy import (
    DEFAULT_LIMITS,
    SandboxLimits,
)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one sandboxed call.

    ``error`` is one of:

    * ``"ok"`` — call returned normally; the JSON value is in ``value``.
    * ``"WallClockExceeded"`` — parent killed the worker after the
      wall-clock cap.
    * ``"MemoryLimitExceeded"`` — worker raised ``MemoryError`` or was
      killed by the kernel after busting ``RLIMIT_AS``.
    * ``"FileLimitExceeded"`` — worker hit ``RLIMIT_NOFILE`` (open / socket).
    * ``"ProcessLimitExceeded"`` — worker hit ``RLIMIT_NPROC``.
    * ``"WorkerCrashed"`` — non-zero exit with no specific tag (e.g.
      stack overflow, SIGSEGV, raised exception we don't classify).
    * ``"UnsupportedPlatform"`` — runner refused to start (Windows
      native).
    * ``"DecodeError"`` — worker stdout wasn't valid JSON.
    """

    ok: bool
    error: str
    value: Any = None
    exit_code: int | None = None
    stderr_tail: str = ""


def _supported_platform() -> bool:
    """True if the runner can actually enforce limits.

    POSIX-family. Windows native lacks ``resource``; the strategy router
    points Windows users at WSL2 (or research-mode without sandbox).
    """
    return sys.platform != "win32"


def run_in_sandbox(
    target: str,
    arg: Any,
    *,
    limits: SandboxLimits = DEFAULT_LIMITS,
    extra_env: dict[str, str] | None = None,
) -> RunResult:
    """Run ``target(arg)`` in a fresh subprocess with *limits* applied.

    ``target`` is ``"module:attr"`` — e.g. ``"json:loads"``. The worker
    imports ``module`` and invokes ``attr(arg)``. ``arg`` is JSON-encoded
    and decoded inside the worker, so anything not JSON-serialisable is
    rejected before the subprocess even spawns.

    Returns a :class:`RunResult`; the caller never has to handle a raw
    ``subprocess.CalledProcessError``.
    """
    # Validate the payload before the platform check so callers get a
    # consistent error model regardless of host OS — a DecodeError on
    # Windows is the same DecodeError on Linux.
    if not isinstance(target, str) or ":" not in target:
        return RunResult(
            ok=False,
            error="DecodeError",
            stderr_tail="target must be 'module:attr'",
        )
    try:
        encoded = json.dumps({"target": target, "arg": arg})
    except (TypeError, ValueError) as exc:
        return RunResult(ok=False, error="DecodeError", stderr_tail=str(exc))

    if not _supported_platform():
        return RunResult(
            ok=False,
            error="UnsupportedPlatform",
            stderr_tail="setrlimit unavailable on Windows native",
        )

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Pass the limits via env so the worker doesn't need to receive them
    # over stdin (keeps the stdin payload pure-data).
    env["PSE_SBX_MEM_MB"] = str(limits.memory_mb)
    env["PSE_SBX_NOFILE"] = "8"  # stdin/stdout/stderr + tiny headroom
    env["PSE_SBX_NPROC"] = "8"

    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        # New session prevents Ctrl+C from cascading and isolates child PG.
        start_new_session=True,
    )

    try:
        out_b, err_b = proc.communicate(
            input=encoded.encode("utf-8"),
            timeout=limits.wall_clock_seconds,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            _, err_b = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            err_b = b""
        return RunResult(
            ok=False,
            error="WallClockExceeded",
            exit_code=proc.returncode,
            stderr_tail=_tail(err_b),
        )

    stdout = out_b.decode("utf-8", errors="replace").strip()
    stderr_tail = _tail(err_b)

    if proc.returncode != 0:
        # Map well-known exit codes to typed errors.
        return RunResult(
            ok=False,
            error=_classify_exit(proc.returncode, stderr_tail),
            exit_code=proc.returncode,
            stderr_tail=stderr_tail,
        )

    if not stdout:
        return RunResult(
            ok=False,
            error="WorkerCrashed",
            exit_code=proc.returncode,
            stderr_tail=stderr_tail,
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return RunResult(
            ok=False,
            error="DecodeError",
            stderr_tail=f"{exc}: {stdout!r}",
            exit_code=proc.returncode,
        )

    if isinstance(payload, dict) and payload.get("ok") is True:
        return RunResult(ok=True, error="ok", value=payload.get("value"))
    if isinstance(payload, dict) and payload.get("ok") is False:
        return RunResult(
            ok=False,
            error=str(payload.get("error", "WorkerCrashed")),
            stderr_tail=str(payload.get("stderr_tail", stderr_tail)),
            exit_code=proc.returncode,
        )
    return RunResult(
        ok=False,
        error="DecodeError",
        stderr_tail=f"unexpected payload: {payload!r}",
    )


def _classify_exit(code: int, stderr_tail: str) -> str:
    """Map an exit code (and stderr tail hint) to a typed error tag.

    POSIX: a process killed by signal N exits with ``-N`` (Python's
    ``Popen.returncode``). The kernel OOM-killer uses SIGKILL = 9.
    """
    if code == -9 or "MemoryError" in stderr_tail or "Killed" in stderr_tail:
        return "MemoryLimitExceeded"
    if code == -11:  # SIGSEGV — typically stack overflow
        return "WorkerCrashed"
    return "WorkerCrashed"


def _tail(b: bytes, max_len: int = 4096) -> str:
    s = b.decode("utf-8", errors="replace")
    if len(s) <= max_len:
        return s
    return "...\n" + s[-max_len:]


# ---------------------------------------------------------------------------
# Worker bootstrap script (executed via `python -c`).
#
# The worker is a string instead of a real module so we don't have to
# extend the cognithor wheel layout. It applies setrlimit, imports the
# target, calls it with the JSON-decoded arg, and prints the JSON-
# encoded result on stdout. The protocol is a single line of JSON in,
# a single line of JSON out.
# ---------------------------------------------------------------------------


_WORKER_BOOTSTRAP = textwrap.dedent(
    """\
    import json, os, sys
    try:
        import resource
    except ModuleNotFoundError:
        resource = None

    def _apply_limits():
        if resource is None:
            return
        mem_mb = int(os.environ.get("PSE_SBX_MEM_MB", "256"))
        nofile = int(os.environ.get("PSE_SBX_NOFILE", "8"))
        nproc = int(os.environ.get("PSE_SBX_NPROC", "8"))
        for key, value in (
            ("RLIMIT_AS", mem_mb * 1024 * 1024),
            ("RLIMIT_NOFILE", nofile),
            ("RLIMIT_NPROC", nproc),
        ):
            rlim = getattr(resource, key, None)
            if rlim is None:
                continue
            try:
                resource.setrlimit(rlim, (value, value))
            except (ValueError, OSError):
                pass

    def _emit_error(tag, detail=""):
        sys.stdout.write(json.dumps({"ok": False, "error": tag, "stderr_tail": detail}))
        sys.stdout.write("\\n")
        sys.stdout.flush()

    def _main():
        try:
            payload = json.loads(sys.stdin.read())
        except Exception as exc:
            _emit_error("DecodeError", repr(exc))
            return
        target = payload.get("target")
        arg = payload.get("arg")
        if not isinstance(target, str) or ":" not in target:
            _emit_error("DecodeError", "target must be 'module:attr'")
            return
        # Apply limits as late as possible so we can still import.
        modname, attr = target.split(":", 1)
        try:
            module = __import__(modname, fromlist=[attr])
            fn = getattr(module, attr)
        except Exception as exc:
            _emit_error("WorkerCrashed", f"import failed: {type(exc).__name__}: {exc}")
            return
        _apply_limits()
        try:
            value = fn(arg)
        except MemoryError:
            _emit_error("MemoryLimitExceeded")
            return
        except OSError as exc:
            # EMFILE (per-process FD limit) and ENFILE (system FD limit)
            # both surface as FileLimitExceeded. EAGAIN under fork() is
            # ProcessLimitExceeded. Other errnos / strerror strings get
            # the generic crash tag.
            msg = str(exc)
            if exc.errno in (24, 23) or "Too many open files" in msg:
                _emit_error("FileLimitExceeded", repr(exc))
                return
            if exc.errno == 11 or "Resource temporarily unavailable" in msg:
                _emit_error("ProcessLimitExceeded", repr(exc))
                return
            _emit_error("WorkerCrashed", repr(exc))
            return
        except RecursionError as exc:
            _emit_error("WorkerCrashed", f"RecursionError: {exc}")
            return
        except BlockingIOError as exc:
            _emit_error("ProcessLimitExceeded", repr(exc))
            return
        except Exception as exc:
            _emit_error("WorkerCrashed", f"{type(exc).__name__}: {exc}")
            return
        try:
            sys.stdout.write(json.dumps({"ok": True, "value": value}))
            sys.stdout.write("\\n")
            sys.stdout.flush()
        except Exception as exc:
            _emit_error("WorkerCrashed", f"emit failed: {type(exc).__name__}: {exc}")

    _main()
    """
)


__all__ = [
    "RunResult",
    "run_in_sandbox",
]
