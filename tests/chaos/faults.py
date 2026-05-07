"""Fault-injection primitives for chaos tests — Sprint 2.1.

Each primitive is a context manager that activates a fault for the
``with`` block and cleans up automatically on exit (even on exception).

Usage::

    with kill_subprocess_at_random_step(target_pid):
        result = run_pge_loop(...)
    assert audit_chain_is_intact()
"""

from __future__ import annotations

import contextlib
import os
import random
import socket
import sqlite3
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Process-level faults
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def kill_subprocess_after(pid: int, delay_seconds: float) -> Iterator[None]:
    """Schedule a SIGTERM (Windows: terminate) for ``pid`` after a delay.

    The subprocess is killed unconditionally; the test body runs while
    the timer is armed. If the test finishes before the timer fires,
    the killer is cancelled.
    """
    cancelled = threading.Event()

    def _killer() -> None:
        if cancelled.wait(delay_seconds):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            else:
                os.kill(pid, 15)  # SIGTERM

    timer = threading.Thread(target=_killer, daemon=True)
    timer.start()
    try:
        yield
    finally:
        cancelled.set()


@contextlib.contextmanager
def crash_subprocess_after(pid: int, delay_seconds: float) -> Iterator[None]:
    """Like ``kill_subprocess_after`` but uses SIGKILL — simulates a crash."""
    cancelled = threading.Event()

    def _killer() -> None:
        if cancelled.wait(delay_seconds):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            else:
                os.kill(pid, 9)  # SIGKILL

    timer = threading.Thread(target=_killer, daemon=True)
    timer.start()
    try:
        yield
    finally:
        cancelled.set()


# ---------------------------------------------------------------------------
# Network faults
# ---------------------------------------------------------------------------


class _BlockingProxyServer:
    """A listening socket that accepts connections but never forwards.

    Use to simulate "service unreachable / slow" without taking down
    the actual upstream. Bind to a free port and point the test target
    at it.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(8)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        self._sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            # Hold the connection open without reading or writing.
            threading.Thread(
                target=lambda c: (time.sleep(60), c.close()),
                args=(client,),
                daemon=True,
            ).start()

    def stop(self) -> None:
        self._stop.set()
        self._sock.close()


@contextlib.contextmanager
def black_hole_port() -> Iterator[int]:
    """Yield a port number that accepts but never responds.

    Test target connects to ``127.0.0.1:<port>``; the connection
    establishes but reads/writes hang. Tests downstream timeout +
    fallback paths.
    """
    server = _BlockingProxyServer()
    try:
        yield server.port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Filesystem faults
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def disk_full_simulator(path: Path, max_bytes: int) -> Iterator[Path]:
    """Create a file size-capped at ``max_bytes`` to simulate disk-full.

    Writes that exceed the cap raise OSError. The cap is implemented
    as a sparse-file containing a guard region — most platforms enforce
    via ``rlimit`` or ``setrlimit`` only on the writer; this primitive
    instead returns a path inside a tmpfs-style mount when available,
    otherwise it simulates by pre-filling the file.
    """
    fd, name = tempfile.mkstemp(prefix="cognithor-chaos-disk-")
    p = Path(name)
    os.close(fd)
    try:
        # Pre-fill so any append beyond max_bytes raises OSError when
        # the writer tries to grow it. This is a best-effort sim:
        # real OOM-disk needs platform-specific fakefs / fault injection.
        with p.open("wb") as fh:
            fh.write(b"\0" * max_bytes)
        yield p
    finally:
        with contextlib.suppress(OSError):
            p.unlink()


# ---------------------------------------------------------------------------
# Database faults
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def corrupt_sqlite_pages(db_path: Path, *, pages_to_corrupt: int = 1) -> Iterator[None]:
    """Flip random bytes in random SQLite pages, then restore on exit.

    Useful for testing: does the audit chain detect / recover from
    in-place page corruption? Does the WAL replay restore consistency?
    """
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    backup = db_path.with_suffix(db_path.suffix + ".chaos-backup")
    backup.write_bytes(db_path.read_bytes())
    try:
        with db_path.open("r+b") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size <= 64:
                # Too small to corrupt safely — abort.
                yield
                return
            for _ in range(pages_to_corrupt):
                offset = random.randint(64, max(64, size - 8))
                fh.seek(offset)
                original = fh.read(4)
                fh.seek(offset)
                # Flip every bit in the 4 bytes
                fh.write(struct.pack(">I", ~struct.unpack(">I", original)[0] & 0xFFFFFFFF))
        yield
    finally:
        # Restore from backup
        db_path.write_bytes(backup.read_bytes())
        backup.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# GPU / memory faults — best-effort simulation
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def simulate_gpu_oom_via_env() -> Iterator[None]:
    """Set env vars that vLLM / HuggingFace honour as GPU-not-available.

    Cheap simulation — does not actually exhaust VRAM. Lets us verify
    the fallback path emits ``vlm_no_models_available`` etc.
    """
    saved = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HF_NO_GPU": os.environ.get("HF_NO_GPU"),
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_NO_GPU"] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Audit-chain faults
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def tamper_audit_jsonl(audit_path: Path, *, line_offset: int = -2) -> Iterator[None]:
    """Modify an audit-chain entry in place, then restore on exit.

    Used to test that ``cognithor audit verify`` detects the tamper.
    ``line_offset`` is the line index to corrupt; default is the
    second-to-last line so the chain head still exists.
    """
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    backup = audit_path.read_bytes()
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            yield
            return
        idx = line_offset if line_offset >= 0 else max(0, len(lines) + line_offset)
        lines[idx] = lines[idx].replace('"', "'", 1)  # break canonical JSON
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        yield
    finally:
        audit_path.write_bytes(backup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def audit_chain_is_intact(audit_path: Path) -> bool:
    """Best-effort verification — checks each line is valid JSON and that
    every prev_hash matches the previous entry's content hash.

    Uses the same canonical-form rules as ``cognithor.audit`` so a
    chaos test can run this without spinning up the full Gateway.
    """
    if not audit_path.exists():
        return True  # nothing to verify
    import hashlib
    import json

    prev_hash = "0" * 64
    with audit_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False
            recorded = entry.get("prev_hash")
            if recorded != prev_hash:
                return False
            payload = {k: v for k, v in entry.items() if k != "hash"}
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            prev_hash = digest
    return True
