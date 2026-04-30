# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Canonical K4 adversarial payloads (spec §11.5).

The payloads are exercised by the subprocess sandbox runner from the
K4 adversarial test suite. They are deliberately split into their own
module so the worker can ``import`` them by ``module:attr`` — the
runner never marshals code, only the *name* of the function to call.

Each payload mirrors a spec §11.5 case:

* :func:`while_true_loop` — busy-loop; killed by wall-clock.
* :func:`numpy_giant_alloc` — busts ``RLIMIT_AS``.
* :func:`open_etc_passwd` — busts ``RLIMIT_NOFILE``.
* :func:`socket_connect` — same (``socket()`` consumes an FD).
* :func:`fork_bomb` — busts ``RLIMIT_NPROC``.
* :func:`stack_overflow` — uncontrolled recursion; only the worker dies.

These are *not* part of the production registry — they live under
``sandbox._adversarial_payloads`` and are imported by name only when
the K4 test suite asks the runner to invoke them.
"""

from __future__ import annotations

import contextlib
import os
import socket
from typing import Any


def while_true_loop(_arg: Any) -> Any:
    """Busy-loop forever. Wall-clock timeout must kick the worker."""
    while True:
        pass


def numpy_giant_alloc(_arg: Any) -> Any:
    """Allocate ~8 GB of int64. RLIMIT_AS at 256 MB must refuse."""
    import numpy as np

    return np.zeros((10**9,), dtype=np.int64).tolist()


def open_etc_passwd(_arg: Any) -> Any:
    """Try to open many files. RLIMIT_NOFILE=8 must produce OSError."""
    handles = []
    # /etc/passwd is the spec's canonical example, but the test only
    # cares that the FD-limit fires — open ``/dev/null`` as a portable
    # equivalent that doesn't need a specific path to exist.
    target = "/dev/null"
    try:
        for _ in range(64):
            handles.append(open(target, "rb"))
        return len(handles)
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.close()


def socket_connect(_arg: Any) -> Any:
    """Allocate a socket. RLIMIT_NOFILE=8 + the existing stdin/stdout/
    stderr FDs leave no room for a new socket; ``socket()`` raises."""
    sockets = []
    try:
        for _ in range(32):
            sockets.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        return len(sockets)
    finally:
        for s in sockets:
            with contextlib.suppress(Exception):
                s.close()


def fork_bomb(_arg: Any) -> Any:
    """fork() until RLIMIT_NPROC bites.

    Bounded loop instead of the classical "while True: fork()" so the
    test stays well-behaved in the unlikely event the limit isn't
    enforced — an unbounded fork bomb could destabilise a CI worker.

    Windows-native callers won't reach this code (the runner refuses
    to start with ``UnsupportedPlatform`` first). The ``getattr``
    guard keeps mypy --strict happy without a per-line ignore: on
    Windows, ``os.fork`` is absent.
    """
    fork = getattr(os, "fork", None)
    if fork is None:
        return 0
    spawned = 0
    for _ in range(64):
        try:
            pid = fork()
        except (BlockingIOError, OSError):
            return spawned
        if pid == 0:
            # Child: exit immediately so the parent's fork() count is
            # the only thing being measured.
            os._exit(0)
        spawned += 1
    return spawned


def stack_overflow(_arg: Any) -> Any:
    """Uncontrolled recursion — must crash *only* the subprocess."""

    def _r(n: int) -> int:
        return _r(n + 1)

    return _r(0)


__all__ = [
    "fork_bomb",
    "numpy_giant_alloc",
    "open_etc_passwd",
    "socket_connect",
    "stack_overflow",
    "while_true_loop",
]
