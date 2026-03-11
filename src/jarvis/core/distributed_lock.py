"""
Jarvis · Distributed Locking.

Provides a lock abstraction with multiple backends for multi-instance support:
  - LOCAL: asyncio.Lock (single-process, default)
  - FILE:  File-based locks (cross-process on same machine)
  - REDIS: Redis-based locks (cross-machine, optional)

Architecture Bible: §4.10
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

log = logging.getLogger(__name__)


# ============================================================================
# Backend Enum
# ============================================================================


class LockBackend(StrEnum):
    """Supported lock backends."""

    LOCAL = "local"
    FILE = "file"
    REDIS = "redis"


# ============================================================================
# Abstract Base
# ============================================================================


class DistributedLock:
    """Distributed lock with backend abstraction.

    Usage::

        lock = create_lock(config)
        async with lock("session_123"):
            # critical section
            ...
    """

    def __init__(self) -> None:
        self._current_name: str | None = None

    # -- Public API ----------------------------------------------------------

    async def acquire(self, name: str, timeout: float = 10.0) -> bool:
        """Acquire a named lock.

        Args:
            name: Logical lock name (e.g. ``"session_abc"``).
            timeout: Maximum seconds to wait. ``0`` means try once.

        Returns:
            ``True`` if the lock was acquired, ``False`` on timeout.
        """
        raise NotImplementedError

    async def release(self, name: str) -> None:
        """Release a previously acquired lock."""
        raise NotImplementedError

    # -- Context-Manager shortcut -------------------------------------------

    def __call__(self, name: str, timeout: float = 10.0) -> _LockContext:
        """Return an async context manager for *name*."""
        return _LockContext(self, name, timeout)

    async def __aenter__(self) -> DistributedLock:
        if self._current_name is None:
            raise RuntimeError("Use lock('name') as context manager, not lock directly")
        acquired = await self.acquire(self._current_name)
        if not acquired:
            raise TimeoutError(f"Could not acquire lock {self._current_name!r}")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._current_name is not None:
            await self.release(self._current_name)
            self._current_name = None


class _LockContext:
    """Async context manager returned by ``lock('name')``."""

    __slots__ = ("_lock", "_name", "_timeout")

    def __init__(self, lock: DistributedLock, name: str, timeout: float) -> None:
        self._lock = lock
        self._name = name
        self._timeout = timeout

    async def __aenter__(self) -> DistributedLock:
        acquired = await self._lock.acquire(self._name, self._timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire lock {self._name!r}")
        return self._lock

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._lock.release(self._name)


# ============================================================================
# LOCAL Backend — asyncio.Lock (single process)
# ============================================================================


class LocalLockBackend(DistributedLock):
    """asyncio.Lock-based backend (single instance only)."""

    def __init__(self) -> None:
        super().__init__()
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    async def acquire(self, name: str, timeout: float = 10.0) -> bool:
        lock = self._get_lock(name)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout if timeout > 0 else None)
            return True
        except asyncio.TimeoutError:
            return False

    async def release(self, name: str) -> None:
        lock = self._locks.get(name)
        if lock is not None and lock.locked():
            lock.release()


# ============================================================================
# FILE Backend — cross-process file locks
# ============================================================================


class FileLockBackend(DistributedLock):
    """File-based locking using lockfiles.

    Creates files like ``<lock_dir>/<name>.lock``.
    On Windows uses ``msvcrt.locking()``, on Unix uses ``fcntl.flock()``.
    """

    def __init__(self, lock_dir: Path | None = None) -> None:
        super().__init__()
        self._lock_dir = lock_dir or (Path.home() / ".jarvis" / "locks")
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, object] = {}  # name -> open file handle

    def _lock_path(self, name: str) -> Path:
        safe_name = name.replace("/", "_").replace("\\", "_")
        return self._lock_dir / f"{safe_name}.lock"

    async def acquire(self, name: str, timeout: float = 10.0) -> bool:
        path = self._lock_path(name)
        deadline = time.monotonic() + timeout
        poll_interval = 0.05  # 50 ms

        while True:
            try:
                acquired = await asyncio.get_running_loop().run_in_executor(
                    None, self._try_lock, path, name
                )
                if acquired:
                    return True
            except OSError:
                log.debug("File lock contention on %s", name)

            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
            poll_interval = min(poll_interval * 1.5, 0.5)  # backoff

    def _try_lock(self, path: Path, name: str) -> bool:
        """Attempt to acquire the file lock (runs in thread executor)."""
        # Open (or create) the lock file
        fh = open(path, "w")  # noqa: SIM115
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write PID for diagnostics
            fh.write(str(os.getpid()))
            fh.flush()
            self._handles[name] = fh
            return True
        except (OSError, IOError):
            fh.close()
            return False

    async def release(self, name: str) -> None:
        fh = self._handles.pop(name, None)
        if fh is None:
            return
        await asyncio.get_running_loop().run_in_executor(None, self._unlock, fh, name)

    def _unlock(self, fh: object, name: str) -> None:
        """Release the file lock (runs in thread executor)."""
        import io

        if not isinstance(fh, io.TextIOWrapper):
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # Best-effort unlock, file handle will be closed below
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                fh.close()
            except OSError:
                pass  # Best-effort close, file may already be closed
            # Best-effort cleanup of lockfile
            try:
                self._lock_path(name).unlink(missing_ok=True)
            except OSError:
                pass  # Best-effort lockfile removal, another process may hold it


# ============================================================================
# REDIS Backend — distributed lock via Redis
# ============================================================================


class RedisLockBackend(DistributedLock):
    """Redis-based distributed lock using SET NX EX pattern.

    Requires ``redis`` (optional dependency).
    Falls back to :class:`FileLockBackend` if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "jarvis:lock:",
        default_ttl: float = 30.0,
        lock_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._tokens: dict[str, str] = {}  # name -> unique token
        self._client: object | None = None
        self._fallback: FileLockBackend | None = None
        self._lock_dir = lock_dir

        # Try to import redis
        try:
            import redis.asyncio as aioredis  # noqa: F401

            self._redis_available = True
        except ImportError:
            log.warning(
                "redis package not installed — falling back to FileLockBackend. "
                "Install with: pip install redis"
            )
            self._redis_available = False

    async def _get_client(self) -> object | None:
        """Lazy-init the Redis async client."""
        if not self._redis_available:
            return None
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                    socket_connect_timeout=3,
                )
                # Ping to verify connection
                await self._client.ping()  # type: ignore[union-attr]
            except Exception:
                log.warning("Redis not reachable at %s — using file lock fallback", self._redis_url)
                self._client = None
                self._redis_available = False
        return self._client

    def _get_fallback(self) -> FileLockBackend:
        if self._fallback is None:
            self._fallback = FileLockBackend(lock_dir=self._lock_dir)
        return self._fallback

    async def acquire(self, name: str, timeout: float = 10.0) -> bool:
        client = await self._get_client()
        if client is None:
            return await self._get_fallback().acquire(name, timeout)

        import uuid

        token = uuid.uuid4().hex
        key = f"{self._key_prefix}{name}"
        deadline = time.monotonic() + timeout
        poll_interval = 0.05

        while True:
            try:
                result = await client.set(  # type: ignore[union-attr]
                    key, token, nx=True, ex=int(max(self._default_ttl, 1))
                )
                if result:
                    self._tokens[name] = token
                    return True
            except Exception:
                log.debug("Redis SET failed for %s, falling back", name)
                return await self._get_fallback().acquire(name, timeout)

            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
            poll_interval = min(poll_interval * 1.5, 0.5)

    async def release(self, name: str) -> None:
        token = self._tokens.pop(name, None)
        client = await self._get_client() if token else None

        if client is not None and token is not None:
            key = f"{self._key_prefix}{name}"
            # Lua script: only delete if token matches (safe release)
            lua = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """
            try:
                await client.eval(lua, 1, key, token)  # type: ignore[union-attr]
                return
            except Exception:
                log.debug("Redis release failed for %s", name)

        # Fallback release
        if self._fallback is not None:
            await self._fallback.release(name)


# ============================================================================
# Factory
# ============================================================================


def create_lock(config: object | None = None) -> DistributedLock:
    """Create the appropriate lock backend based on configuration.

    Args:
        config: A :class:`~jarvis.config.JarvisConfig` instance (or ``None``
                for defaults).

    Returns:
        A ready-to-use :class:`DistributedLock` instance.
    """
    backend = "local"
    redis_url = "redis://localhost:6379/0"
    lock_dir: Path | None = None

    if config is not None:
        backend = getattr(config, "lock_backend", "local")
        redis_url = getattr(config, "redis_url", redis_url)
        jarvis_home = getattr(config, "jarvis_home", None)
        if jarvis_home is not None:
            lock_dir = Path(jarvis_home) / "locks"

    if backend == LockBackend.REDIS:
        log.info("Using Redis lock backend (%s)", redis_url)
        return RedisLockBackend(redis_url=redis_url, lock_dir=lock_dir)
    elif backend == LockBackend.FILE:
        log.info("Using file lock backend (%s)", lock_dir or "~/.jarvis/locks")
        return FileLockBackend(lock_dir=lock_dir)
    else:
        log.info("Using local (asyncio) lock backend")
        return LocalLockBackend()
