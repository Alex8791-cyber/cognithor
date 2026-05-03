"""Tests für Distributed Locking.

Testet: LocalLockBackend, FileLockBackend, RedisLockBackend, create_lock factory.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.core.distributed_lock import (
    DistributedLock,
    FileLockBackend,
    LocalLockBackend,
    LockBackend,
    RedisLockBackend,
    create_lock,
)

if TYPE_CHECKING:
    from pathlib import Path

    pass


# ============================================================================
# LockBackend Enum
# ============================================================================


class TestLockBackend:
    def test_values(self) -> None:
        assert LockBackend.LOCAL == "local"
        assert LockBackend.FILE == "file"
        assert LockBackend.REDIS == "redis"

    def test_str_enum(self) -> None:
        assert str(LockBackend.LOCAL) == "local"
        assert f"{LockBackend.FILE}" == "file"


# ============================================================================
# DistributedLock (base class)
# ============================================================================


class TestDistributedLockBase:
    def test_acquire_not_implemented(self) -> None:
        # ``asyncio.get_event_loop()`` from a main thread without a
        # running loop raises ``RuntimeError`` on Python 3.12+.
        # ``asyncio.run`` is the modern equivalent.
        lock = DistributedLock()
        with pytest.raises(NotImplementedError):
            asyncio.run(lock.acquire("test"))

    def test_release_not_implemented(self) -> None:
        lock = DistributedLock()
        with pytest.raises(NotImplementedError):
            asyncio.run(lock.release("test"))

    async def test_direct_aenter_without_name_raises(self) -> None:
        lock = DistributedLock()
        with pytest.raises(RuntimeError, match="Use lock"):
            async with lock:
                pass


# ============================================================================
# LocalLockBackend
# ============================================================================


class TestLocalLockBackend:
    async def test_acquire_and_release(self) -> None:
        lock = LocalLockBackend()
        acquired = await lock.acquire("test_lock")
        assert acquired is True
        await lock.release("test_lock")

    async def test_acquire_twice_blocks(self) -> None:
        lock = LocalLockBackend()
        await lock.acquire("blocker")
        # Second acquire should timeout
        acquired = await lock.acquire("blocker", timeout=0.1)
        assert acquired is False
        await lock.release("blocker")

    async def test_release_unlocks(self) -> None:
        lock = LocalLockBackend()
        await lock.acquire("cycle")
        await lock.release("cycle")
        # Should be re-acquirable after release
        acquired = await lock.acquire("cycle", timeout=0.1)
        assert acquired is True
        await lock.release("cycle")

    async def test_release_nonexistent_is_noop(self) -> None:
        lock = LocalLockBackend()
        # Should not raise
        await lock.release("nonexistent")

    async def test_context_manager(self) -> None:
        lock = LocalLockBackend()
        async with lock("resource_a"):
            # Lock should be held
            acquired = await lock.acquire("resource_a", timeout=0.05)
            assert acquired is False
        # Lock should be released
        acquired = await lock.acquire("resource_a", timeout=0.1)
        assert acquired is True
        await lock.release("resource_a")

    async def test_context_manager_timeout_raises(self) -> None:
        lock = LocalLockBackend()
        await lock.acquire("contested")
        with pytest.raises(TimeoutError, match="contested"):
            async with lock("contested", timeout=0.05):
                pass  # pragma: no cover
        await lock.release("contested")

    async def test_independent_locks(self) -> None:
        lock = LocalLockBackend()
        await lock.acquire("alpha")
        # Different name should be independent
        acquired = await lock.acquire("beta", timeout=0.1)
        assert acquired is True
        await lock.release("alpha")
        await lock.release("beta")

    async def test_concurrent_acquires(self) -> None:
        lock = LocalLockBackend()
        results: list[str] = []

        async def worker(name: str) -> None:
            async with lock("shared"):
                results.append(f"{name}_start")
                await asyncio.sleep(0.01)
                results.append(f"{name}_end")

        await asyncio.gather(worker("A"), worker("B"))
        # Workers must not interleave (lock ensures serialization)
        assert results.index("A_start") < results.index("A_end")
        assert results.index("B_start") < results.index("B_end")
        # One must complete before the other starts
        a_end = results.index("A_end")
        b_start = results.index("B_start")
        a_start = results.index("A_start")
        b_end = results.index("B_end")
        assert (a_end < b_start) or (b_end < a_start)


# ============================================================================
# FileLockBackend
# ============================================================================


class TestFileLockBackend:
    async def test_acquire_creates_lockfile(self, tmp_path: Path) -> None:
        lock = FileLockBackend(lock_dir=tmp_path)
        acquired = await lock.acquire("session_42")
        assert acquired is True
        lockfile = tmp_path / "session_42.lock"
        assert lockfile.exists()
        await lock.release("session_42")

    async def test_release_cleans_up(self, tmp_path: Path) -> None:
        lock = FileLockBackend(lock_dir=tmp_path)
        await lock.acquire("cleanup_test")
        await lock.release("cleanup_test")
        # File should be removed after release
        lockfile = tmp_path / "cleanup_test.lock"
        assert not lockfile.exists()

    async def test_acquire_and_release_cycle(self, tmp_path: Path) -> None:
        lock = FileLockBackend(lock_dir=tmp_path)
        for _ in range(3):
            acquired = await lock.acquire("cycle", timeout=1.0)
            assert acquired is True
            await lock.release("cycle")

    async def test_release_nonexistent_is_noop(self, tmp_path: Path) -> None:
        lock = FileLockBackend(lock_dir=tmp_path)
        await lock.release("never_acquired")  # Should not raise

    async def test_context_manager(self, tmp_path: Path) -> None:
        lock = FileLockBackend(lock_dir=tmp_path)
        async with lock("ctx_file"):
            lockfile = tmp_path / "ctx_file.lock"
            assert lockfile.exists()
        # After exit, lockfile should be cleaned up
        assert not lockfile.exists()

    async def test_lock_dir_created_automatically(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "locks"
        lock = FileLockBackend(lock_dir=nested)
        assert nested.is_dir()
        acquired = await lock.acquire("auto_dir")
        assert acquired is True
        await lock.release("auto_dir")

    async def test_slash_in_name_sanitized(self, tmp_path: Path) -> None:
        lock = FileLockBackend(lock_dir=tmp_path)
        acquired = await lock.acquire("user/session/123")
        assert acquired is True
        lockfile = tmp_path / "user_session_123.lock"
        assert lockfile.exists()
        await lock.release("user/session/123")

    async def test_concurrent_file_locks(self, tmp_path: Path) -> None:
        """Two FileLockBackend instances contending for the same lock."""
        lock_a = FileLockBackend(lock_dir=tmp_path)
        lock_b = FileLockBackend(lock_dir=tmp_path)

        acquired_a = await lock_a.acquire("contended", timeout=1.0)
        assert acquired_a is True

        # Second instance should fail to acquire with short timeout
        acquired_b = await lock_b.acquire("contended", timeout=0.2)
        assert acquired_b is False

        await lock_a.release("contended")

        # Now second instance should succeed
        acquired_b = await lock_b.acquire("contended", timeout=1.0)
        assert acquired_b is True
        await lock_b.release("contended")


# ============================================================================
# RedisLockBackend
# ============================================================================


class TestRedisLockBackend:
    async def test_fallback_when_redis_not_installed(self, tmp_path: Path) -> None:
        """Without redis package, falls back to FileLockBackend."""
        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            lock = RedisLockBackend.__new__(RedisLockBackend)
            lock._redis_url = "redis://localhost:6379/0"
            lock._key_prefix = "jarvis:lock:"
            lock._default_ttl = 30.0
            lock._tokens = {}
            lock._client = None
            lock._fallback = None
            lock._lock_dir = tmp_path
            lock._current_name = None
            lock._redis_available = False

            acquired = await lock.acquire("fallback_test", timeout=1.0)
            assert acquired is True
            await lock.release("fallback_test")

    async def test_acquire_with_mock_redis(self) -> None:
        """Mock Redis client for acquire."""
        lock = RedisLockBackend.__new__(RedisLockBackend)
        lock._redis_url = "redis://localhost:6379/0"
        lock._key_prefix = "jarvis:lock:"
        lock._default_ttl = 30.0
        lock._tokens = {}
        lock._client = None
        lock._fallback = None
        lock._lock_dir = None
        lock._current_name = None
        lock._redis_available = True

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        lock._client = mock_client

        acquired = await lock.acquire("redis_test", timeout=1.0)
        assert acquired is True
        assert "redis_test" in lock._tokens
        mock_client.set.assert_called_once()

    async def test_release_with_mock_redis(self) -> None:
        """Mock Redis client for release (Lua script)."""
        lock = RedisLockBackend.__new__(RedisLockBackend)
        lock._redis_url = "redis://localhost:6379/0"
        lock._key_prefix = "jarvis:lock:"
        lock._default_ttl = 30.0
        lock._tokens = {"redis_rel": "tok123"}
        lock._client = None
        lock._fallback = None
        lock._lock_dir = None
        lock._current_name = None
        lock._redis_available = True

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.eval = AsyncMock(return_value=1)
        lock._client = mock_client

        await lock.release("redis_rel")
        assert "redis_rel" not in lock._tokens
        mock_client.eval.assert_called_once()

    async def test_acquire_fails_on_nx_false(self) -> None:
        """If SET NX returns None (lock held), acquire should timeout."""
        lock = RedisLockBackend.__new__(RedisLockBackend)
        lock._redis_url = "redis://localhost:6379/0"
        lock._key_prefix = "jarvis:lock:"
        lock._default_ttl = 30.0
        lock._tokens = {}
        lock._client = None
        lock._fallback = None
        lock._lock_dir = None
        lock._current_name = None
        lock._redis_available = True

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.set = AsyncMock(return_value=None)  # NX failed
        lock._client = mock_client

        acquired = await lock.acquire("contested", timeout=0.15)
        assert acquired is False

    async def test_redis_error_falls_back_to_file(self, tmp_path: Path) -> None:
        """If Redis raises during SET, fall back to file locks."""
        lock = RedisLockBackend.__new__(RedisLockBackend)
        lock._redis_url = "redis://localhost:6379/0"
        lock._key_prefix = "jarvis:lock:"
        lock._default_ttl = 30.0
        lock._tokens = {}
        lock._client = None
        lock._fallback = None
        lock._lock_dir = tmp_path
        lock._current_name = None
        lock._redis_available = True

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.set = AsyncMock(side_effect=ConnectionError("redis down"))
        lock._client = mock_client

        acquired = await lock.acquire("failover", timeout=1.0)
        assert acquired is True
        # Should have created a file fallback
        assert lock._fallback is not None
        await lock.release("failover")

    async def test_context_manager_with_mock(self) -> None:
        """Context manager with mocked Redis."""
        lock = RedisLockBackend.__new__(RedisLockBackend)
        lock._redis_url = "redis://localhost:6379/0"
        lock._key_prefix = "jarvis:lock:"
        lock._default_ttl = 30.0
        lock._tokens = {}
        lock._client = None
        lock._fallback = None
        lock._lock_dir = None
        lock._current_name = None
        lock._redis_available = True

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        mock_client.eval = AsyncMock(return_value=1)
        lock._client = mock_client

        async with lock("ctx_redis"):
            assert "ctx_redis" in lock._tokens
        assert "ctx_redis" not in lock._tokens

    async def test_expiry_passed_to_redis(self) -> None:
        """TTL is passed to the Redis SET command."""
        lock = RedisLockBackend.__new__(RedisLockBackend)
        lock._redis_url = "redis://localhost:6379/0"
        lock._key_prefix = "jarvis:lock:"
        lock._default_ttl = 60.0
        lock._tokens = {}
        lock._client = None
        lock._fallback = None
        lock._lock_dir = None
        lock._current_name = None
        lock._redis_available = True

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        lock._client = mock_client

        await lock.acquire("ttl_test")
        call_kwargs = mock_client.set.call_args
        # ex parameter should be the TTL
        assert call_kwargs.kwargs.get("ex") == 60 or call_kwargs[1].get("ex") == 60


# ============================================================================
# create_lock Factory
# ============================================================================


class TestCreateLock:
    def test_default_creates_local(self) -> None:
        lock = create_lock()
        assert isinstance(lock, LocalLockBackend)

    def test_none_config_creates_local(self) -> None:
        lock = create_lock(None)
        assert isinstance(lock, LocalLockBackend)

    def test_local_backend(self) -> None:
        cfg = MagicMock()
        cfg.lock_backend = "local"
        cfg.redis_url = "redis://localhost:6379/0"
        cfg.cognithor_home = None
        lock = create_lock(cfg)
        assert isinstance(lock, LocalLockBackend)

    def test_file_backend(self, tmp_path: Path) -> None:
        cfg = MagicMock()
        cfg.lock_backend = "file"
        cfg.redis_url = "redis://localhost:6379/0"
        cfg.cognithor_home = str(tmp_path)
        lock = create_lock(cfg)
        assert isinstance(lock, FileLockBackend)

    def test_redis_backend(self) -> None:
        cfg = MagicMock()
        cfg.lock_backend = "redis"
        cfg.redis_url = "redis://localhost:6379/1"
        cfg.cognithor_home = None
        lock = create_lock(cfg)
        assert isinstance(lock, RedisLockBackend)

    def test_file_backend_uses_jarvis_home(self, tmp_path: Path) -> None:
        cfg = MagicMock()
        cfg.lock_backend = "file"
        cfg.redis_url = "redis://localhost:6379/0"
        cfg.cognithor_home = str(tmp_path / "custom_home")
        lock = create_lock(cfg)
        assert isinstance(lock, FileLockBackend)
        expected_dir = tmp_path / "custom_home" / "locks"
        assert lock._lock_dir == expected_dir


# ============================================================================
# Config Integration
# ============================================================================


class TestConfigIntegration:
    def test_config_has_lock_backend(self) -> None:
        from cognithor.config import CognithorConfig

        cfg = CognithorConfig()
        assert cfg.lock_backend == "local"

    def test_config_has_redis_url(self) -> None:
        from cognithor.config import CognithorConfig

        cfg = CognithorConfig()
        assert cfg.redis_url == "redis://localhost:6379/0"

    def test_config_lock_backend_values(self) -> None:
        from cognithor.config import CognithorConfig

        for value in ("local", "file", "redis"):
            cfg = CognithorConfig(lock_backend=value)
            assert cfg.lock_backend == value

    def test_create_lock_from_real_config(self) -> None:
        from cognithor.config import CognithorConfig

        cfg = CognithorConfig()
        lock = create_lock(cfg)
        assert isinstance(lock, LocalLockBackend)

    def test_create_lock_file_from_config(self, tmp_path: Path) -> None:
        from cognithor.config import CognithorConfig

        cfg = CognithorConfig(lock_backend="file", cognithor_home=tmp_path / ".cognithor")
        lock = create_lock(cfg)
        assert isinstance(lock, FileLockBackend)
