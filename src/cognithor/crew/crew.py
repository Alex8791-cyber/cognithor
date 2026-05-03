"""Crew — top-level orchestration object."""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cognithor.crew.agent import CrewAgent, LLMConfig
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask

if TYPE_CHECKING:
    from cognithor.crew.output import CrewOutput

log = logging.getLogger(__name__)


# Module-level bounded cache keyed by kickoff_id (best-effort, per-process).
# OrderedDict + LRU-style eviction caps memory growth. See R3-NI7.
_KICKOFF_CACHE_MAX_SIZE = 128
_KICKOFF_CACHE: OrderedDict[str, CrewOutput] = OrderedDict()


def _cache_put(key: str, value: CrewOutput) -> None:
    """Insert or refresh a cache entry, evicting oldest when over capacity."""
    _KICKOFF_CACHE[key] = value
    _KICKOFF_CACHE.move_to_end(key)
    while len(_KICKOFF_CACHE) > _KICKOFF_CACHE_MAX_SIZE:
        _KICKOFF_CACHE.popitem(last=False)


def _cache_get(key: str) -> CrewOutput | None:
    """Return cached value (refreshing LRU position) or None."""
    if key in _KICKOFF_CACHE:
        _KICKOFF_CACHE.move_to_end(key)
        return _KICKOFF_CACHE[key]
    return None


# Process-wide distributed-lock singleton. create_lock() with a
# LocalLockBackend builds a fresh dict[str, asyncio.Lock] per call —
# so a new DistributedLock per kickoff_async() call would NEVER serialize
# two concurrent same-id kickoffs inside one process. See R3-NC2.
_lock_singleton: Any = None
_lock_singleton_init = threading.Lock()


def _get_distributed_lock() -> Any:
    """Return the process-wide DistributedLock, constructing it lazily once.

    Double-checked locking: candidate built OUTSIDE the threading.Lock so
    two racing threads don't block on config loading, and only one
    candidate wins the singleton slot.
    """
    global _lock_singleton
    if _lock_singleton is not None:
        return _lock_singleton

    from cognithor.config import load_config
    from cognithor.core.distributed_lock import create_lock

    candidate = create_lock(load_config())  # built outside critical section
    with _lock_singleton_init:
        if _lock_singleton is None:
            _lock_singleton = candidate
        # Return INSIDE the critical section so an external reset of
        # _lock_singleton (e.g. test monkeypatch) between the lock exit and
        # the return can't leave us handing back None.
        return _lock_singleton


# In-process fallback, lazily constructed (asyncio.Lock() needs a running loop).
_fallback_lock: asyncio.Lock | None = None


async def _get_fallback_lock() -> asyncio.Lock:
    """Lazily construct a module-level asyncio.Lock bound to the running loop."""
    global _fallback_lock
    if _fallback_lock is None:
        _fallback_lock = asyncio.Lock()
    return _fallback_lock


class Crew(BaseModel):
    """A Crew is a declarative bundle of agents + tasks + process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agents: list[CrewAgent] = Field(..., min_length=1)
    tasks: list[CrewTask] = Field(..., min_length=1)
    process: CrewProcess = CrewProcess.SEQUENTIAL
    verbose: bool = False
    planning: bool = False
    # Spec §1.2 — matches CrewAgent.llm: str | LLMConfig | None.
    manager_llm: str | LLMConfig | None = None

    def __init__(self, *, planner: Any = None, **kwargs: Any) -> None:
        """Custom init so callers can inject a live Planner.

        ``planner`` is NOT a Pydantic field (it wraps non-serializable state
        — Ollama client, model router, cost tracker, etc.), so we stash it
        on the instance via ``object.__setattr__`` after ``super().__init__``
        has finished validating the declared fields. Gateway + tests pass a
        live Planner; standalone scripts omit the kwarg and ``kickoff_async``
        falls back to :func:`get_default_planner`.
        """
        super().__init__(**kwargs)
        object.__setattr__(self, "_planner", planner)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Crew:
        """Preserve the injected ``_planner`` across ``model_copy()``.

        Pydantic v2's default ``model_copy`` walks only declared fields, so
        an injected Planner would silently disappear on any copy, forcing
        the copy to fall back to :func:`get_default_planner` at kickoff
        time — a subtle production footgun. We re-attach the original
        planner reference (shared — a copy should route through the same
        live client by default). Callers who want a different planner can
        ``object.__setattr__`` after the copy.
        """
        copied = super().model_copy(update=update, deep=deep)
        object.__setattr__(copied, "_planner", getattr(self, "_planner", None))
        return copied

    @model_validator(mode="after")
    def _warn_on_hierarchical_without_manager(self) -> Crew:
        if self.process is CrewProcess.HIERARCHICAL and self.manager_llm is None:
            warnings.warn(
                "CrewProcess.HIERARCHICAL without manager_llm falls back to the "
                "first agent's llm for routing decisions. For production, set "
                "manager_llm explicitly.",
                stacklevel=3,
            )
        return self

    def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Synchronous kickoff — trampoline through ``asyncio.run`` so both
        paths share the same compiler + planner wiring.

        Refuses to run from inside a running event loop: ``asyncio.run``
        cannot be called when one is already active. Callers inside async
        contexts must use :meth:`kickoff_async` directly.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.kickoff_async(inputs))
        raise RuntimeError(
            "Crew.kickoff() called from within a running event loop. "
            "Use `await crew.kickoff_async(inputs)` instead."
        )

    async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Async kickoff with parallel fan-out for tasks marked async_execution=True.

        ``_kickoff_id`` in ``inputs`` enables idempotent replay: calling twice
        with the same id returns the first result without re-running. Concurrent
        same-id kickoffs serialize via the distributed-lock singleton; if the
        distributed_lock module isn't available we fall back to an in-process
        ``asyncio.Lock`` rather than silently bypassing serialization.
        """
        # Non-destructive strip: comprehension copy, caller's dict stays intact.
        kickoff_id: str | None = None
        if inputs:
            kickoff_id = inputs.get("_kickoff_id")
            inputs = {k: v for k, v in inputs.items() if k != "_kickoff_id"}

        if kickoff_id:
            cached = _cache_get(kickoff_id)
            if cached is not None:
                return cached

        from cognithor.crew.compiler import compile_and_run_async
        from cognithor.crew.runtime import get_default_planner, get_default_tool_registry

        planner = getattr(self, "_planner", None) or get_default_planner()
        registry = get_default_tool_registry()
        manager_llm = self.manager_llm if isinstance(self.manager_llm, str) else None

        async def _run_guarded() -> CrewOutput:
            """Run the compiler under whichever lock is active and populate the cache."""
            # Inside-lock double-check: if another coroutine finished while we
            # were waiting for the lock, return its cached result.
            if kickoff_id:
                cached_inner = _cache_get(kickoff_id)
                if cached_inner is not None:
                    return cached_inner
            result = await compile_and_run_async(
                agents=self.agents,
                tasks=self.tasks,
                process=self.process,
                inputs=inputs,
                registry=registry,
                planner=planner,
                manager_llm=manager_llm,
            )
            if kickoff_id:
                _cache_put(kickoff_id, result)
            return result

        if kickoff_id:
            try:
                lock = _get_distributed_lock()
            except ImportError:
                log.warning(
                    "cognithor.core.distributed_lock unavailable — falling back "
                    "to in-process asyncio.Lock for crew kickoff serialization. "
                    "Cross-process idempotency is NOT guaranteed in this config."
                )
                async with await _get_fallback_lock():
                    return await _run_guarded()
            async with lock(f"crew:kickoff:{kickoff_id}", 300.0):
                return await _run_guarded()

        # No kickoff_id — plain unlocked execution.
        return await compile_and_run_async(
            agents=self.agents,
            tasks=self.tasks,
            process=self.process,
            inputs=inputs,
            registry=registry,
            planner=planner,
            manager_llm=manager_llm,
        )
