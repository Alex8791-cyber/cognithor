"""Method decorators for building a Crew from a Python class.

Concept inspired by CrewAI's @agent/@task/@crew pattern — implementation
is Apache 2.0, no verbatim borrow.

Usage::

    class MyCrew:
        @agent
        def researcher(self) -> CrewAgent: ...

        @task
        def research(self) -> CrewTask: ...

        @crew
        def assemble(self) -> Crew: ...
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def agent[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Mark a zero-arg method as a :class:`CrewAgent` factory.

    Caches the result per instance so repeated calls return the same agent
    object — needed because Pydantic models are compared by identity in the
    ``CrewTask.context`` graph.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        attr = f"_crew_agent_cache__{fn.__name__}"
        if not hasattr(self, attr):
            setattr(self, attr, fn(self, *args, **kwargs))
        return getattr(self, attr)

    wrapper._crew_role = "agent"  # type: ignore[attr-defined]
    return wrapper


def task[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Mark a zero-arg method as a :class:`CrewTask` factory.

    Same per-instance caching as :func:`agent` — CrewTask participates in
    the context-graph by identity and must be de-duplicated across repeated
    calls within a single Crew assembly.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        attr = f"_crew_task_cache__{fn.__name__}"
        if not hasattr(self, attr):
            setattr(self, attr, fn(self, *args, **kwargs))
        return getattr(self, attr)

    wrapper._crew_role = "task"  # type: ignore[attr-defined]
    return wrapper


def crew[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Mark a method as the Crew assembly point.

    Does not cache — the assembly function typically builds a fresh Crew
    each call (e.g. to parameterize inputs from the caller). Agents and
    tasks it references will still be cached by their own decorators.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        return fn(self, *args, **kwargs)

    wrapper._crew_role = "crew"  # type: ignore[attr-defined]
    return wrapper
