# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Internal typing helpers for the config-routes sub-package.

The 16 ``_register_*_routes()`` helpers all take an ``app`` parameter
that is duck-typed to FastAPI's ``app.get(path)`` / ``app.post(path)``
/ ... decorator interface. Passing ``app: Any`` leaves every
decorator call as an *untyped decorator* under ``mypy --strict``,
producing 300+ false-positive errors that drown out real signal.

The ``RoutableApp`` Protocol below pins the surface we actually use
(GET/POST/PUT/PATCH/DELETE/WEBSOCKET) so each handler decoration is
typed without forcing an actual ``fastapi.FastAPI`` import at the
call-site (helpful when FastAPI is missing in trimmed-down test
environments).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class RoutableApp(Protocol):
    """Subset of the FastAPI app surface the config-routes modules use.

    Each method here mirrors FastAPI's decorator factories — they
    accept the route path plus arbitrary keyword arguments
    (``dependencies=``, ``response_model=``, ...) and return a
    decorator that leaves the handler's signature unchanged.
    """

    def get(self, path: str, **kwargs: Any) -> Callable[[F], F]: ...

    def post(self, path: str, **kwargs: Any) -> Callable[[F], F]: ...

    def put(self, path: str, **kwargs: Any) -> Callable[[F], F]: ...

    def patch(self, path: str, **kwargs: Any) -> Callable[[F], F]: ...

    def delete(self, path: str, **kwargs: Any) -> Callable[[F], F]: ...

    def websocket(self, path: str, **kwargs: Any) -> Callable[[F], F]: ...


__all__ = ["RoutableApp"]
