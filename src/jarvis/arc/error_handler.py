"""ARC-AGI-3 error handling utilities: exceptions, retry decorator, and safe frame extraction."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger(__name__)

__all__ = [
    "ArcAgentError",
    "EnvironmentConnectionError",
    "FrameExtractionError",
    "GameRunGuard",
    "retry_on_error",
    "safe_frame_extract",
]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ArcAgentError(Exception):
    """Base exception for ARC agent errors."""


class FrameExtractionError(ArcAgentError):
    """Raised when a grid frame cannot be extracted from an observation."""


class EnvironmentConnectionError(ArcAgentError):
    """Raised when the ARC arcade environment cannot be created or connected."""


# ---------------------------------------------------------------------------
# retry_on_error decorator
# ---------------------------------------------------------------------------


def retry_on_error(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """Decorator that retries a function with exponential backoff on specified exceptions.

    Args:
        max_retries: Maximum number of retry attempts after initial failure.
        delay_seconds: Initial delay in seconds between retries.
        backoff_factor: Multiplier applied to delay after each retry.
        exceptions: Tuple of exception types to catch and retry on.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay_seconds
            last_exc: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        log.warning(
                            "arc.retry",
                            func=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=current_delay,
                            error=str(exc),
                        )
                        if current_delay > 0:
                            time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        log.error(
                            "arc.retry_exhausted",
                            func=func.__name__,
                            max_retries=max_retries,
                            error=str(exc),
                        )

            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# safe_frame_extract
# ---------------------------------------------------------------------------

_FRAME_ATTRS = ("frame", "frame_data", "grid", "pixels", "data", "image")


def safe_frame_extract(
    obs: Any,
    fallback_shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Safely extract a 2-D colour-index grid from an ARC observation.

    The real SDK delivers ``obs.frame`` with shape ``(1, 64, 64)`` and dtype
    ``int8``.  This function normalises the many shapes that can appear during
    testing and falls back to a zero-filled array on complete failure.

    Args:
        obs: Observation object (FrameDataRaw or any duck-typed stand-in).
        fallback_shape: Shape of the zero array returned when extraction fails.

    Returns:
        2-D numpy array of shape ``fallback_shape`` (or extracted grid shape).
    """
    h, w = fallback_shape

    if obs is None:
        log.debug("arc.safe_frame_extract.none_obs")
        return np.zeros(fallback_shape, dtype=np.int8)

    raw: Any = None
    used_attr: str | None = None

    for attr in _FRAME_ATTRS:
        if hasattr(obs, attr):
            raw = getattr(obs, attr)
            used_attr = attr
            break

    if raw is None:
        log.warning("arc.safe_frame_extract.no_attr", attrs=_FRAME_ATTRS)
        return np.zeros(fallback_shape, dtype=np.int8)

    try:
        arr = np.asarray(raw, dtype=np.int8)
    except Exception as exc:
        log.warning("arc.safe_frame_extract.convert_failed", attr=used_attr, error=str(exc))
        return np.zeros(fallback_shape, dtype=np.int8)

    # Already 2-D and correct shape → pass through
    if arr.ndim == 2:
        return arr

    # Real SDK: (1, H, W) → squeeze leading dimension
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr.squeeze(0)

    # Multi-layer frame: (N, H, W) where N > 1 → take first layer
    # ARC-AGI-3 can return multi-layer frames as game complexity increases
    if arr.ndim == 3 and arr.shape[1:] == (h, w):
        return arr[0]

    # Flat 1-D array with exactly H*W elements → reshape
    if arr.ndim == 1 and arr.size == h * w:
        return arr.reshape(fallback_shape)

    # Unexpected shape — fall back
    log.warning(
        "arc.safe_frame_extract.unexpected_shape",
        attr=used_attr,
        shape=arr.shape,
    )
    return np.zeros(fallback_shape, dtype=np.int8)


# ---------------------------------------------------------------------------
# GameRunGuard context manager
# ---------------------------------------------------------------------------


class GameRunGuard:
    """Context manager that wraps a single ARC game run.

    Ensures the environment is properly created and always attempts to fetch a
    scorecard on exit, regardless of whether an exception occurred during the
    run.  Any exceptions raised inside the ``with`` block are suppressed and
    recorded in ``self.errors``.

    Args:
        arcade: The ARC arcade object — must support ``make(game_id)`` and
            ``get_scorecard()``.
        game_id: Identifier of the game/environment to create.
    """

    def __init__(self, arcade: Any, game_id: str) -> None:
        self.arcade = arcade
        self.game_id = game_id
        self.env: Any = None
        self.errors: list[dict[str, Any]] = []

    def __enter__(self) -> GameRunGuard:
        try:
            env = self.arcade.make(self.game_id)
        except Exception as exc:
            raise EnvironmentConnectionError(
                f"Failed to create environment for game '{self.game_id}': {exc}"
            ) from exc

        if env is None:
            raise EnvironmentConnectionError(f"arcade.make('{self.game_id}') returned None")

        self.env = env
        log.info("arc.game_run_guard.enter", game_id=self.game_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if exc_val is not None:
            error_entry: dict[str, Any] = {
                "game_id": self.game_id,
                "error": str(exc_val),
                "type": type(exc_val).__name__,
            }
            self.errors.append(error_entry)
            log.error(
                "arc.game_run_guard.exception",
                game_id=self.game_id,
                error=str(exc_val),
                exc_type=type(exc_val).__name__,
            )

        # Always attempt to retrieve scorecard
        try:
            scorecard = self.arcade.get_scorecard()
            if scorecard is not None:
                log.info(
                    "arc.game_run_guard.scorecard",
                    game_id=self.game_id,
                    score=getattr(scorecard, "score", None),
                )
        except Exception as sc_exc:
            log.warning(
                "arc.game_run_guard.scorecard_failed",
                game_id=self.game_id,
                error=str(sc_exc),
            )

        log.info("arc.game_run_guard.exit", game_id=self.game_id)
        if exc_type is not None and not issubclass(exc_type, Exception):
            return False
        return True
