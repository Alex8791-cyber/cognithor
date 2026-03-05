"""CircuitBreaker — Schutz gegen kaskadierende Fehler bei externen APIs.

State-Machine:
  CLOSED → (failure_threshold konsekutive Fehler) → OPEN
  OPEN   → (recovery_timeout abgelaufen)          → HALF_OPEN
  HALF_OPEN → (half_open_max_calls Erfolge)        → CLOSED
  HALF_OPEN → (ein Fehler)                         → OPEN

Referenz: Stabilitäts-Verbesserung §8 (Channel Circuit Breaker)
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from typing import Awaitable, TypeVar

__all__ = ["CircuitBreaker", "CircuitBreakerOpen", "CircuitState"]

log = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(StrEnum):
    """Circuit-Breaker-Zustände."""

    closed = "closed"
    open = "open"
    half_open = "half_open"


class CircuitBreakerOpen(Exception):
    """Wird geworfen wenn der Circuit offen ist."""

    def __init__(self, name: str, remaining_seconds: float) -> None:
        self.name = name
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit breaker '{name}' is open (retry in {remaining_seconds:.1f}s)"
        )


class CircuitBreaker:
    """Async Circuit Breaker für externe API-Aufrufe.

    Args:
        name: Bezeichnung des Circuit Breakers (für Logging).
        failure_threshold: Konsekutive Fehler bis OPEN.
        recovery_timeout: Sekunden bis HALF_OPEN nach OPEN.
        half_open_max_calls: Erfolge in HALF_OPEN bis CLOSED.
        excluded_exceptions: Exception-Typen die nicht als Failure zählen.
    """

    def __init__(
        self,
        *,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._excluded_exceptions = excluded_exceptions

        self._state = CircuitState.closed
        self._failure_count = 0
        self._half_open_successes = 0
        self._half_open_inflight = 0  # Anzahl aktiver Calls in HALF_OPEN
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

        # Statistiken
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._state_changes = 0

    @property
    def state(self) -> CircuitState:
        """Aktueller Zustand."""
        return self._state

    @property
    def stats(self) -> dict[str, str | int]:
        """Statistiken des Circuit Breakers."""
        return {
            "name": self._name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "state_changes": self._state_changes,
        }

    async def call(self, coro: Awaitable[T]) -> T:
        """Führt eine async Operation unter Circuit-Breaker-Schutz aus.

        Args:
            coro: Awaitable das ausgeführt werden soll.

        Returns:
            Ergebnis der Operation.

        Raises:
            CircuitBreakerOpen: Wenn der Circuit offen ist.
        """
        async with self._lock:
            self._total_calls += 1
            now = time.monotonic()

            if self._state == CircuitState.open:
                elapsed = now - self._opened_at
                if elapsed < self._recovery_timeout:
                    self._total_rejections += 1
                    raise CircuitBreakerOpen(
                        self._name, self._recovery_timeout - elapsed
                    )
                # Timeout abgelaufen → HALF_OPEN
                self._transition(CircuitState.half_open)
                self._half_open_successes = 0
                self._half_open_inflight = 0

            # HALF_OPEN Admission Control: nur max N Calls gleichzeitig
            if self._state == CircuitState.half_open:
                if self._half_open_inflight >= self._half_open_max_calls:
                    self._total_rejections += 1
                    raise CircuitBreakerOpen(
                        self._name, self._recovery_timeout
                    )
                self._half_open_inflight += 1

        # Track whether we incremented inflight (only in HALF_OPEN)
        _was_half_open = False
        async with self._lock:
            _was_half_open = self._state == CircuitState.half_open

        try:
            result = await coro
        except BaseException as exc:
            async with self._lock:
                if _was_half_open:
                    self._half_open_inflight = max(0, self._half_open_inflight - 1)
                if not isinstance(exc, self._excluded_exceptions):
                    self._record_failure()
            raise
        else:
            async with self._lock:
                if _was_half_open:
                    self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._record_success()
            return result

    def reset(self) -> None:
        """Manueller Reset zu CLOSED."""
        self._state = CircuitState.closed
        self._failure_count = 0
        self._half_open_successes = 0
        self._state_changes += 1
        log.info("circuit_breaker_reset", extra={"cb_name": self._name})

    # ------------------------------------------------------------------
    # Interne State-Transitions (müssen unter Lock aufgerufen werden)
    # ------------------------------------------------------------------

    def _record_failure(self) -> None:
        """Registriert einen Fehler und transitiert ggf. zu OPEN."""
        self._total_failures += 1
        self._failure_count += 1

        if self._state == CircuitState.half_open:
            # Ein Fehler in HALF_OPEN → sofort OPEN
            self._transition(CircuitState.open)
            self._opened_at = time.monotonic()
        elif self._state == CircuitState.closed:
            if self._failure_count >= self._failure_threshold:
                self._transition(CircuitState.open)
                self._opened_at = time.monotonic()

    def _record_success(self) -> None:
        """Registriert einen Erfolg und transitiert ggf. zu CLOSED."""
        if self._state == CircuitState.half_open:
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_max_calls:
                self._transition(CircuitState.closed)
                self._failure_count = 0
        elif self._state == CircuitState.closed:
            self._failure_count = 0

    def _transition(self, new_state: CircuitState) -> None:
        """State-Transition mit Logging."""
        old_state = self._state
        self._state = new_state
        self._state_changes += 1

        if new_state == CircuitState.open:
            log.warning(
                "circuit_breaker_opened",
                extra={
                    "cb_name": self._name,
                    "from_state": old_state.value,
                    "failure_count": self._failure_count,
                },
            )
        elif new_state == CircuitState.closed:
            log.info(
                "circuit_breaker_closed",
                extra={
                    "cb_name": self._name,
                    "from_state": old_state.value,
                },
            )
        else:
            log.info(
                "circuit_breaker_half_open",
                extra={
                    "cb_name": self._name,
                    "from_state": old_state.value,
                },
            )
