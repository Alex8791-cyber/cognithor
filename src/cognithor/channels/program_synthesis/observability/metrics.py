# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Telemetry counters + histograms (spec §17.1, §17.2).

Phase-1 ships a Prometheus-shaped, in-process metrics registry. Real
Prometheus / OpenTelemetry export is a thin wrapper over this — it
reads ``Registry.snapshot()`` and forwards the values. The channel
emits its metrics through ``DEFAULT_REGISTRY`` so callers can either
poll programmatically (cheap, deterministic) or hook a metrics-export
backend later.

Spec inventory of metric names::

    Counters
        pse_synthesis_requests_total{status, domain}
        pse_sandbox_violations_total{kind}
        pse_cache_hits_total
        pse_cache_misses_total
        pse_dsl_primitive_uses_total{primitive}

    Histograms (bucketed)
        pse_synthesis_duration_seconds  buckets=(0.1, 0.5, 1, 5, 10, 30, 60)
        pse_candidates_explored         buckets=(100, 1k, 10k, 100k, 1M)
        pse_program_depth               buckets=(1..6)
        pse_program_size                buckets=(1, 5, 10, 20, 50)

Counters are append-only; histograms are bucket counters + sum.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


def _label_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Canonical hashable form of a label-set."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


class Counter:
    """Append-only counter with optional label dimensions.

    Thread-safe: the channel's enumerator is single-threaded but
    metric reads come from observers that may live on other threads
    (Prometheus scrape, dashboard polling).
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        if value < 0:
            raise ValueError(f"Counter.inc must be non-negative; got {value}")
        key = _label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def value(self, **labels: str) -> float:
        key = _label_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def snapshot(self) -> dict[tuple[tuple[str, str], ...], float]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistogramSnapshot:
    """Immutable view of a Histogram's state."""

    buckets: tuple[float, ...]
    counts: tuple[int, ...]  # cumulative ≤-bucket counts
    sum_value: float
    count: int


class Histogram:
    """Prometheus-style cumulative-bucket histogram.

    ``buckets`` are the upper bounds (exclusive on the upper end of
    each bucket; the final +Inf bucket is implicit and equals
    ``count``).
    """

    def __init__(
        self,
        name: str,
        buckets: tuple[float, ...],
        description: str = "",
    ) -> None:
        if not buckets or list(buckets) != sorted(buckets):
            raise ValueError("buckets must be a non-empty ascending sequence")
        self.name = name
        self.description = description
        self._buckets = tuple(buckets)
        self._counts: list[int] = [0] * len(self._buckets)
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for i, upper in enumerate(self._buckets):
                if value <= upper:
                    self._counts[i] += 1

    def snapshot(self) -> HistogramSnapshot:
        with self._lock:
            return HistogramSnapshot(
                buckets=self._buckets,
                counts=tuple(self._counts),
                sum_value=self._sum,
                count=self._count,
            )

    def reset(self) -> None:
        with self._lock:
            self._counts = [0] * len(self._buckets)
            self._sum = 0.0
            self._count = 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class _RegistrySnapshot:
    counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = field(default_factory=dict)
    histograms: dict[str, HistogramSnapshot] = field(default_factory=dict)


class Registry:
    """Holds named Counter / Histogram instances.

    Each metric is registered exactly once; subsequent ``counter`` /
    ``histogram`` calls with the same name return the existing instance
    (idempotent — the channel can wire its emit-points without
    coordinating who registers first).
    """

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            existing = self._counters.get(name)
            if existing is not None:
                return existing
            c = Counter(name, description)
            self._counters[name] = c
            return c

    def histogram(
        self,
        name: str,
        buckets: tuple[float, ...],
        description: str = "",
    ) -> Histogram:
        with self._lock:
            existing = self._histograms.get(name)
            if existing is not None:
                if existing._buckets != tuple(buckets):
                    raise ValueError(
                        f"Histogram {name!r} already registered with different buckets"
                    )
                return existing
            h = Histogram(name, buckets, description)
            self._histograms[name] = h
            return h

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted({*self._counters, *self._histograms}))

    def snapshot(self) -> _RegistrySnapshot:
        with self._lock:
            return _RegistrySnapshot(
                counters={n: c.snapshot() for n, c in self._counters.items()},
                histograms={n: h.snapshot() for n, h in self._histograms.items()},
            )

    def reset(self) -> None:
        with self._lock:
            for c in self._counters.values():
                c.reset()
            for h in self._histograms.values():
                h.reset()


# Default bucket sequences for the spec-listed histograms.
DURATION_SECONDS_BUCKETS: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
CANDIDATES_EXPLORED_BUCKETS: tuple[float, ...] = (100, 1_000, 10_000, 100_000, 1_000_000)
PROGRAM_DEPTH_BUCKETS: tuple[float, ...] = (1, 2, 3, 4, 5, 6)
PROGRAM_SIZE_BUCKETS: tuple[float, ...] = (1, 5, 10, 20, 50)


# Default registry. Tests construct their own ``Registry()`` instances
# rather than mutating this one.
DEFAULT_REGISTRY = Registry()


# ---------------------------------------------------------------------------
# Pre-registered standard metrics — call from the channel's hot paths.
# ---------------------------------------------------------------------------


def standard_counters(registry: Registry | None = None) -> dict[str, Counter]:
    """Bind the spec-mandated counters once.

    Returns a dict so the channel can reference them by short key
    rather than re-resolving the registry on every emit.
    """
    r = registry if registry is not None else DEFAULT_REGISTRY
    return {
        "synthesis_requests_total": r.counter(
            "pse_synthesis_requests_total",
            "Total synthesis requests by status + domain.",
        ),
        "sandbox_violations_total": r.counter(
            "pse_sandbox_violations_total",
            "Sandbox-policy violations by kind.",
        ),
        "cache_hits_total": r.counter("pse_cache_hits_total", "Tactical-memory cache hits."),
        "cache_misses_total": r.counter("pse_cache_misses_total", "Tactical-memory cache misses."),
        "dsl_primitive_uses_total": r.counter(
            "pse_dsl_primitive_uses_total",
            "Per-primitive use count across all solved programs.",
        ),
    }


def standard_histograms(registry: Registry | None = None) -> dict[str, Histogram]:
    r = registry if registry is not None else DEFAULT_REGISTRY
    return {
        "synthesis_duration_seconds": r.histogram(
            "pse_synthesis_duration_seconds",
            DURATION_SECONDS_BUCKETS,
            "Wall-clock seconds per synthesis request.",
        ),
        "candidates_explored": r.histogram(
            "pse_candidates_explored",
            CANDIDATES_EXPLORED_BUCKETS,
            "Number of enumerated candidates per request.",
        ),
        "program_depth": r.histogram(
            "pse_program_depth", PROGRAM_DEPTH_BUCKETS, "Depth of solved programs."
        ),
        "program_size": r.histogram(
            "pse_program_size", PROGRAM_SIZE_BUCKETS, "Node count of solved programs."
        ),
    }


__all__ = [
    "CANDIDATES_EXPLORED_BUCKETS",
    "DEFAULT_REGISTRY",
    "DURATION_SECONDS_BUCKETS",
    "PROGRAM_DEPTH_BUCKETS",
    "PROGRAM_SIZE_BUCKETS",
    "Counter",
    "Histogram",
    "HistogramSnapshot",
    "Registry",
    "standard_counters",
    "standard_histograms",
]


# Suppress unused-import lint for the Iterable forward-reference (kept
# for future signature additions).
_ = Iterable
