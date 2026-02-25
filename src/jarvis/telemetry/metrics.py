"""Metrics Provider -- Counter, Histogram, Gauge (v19).

OTLP-kompatible Metriken für Jarvis:
  - Counter:   Monoton steigende Zähler (Requests, Errors, Tokens)
  - Histogram: Verteilungen (Latenz, Token-Counts)
  - Gauge:     Aktuelle Werte (aktive Connections, Queue-Größe)

Usage:
    metrics = MetricsProvider(service_name="jarvis")
    metrics.counter("requests_total", 1, method="POST")
    metrics.histogram("latency_ms", 42.5, endpoint="/chat")
    metrics.gauge("active_sessions", 12)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from jarvis.telemetry.types import (
    HistogramDataPoint,
    MetricDataPoint,
    MetricDefinition,
    MetricKind,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class MetricsProvider:
    """Zentrale Instanz für Metriken-Erfassung."""

    def __init__(self, service_name: str = "jarvis",
                 max_points_per_metric: int = 500) -> None:
        self._service_name = service_name
        self._max_points = max_points_per_metric
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, HistogramDataPoint] = {}
        self._time_series: dict[str, deque[MetricDataPoint]] = {}
        self._descriptions: dict[str, str] = {}
        self._units: dict[str, str] = {}

    # ── Counter ──────────────────────────────────────────────────

    def counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Inkrementiert einen Counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        self._record_point(key, self._counters[key], labels)

    def get_counter(self, name: str, **labels: str) -> float:
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    # ── Gauge ────────────────────────────────────────────────────

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Setzt einen Gauge-Wert."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._record_point(key, value, labels)

    def get_gauge(self, name: str, **labels: str) -> float:
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    # ── Histogram ────────────────────────────────────────────────

    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Zeichnet einen Wert in ein Histogram auf."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = HistogramDataPoint(attributes=labels)
        self._histograms[key].record(value)
        self._record_point(key, value, labels)

    def get_histogram(self, name: str, **labels: str) -> HistogramDataPoint | None:
        key = self._make_key(name, labels)
        return self._histograms.get(key)

    # ── Description & Units ──────────────────────────────────────

    def describe(self, name: str, description: str, unit: str = "") -> None:
        """Setzt Beschreibung und Einheit einer Metrik."""
        self._descriptions[name] = description
        if unit:
            self._units[name] = unit

    # ── Time Series ──────────────────────────────────────────────

    def get_history(self, name: str, last_n: int = 60,
                    **labels: str) -> list[dict[str, Any]]:
        """Gibt Zeitreihe einer Metrik zurück."""
        key = self._make_key(name, labels)
        points = self._time_series.get(key, deque())
        return [p.to_dict() for p in list(points)[-last_n:]]

    # ── Snapshot & Export ────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Vollständiger Snapshot aller Metriken."""
        result: dict[str, Any] = {
            "service": self._service_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: v.to_dict() for k, v in self._histograms.items()
            },
        }
        return result

    def get_all_metrics(self) -> list[MetricDefinition]:
        """Gibt alle Metriken als MetricDefinition zurück."""
        metrics: list[MetricDefinition] = []

        for key, value in self._counters.items():
            name = key.split("{")[0]
            metrics.append(MetricDefinition(
                name=name, kind=MetricKind.COUNTER,
                description=self._descriptions.get(name, ""),
                unit=self._units.get(name, ""),
                data_points=[MetricDataPoint(value=value)],
            ))

        for key, value in self._gauges.items():
            name = key.split("{")[0]
            metrics.append(MetricDefinition(
                name=name, kind=MetricKind.GAUGE,
                description=self._descriptions.get(name, ""),
                unit=self._units.get(name, ""),
                data_points=[MetricDataPoint(value=value)],
            ))

        for key, hist in self._histograms.items():
            name = key.split("{")[0]
            metrics.append(MetricDefinition(
                name=name, kind=MetricKind.HISTOGRAM,
                description=self._descriptions.get(name, ""),
                unit=self._units.get(name, ""),
                histogram=hist,
            ))

        return metrics

    def to_otlp(self) -> dict[str, Any]:
        """Exportiert Metriken im OTLP-Format."""
        scope_metrics: list[dict] = []

        for metric in self.get_all_metrics():
            m: dict[str, Any] = {
                "name": metric.name,
                "description": metric.description,
                "unit": metric.unit,
            }
            if metric.kind == MetricKind.COUNTER:
                m["sum"] = {
                    "dataPoints": [
                        {
                            "asDouble": dp.value,
                            "timeUnixNano": str(dp.timestamp_ns),
                            "attributes": [
                                {"key": k, "value": {"stringValue": v}}
                                for k, v in dp.attributes.items()
                            ],
                        }
                        for dp in metric.data_points
                    ],
                    "isMonotonic": True,
                    "aggregationTemporality": 2,  # CUMULATIVE
                }
            elif metric.kind == MetricKind.GAUGE:
                m["gauge"] = {
                    "dataPoints": [
                        {
                            "asDouble": dp.value,
                            "timeUnixNano": str(dp.timestamp_ns),
                        }
                        for dp in metric.data_points
                    ],
                }
            elif metric.kind == MetricKind.HISTOGRAM and metric.histogram:
                h = metric.histogram
                m["histogram"] = {
                    "dataPoints": [{
                        "count": str(h.count),
                        "sum": h.total,
                        "min": h.min_value if h.count else 0,
                        "max": h.max_value if h.count else 0,
                        "bucketCounts": [str(c) for c in h.bucket_counts],
                        "explicitBounds": h.bucket_boundaries,
                        "timeUnixNano": str(h.timestamp_ns),
                    }],
                    "aggregationTemporality": 2,
                }
            scope_metrics.append(m)

        return {
            "resourceMetrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name",
                         "value": {"stringValue": self._service_name}},
                    ],
                },
                "scopeMetrics": [{
                    "scope": {"name": "jarvis.telemetry", "version": "1.0"},
                    "metrics": scope_metrics,
                }],
            }],
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _make_key(self, name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record_point(self, key: str, value: float,
                      labels: dict[str, str]) -> None:
        if key not in self._time_series:
            self._time_series[key] = deque(maxlen=self._max_points)
        self._time_series[key].append(
            MetricDataPoint(value=value, attributes=labels)
        )

    def reset(self) -> None:
        """Setzt alle Metriken zurück."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._time_series.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "service_name": self._service_name,
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
            "total_metrics": len(self._counters) + len(self._gauges) + len(self._histograms),
        }
