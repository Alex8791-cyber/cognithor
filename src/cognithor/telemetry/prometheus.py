"""Prometheus-compatible metrics export for Cognithor.

Exports metrics in Prometheus Text Exposition Format
without external dependencies (no prometheus_client needed).

Format-Spezifikation: https://prometheus.io/docs/instrumenting/exposition_formats/

Unterstuetzte Metriken:
  - Counter:   jarvis_requests_total, jarvis_errors_total,
               jarvis_tokens_used_total, jarvis_tool_calls_total
  - Gauge:     jarvis_active_sessions, jarvis_queue_depth,
               jarvis_memory_usage_bytes, jarvis_uptime_seconds
  - Histogram: jarvis_request_duration_ms, jarvis_tool_duration_ms
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# ── Standard Jarvis Metrics ──────────────────────────────────────

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    # Counters
    "requests_total": {
        "type": "counter",
        "help": "Total number of messages processed",
    },
    "errors_total": {
        "type": "counter",
        "help": "Total number of errors",
    },
    "tokens_used_total": {
        "type": "counter",
        "help": "Total LLM tokens consumed",
    },
    "tool_calls_total": {
        "type": "counter",
        "help": "Total tool execution count",
    },
    # Gauges
    "active_sessions": {
        "type": "gauge",
        "help": "Number of active sessions",
    },
    "queue_depth": {
        "type": "gauge",
        "help": "Current message queue depth",
    },
    "memory_usage_bytes": {
        "type": "gauge",
        "help": "Process memory usage in bytes",
    },
    "uptime_seconds": {
        "type": "gauge",
        "help": "Process uptime in seconds",
    },
    # Histograms
    "request_duration_ms": {
        "type": "histogram",
        "help": "Request latency in milliseconds",
    },
    "tool_duration_ms": {
        "type": "histogram",
        "help": "Tool execution time in milliseconds",
    },
}


# ── Label Escaping ────────────────────────────────────────────────

_LABEL_VALUE_ESCAPE_RE = re.compile(r'([\\"\n])')
_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


def _escape_label_value(value: str) -> str:
    r"""Escape a label value for Prometheus text format.

    Rules (per spec):
      - backslash  → \\
      - double-quote → \"
      - newline → \n
    """
    return _LABEL_VALUE_ESCAPE_RE.sub(
        lambda m: {"\\": "\\\\", '"': '\\"', "\n": "\\n"}[m.group(0)],
        str(value),
    )


def _sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name to match [a-zA-Z_:][a-zA-Z0-9_:]*."""
    # Replace dots, hyphens with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _format_labels(labels: dict[str, str]) -> str:
    """Format a label dict as Prometheus label string: {key="val",key2="val2"}."""
    if not labels:
        return ""
    parts = []
    for k, v in sorted(labels.items()):
        safe_key = _sanitize_metric_name(k)
        safe_val = _escape_label_value(v)
        parts.append(f'{safe_key}="{safe_val}"')
    return "{" + ",".join(parts) + "}"


def _format_value(value: float) -> str:
    """Format a numeric value for Prometheus output."""
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    # Use integer format for whole numbers
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


# ── Prometheus Exporter ──────────────────────────────────────────


class PrometheusExporter:
    """Exports Jarvis metrics in Prometheus text format.

    Works with the existing MetricsProvider (telemetry/metrics.py)
    and the MetricCollector (gateway/monitoring.py).

    No prometheus_client needed -- pure text export.

    Usage:
        exporter = PrometheusExporter(metrics_provider=my_provider)
        text = exporter.export()  # -> Prometheus text exposition format
    """

    # Process start time for uptime calculation
    _process_start = time.monotonic()

    def __init__(
        self,
        metrics_provider: Any = None,
        metric_collector: Any = None,
        prefix: str = "jarvis",
    ) -> None:
        """Initialize the Prometheus exporter.

        Args:
            metrics_provider: A MetricsProvider instance (telemetry/metrics.py).
            metric_collector: A MetricCollector instance (gateway/monitoring.py).
            prefix: Prefix for all metric names (default: "jarvis").
        """
        self._metrics_provider = metrics_provider
        self._metric_collector = metric_collector
        self._prefix = prefix

    def export(self) -> str:
        """Generates Prometheus Text Exposition Format.

        Returns:
            String in Prometheus text exposition format, ready for /metrics.
        """
        lines: list[str] = []

        # 1. Export from MetricsProvider (telemetry/metrics.py)
        if self._metrics_provider is not None:
            lines.extend(self._export_metrics_provider())

        # 2. Export from MetricCollector (gateway/monitoring.py)
        if self._metric_collector is not None:
            lines.extend(self._export_metric_collector())

        # 3. Always export process-level metrics
        lines.extend(self._export_process_metrics())

        return "\n".join(lines) + "\n" if lines else ""

    # ── MetricsProvider export ────────────────────────────────────

    def _export_metrics_provider(self) -> list[str]:
        """Export metrics from the telemetry MetricsProvider."""
        lines: list[str] = []
        provider = self._metrics_provider

        # Track which metric names we've already emitted TYPE/HELP for
        emitted_names: set[str] = set()

        # Counters
        for key, value in provider._counters.items():
            name, labels = self._parse_key(key)
            full_name = self._full_name(name)
            if full_name not in emitted_names:
                lines.extend(
                    self._type_help_lines(
                        full_name,
                        "counter",
                        METRIC_DEFINITIONS.get(name, {}).get("help", ""),
                    )
                )
                emitted_names.add(full_name)
            lines.append(f"{full_name}{_format_labels(labels)} {_format_value(value)}")

        # Gauges
        for key, value in provider._gauges.items():
            name, labels = self._parse_key(key)
            full_name = self._full_name(name)
            if full_name not in emitted_names:
                lines.extend(
                    self._type_help_lines(
                        full_name,
                        "gauge",
                        METRIC_DEFINITIONS.get(name, {}).get("help", ""),
                    )
                )
                emitted_names.add(full_name)
            lines.append(f"{full_name}{_format_labels(labels)} {_format_value(value)}")

        # Histograms
        for key, hist in provider._histograms.items():
            name, labels = self._parse_key(key)
            full_name = self._full_name(name)
            if full_name not in emitted_names:
                lines.extend(
                    self._type_help_lines(
                        full_name,
                        "histogram",
                        METRIC_DEFINITIONS.get(name, {}).get("help", ""),
                    )
                )
                emitted_names.add(full_name)
            lines.extend(self._format_histogram(full_name, hist, labels))

        return lines

    # ── MetricCollector export ────────────────────────────────────

    def _export_metric_collector(self) -> list[str]:
        """Export metrics from the gateway MetricCollector."""
        lines: list[str] = []
        collector = self._metric_collector

        emitted_names: set[str] = set()

        # Counters
        for name, value in collector._counters.items():
            full_name = self._full_name(_sanitize_metric_name(name))
            if full_name not in emitted_names:
                lines.extend(self._type_help_lines(full_name, "counter", ""))
                emitted_names.add(full_name)
            lines.append(f"{full_name} {_format_value(value)}")

        # Gauges
        for name, value in collector._gauges.items():
            full_name = self._full_name(_sanitize_metric_name(name))
            if full_name not in emitted_names:
                lines.extend(self._type_help_lines(full_name, "gauge", ""))
                emitted_names.add(full_name)
            lines.append(f"{full_name} {_format_value(value)}")

        return lines

    # ── Process metrics ───────────────────────────────────────────

    def _export_process_metrics(self) -> list[str]:
        """Export process-level metrics (memory, uptime)."""
        lines: list[str] = []

        # Uptime
        uptime_name = self._full_name("uptime_seconds")
        uptime = time.monotonic() - self._process_start
        lines.extend(
            self._type_help_lines(
                uptime_name,
                "gauge",
                "Process uptime in seconds",
            )
        )
        lines.append(f"{uptime_name} {_format_value(uptime)}")

        # Memory usage (RSS)
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KB -> bytes
            mem_name = self._full_name("memory_usage_bytes")
            lines.extend(
                self._type_help_lines(
                    mem_name,
                    "gauge",
                    "Process memory usage in bytes (RSS)",
                )
            )
            lines.append(f"{mem_name} {_format_value(float(rss))}")
        except ImportError:
            # Windows: use psutil or os-specific approach
            try:
                import psutil

                proc = psutil.Process(os.getpid())
                rss = proc.memory_info().rss
                mem_name = self._full_name("memory_usage_bytes")
                lines.extend(
                    self._type_help_lines(
                        mem_name,
                        "gauge",
                        "Process memory usage in bytes (RSS)",
                    )
                )
                lines.append(f"{mem_name} {_format_value(float(rss))}")
            except (ImportError, Exception):
                pass  # Memory metric unavailable on this platform

        return lines

    # ── Histogram formatting ──────────────────────────────────────

    def _format_histogram(
        self,
        full_name: str,
        hist: Any,
        labels: dict[str, str],
    ) -> list[str]:
        """Format a HistogramDataPoint as Prometheus histogram lines.

        Prometheus histogram format:
          metric_bucket{le="5"} 24054
          metric_bucket{le="10"} 33444
          ...
          metric_bucket{le="+Inf"} 144320
          metric_count 144320
          metric_sum 53423
        """
        lines: list[str] = []
        label_str_base = labels.copy()

        # Cumulative bucket counts (Prometheus requires cumulative)
        cumulative = 0
        for i, boundary in enumerate(hist.bucket_boundaries):
            cumulative += hist.bucket_counts[i] if i < len(hist.bucket_counts) else 0
            bucket_labels = label_str_base.copy()
            bucket_labels["le"] = _format_value(float(boundary))
            lines.append(f"{full_name}_bucket{_format_labels(bucket_labels)} {cumulative}")

        # +Inf bucket (total count)
        if hist.bucket_counts:
            cumulative += (
                hist.bucket_counts[-1]
                if len(hist.bucket_counts) > len(hist.bucket_boundaries)
                else 0
            )
        inf_labels = label_str_base.copy()
        inf_labels["le"] = "+Inf"
        lines.append(f"{full_name}_bucket{_format_labels(inf_labels)} {hist.count}")

        # _sum and _count
        lines.append(f"{full_name}_sum{_format_labels(label_str_base)} {_format_value(hist.total)}")
        lines.append(f"{full_name}_count{_format_labels(label_str_base)} {hist.count}")

        return lines

    # ── Helpers ───────────────────────────────────────────────────

    def _full_name(self, name: str) -> str:
        """Prefix a metric name."""
        sanitized = _sanitize_metric_name(name)
        if self._prefix:
            prefix = _sanitize_metric_name(self._prefix)
            if sanitized.startswith(prefix + "_"):
                return sanitized
            return f"{prefix}_{sanitized}"
        return sanitized

    def _parse_key(self, key: str) -> tuple[str, dict[str, str]]:
        """Parse a MetricsProvider key like 'name{k1=v1,k2=v2}' into (name, labels)."""
        if "{" not in key:
            return key, {}
        name, rest = key.split("{", 1)
        label_str = rest.rstrip("}")
        labels: dict[str, str] = {}
        if label_str:
            for pair in label_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip()
        return name, labels

    def _type_help_lines(self, full_name: str, type_str: str, help_str: str) -> list[str]:
        """Generate # HELP and # TYPE lines."""
        lines: list[str] = []
        if help_str:
            lines.append(f"# HELP {full_name} {help_str}")
        lines.append(f"# TYPE {full_name} {type_str}")
        return lines
