# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PSE observability: telemetry counters, histograms, audit trail.

Phase-1 ships in-process emitters that the channel hot-path can call
synchronously. External wiring (Prometheus scrape endpoint,
Hashline-Guard writer) reads the snapshots / chain-hash these modules
expose.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.observability.audit import (
    AuditEntry,
    AuditTrail,
    audit_entry_for,
)
from cognithor.channels.program_synthesis.observability.metrics import (
    CANDIDATES_EXPLORED_BUCKETS,
    DEFAULT_REGISTRY,
    DURATION_SECONDS_BUCKETS,
    PROGRAM_DEPTH_BUCKETS,
    PROGRAM_SIZE_BUCKETS,
    Counter,
    Histogram,
    HistogramSnapshot,
    Registry,
    standard_counters,
    standard_histograms,
)

__all__ = [
    "CANDIDATES_EXPLORED_BUCKETS",
    "DEFAULT_REGISTRY",
    "DURATION_SECONDS_BUCKETS",
    "PROGRAM_DEPTH_BUCKETS",
    "PROGRAM_SIZE_BUCKETS",
    "AuditEntry",
    "AuditTrail",
    "Counter",
    "Histogram",
    "HistogramSnapshot",
    "Registry",
    "audit_entry_for",
    "standard_counters",
    "standard_histograms",
]
