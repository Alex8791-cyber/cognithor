"""Automatisches Error-Clustering basierend auf Levenshtein-Aehnlichkeit."""

from __future__ import annotations

import hashlib
from datetime import datetime, UTC
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Berechnet Levenshtein-Aehnlichkeit (0-1). 1 = identisch."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Truncate for performance
    s1 = s1[:200]
    s2 = s2[:200]

    len1, len2 = len(s1), len(s2)
    # Quick length-based rejection
    if abs(len1 - len2) > max(len1, len2) * 0.5:
        return 0.0

    # Standard DP Levenshtein
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev_row = list(range(len1 + 1))
    for j in range(1, len2 + 1):
        curr_row = [j] + [0] * len1
        for i in range(1, len1 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                curr_row[i - 1] + 1,
                prev_row[i] + 1,
                prev_row[i - 1] + cost,
            )
        prev_row = curr_row

    distance = prev_row[len1]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


class _ErrorEntry:
    """Interner Error-Eintrag."""
    __slots__ = ("error_type", "message", "context", "timestamp")

    def __init__(self, error_type: str, message: str, context: str, timestamp: datetime) -> None:
        self.error_type = error_type
        self.message = message
        self.context = context
        self.timestamp = timestamp


class ErrorClusterer:
    """Gruppiert aehnliche Fehler fuer Pattern-Erkennung."""

    def __init__(self, similarity_threshold: float = 0.6, max_entries: int = 10000) -> None:
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._errors: list[_ErrorEntry] = []

    def add_error(
        self,
        error_type: str,
        message: str,
        context: str = "",
    ) -> None:
        """Nimmt einen Fehler auf."""
        entry = _ErrorEntry(
            error_type=error_type,
            message=message,
            context=context,
            timestamp=datetime.now(UTC),
        )
        self._errors.append(entry)
        # Evict old entries if over limit
        if len(self._errors) > self._max_entries:
            self._errors = self._errors[-self._max_entries:]

    def get_clusters(self) -> list[dict[str, Any]]:
        """Gruppiert aehnliche Fehler (pre-bucketed by error_type)."""
        if not self._errors:
            return []

        # Pre-bucket by error_type
        buckets: dict[str, list[tuple[int, _ErrorEntry]]] = {}
        for i, entry in enumerate(self._errors):
            buckets.setdefault(entry.error_type, []).append((i, entry))

        clusters: list[dict[str, Any]] = []
        assigned: set[int] = set()

        for _etype, bucket in buckets.items():
            for bi, (i, entry) in enumerate(bucket):
                if i in assigned:
                    continue
                cluster_entries = [entry]
                assigned.add(i)
                # Limit comparisons per seed to avoid O(n^2)
                compare_limit = min(len(bucket), bi + 200)
                for bj in range(bi + 1, compare_limit):
                    j, other = bucket[bj]
                    if j in assigned:
                        continue
                    ratio = _levenshtein_ratio(entry.message, other.message)
                    if ratio >= self._threshold:
                        cluster_entries.append(other)
                        assigned.add(j)

                timestamps = [e.timestamp for e in cluster_entries]
                examples = list({e.message for e in cluster_entries})[:5]

                hash_input = f"{entry.error_type}:{entry.message[:100]}"
                cluster_id = f"{entry.error_type}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
                clusters.append({
                    "id": cluster_id,
                    "pattern": f"{entry.error_type}: {entry.message[:100]}",
                    "count": len(cluster_entries),
                    "examples": examples,
                    "first_seen": min(timestamps),
                    "last_seen": max(timestamps),
                    "severity": "high" if len(cluster_entries) >= 10 else
                               "medium" if len(cluster_entries) >= 3 else "low",
                })

        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters

    def get_top_errors(self, n: int = 5) -> list[dict[str, Any]]:
        """Die haeufigsten Fehler-Muster."""
        return self.get_clusters()[:n]

    @property
    def total_errors(self) -> int:
        return len(self._errors)

    def clear(self) -> None:
        self._errors.clear()
