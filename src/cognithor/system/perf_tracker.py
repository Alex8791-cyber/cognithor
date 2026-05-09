"""Layer 7 — Performance Telemetry (local-only).

Tracks per-model + per-tier inference latency in `~/.cognithor/perf_telemetry.jsonl`.
Used by the drift-detector to spot regressions vs the manifest's
`performance_estimates.planner_tok_s_p50` baseline.

Privacy: ALL data stays local. No fields contain prompts, responses, or
hostnames. Only model-id + tier-id + numeric metrics.

Rotation: file is capped at 50 MB; on overflow we rotate to
`perf_telemetry.jsonl.1` (single-step, no archival chain).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["PerfRecord", "PerfTracker", "get_default_tracker"]


_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB before rotation
_MAX_RECORDS_IN_MEMORY = 5_000  # keep recent records in-memory for fast p95


@dataclass(frozen=True)
class PerfRecord:
    timestamp: float
    model_id: str
    tier_id: str
    backend: str
    prompt_tokens: int
    completion_tokens: int
    first_token_ms: int
    total_ms: int
    error: str | None = None

    @property
    def tokens_per_s(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return (self.completion_tokens * 1000.0) / self.total_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": round(self.timestamp, 3),
            "model_id": self.model_id,
            "tier_id": self.tier_id,
            "backend": self.backend,
            "in": self.prompt_tokens,
            "out": self.completion_tokens,
            "ftt_ms": self.first_token_ms,
            "ms": self.total_ms,
            "tps": round(self.tokens_per_s, 2),
            "err": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerfRecord:
        return cls(
            timestamp=float(d.get("ts", 0)),
            model_id=str(d.get("model_id", "")),
            tier_id=str(d.get("tier_id", "")),
            backend=str(d.get("backend", "")),
            prompt_tokens=int(d.get("in", 0)),
            completion_tokens=int(d.get("out", 0)),
            first_token_ms=int(d.get("ftt_ms", 0)),
            total_ms=int(d.get("ms", 0)),
            error=d.get("err"),
        )


class PerfTracker:
    """Thread-safe append-only perf log + rolling in-memory window."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".cognithor" / "perf_telemetry.jsonl")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: list[PerfRecord] = []
        self._lock = Lock()
        self._load_recent()

    def record(
        self,
        *,
        model_id: str,
        tier_id: str,
        backend: str,
        prompt_tokens: int,
        completion_tokens: int,
        first_token_ms: int,
        total_ms: int,
        error: str | None = None,
    ) -> None:
        rec = PerfRecord(
            timestamp=time.time(),
            model_id=model_id,
            tier_id=tier_id,
            backend=backend,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            error=error,
        )
        with self._lock:
            self._memory.append(rec)
            if len(self._memory) > _MAX_RECORDS_IN_MEMORY:
                self._memory.pop(0)
            self._append_disk(rec)
            self._maybe_rotate()

    def rolling_p95_tokens_per_s(self, model_id: str, *, window_s: float = 86400) -> float | None:
        cutoff = time.time() - window_s
        with self._lock:
            samples = [
                r.tokens_per_s
                for r in self._memory
                if r.model_id == model_id and r.error is None and r.timestamp >= cutoff
            ]
        if len(samples) < 5:
            return None  # Not enough data for a meaningful p95
        samples.sort()
        idx = int(len(samples) * 0.95)
        return samples[min(idx, len(samples) - 1)]

    def rolling_p50_first_token_ms(self, model_id: str, *, window_s: float = 86400) -> float | None:
        cutoff = time.time() - window_s
        with self._lock:
            samples = [
                r.first_token_ms
                for r in self._memory
                if r.model_id == model_id and r.error is None and r.timestamp >= cutoff
            ]
        if len(samples) < 5:
            return None
        samples.sort()
        return float(samples[len(samples) // 2])

    def model_summary(self, *, window_s: float = 86400) -> dict[str, dict[str, Any]]:
        cutoff = time.time() - window_s
        out: dict[str, list[PerfRecord]] = {}
        with self._lock:
            for r in self._memory:
                if r.timestamp < cutoff or r.error is not None:
                    continue
                out.setdefault(r.model_id, []).append(r)
        result = {}
        for model_id, recs in out.items():
            tps = sorted(r.tokens_per_s for r in recs)
            ftt = sorted(r.first_token_ms for r in recs)
            n = len(tps)
            result[model_id] = {
                "samples": n,
                "tps_p50": tps[n // 2] if n else 0.0,
                "tps_p95": tps[min(int(n * 0.95), n - 1)] if n else 0.0,
                "first_token_ms_p50": float(ftt[n // 2]) if n else 0.0,
                "first_token_ms_p95": float(ftt[min(int(n * 0.95), n - 1)]) if n else 0.0,
            }
        return result

    # ── Disk I/O ──────────────────────────────────────────────────────

    def _load_recent(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        # Load last N entries
        for line in lines[-_MAX_RECORDS_IN_MEMORY:]:
            with contextlib.suppress(json.JSONDecodeError):
                self._memory.append(PerfRecord.from_dict(json.loads(line)))

    def _append_disk(self, rec: PerfRecord) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict()) + "\n")
        except OSError as exc:
            log.warning("perf_tracker_disk_write_failed", error=str(exc))

    def _maybe_rotate(self) -> None:
        try:
            if not self._path.exists():
                return
            size = self._path.stat().st_size
        except OSError:
            return
        if size < _MAX_FILE_BYTES:
            return
        old = self._path.with_suffix(self._path.suffix + ".1")
        with contextlib.suppress(OSError):
            if old.exists():
                old.unlink()
            os.replace(self._path, old)
        log.info("perf_tracker_rotated", path=str(self._path))


# ── Default tracker singleton ──────────────────────────────────────────────

_DEFAULT_TRACKER: PerfTracker | None = None


def get_default_tracker() -> PerfTracker:
    """Lazy-singleton tracker for runtime use."""
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = PerfTracker()
    return _DEFAULT_TRACKER
