"""Adaptive Skill-Performance-Tracking: auto-disable degraded skills.

Monitors skill execution outcomes over a sliding window and automatically
disables skills that exceed failure thresholds.  Skills are re-enabled
after a configurable cooldown period so they get a second chance.

Persistence is stored in ``~/.cognithor/data/skill_performance.json``.

.. note::

    **NOT WIRED INTO PRODUCTION** (operational-trust audit, A-8,
    2026-05-04). Tested via
    ``tests/test_skills/test_performance_tracker.py`` but no
    production boot path imports the tracker. The marketplace
    feature that depends on this (auto-disable degraded skills) is
    on the v1.0 roadmap.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DegradationConfig",
    "SkillHealth",
    "SkillPerformanceTracker",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DegradationConfig:
    """Thresholds that control when a skill is considered degraded."""

    min_executions: int = 5
    failure_rate_threshold: float = 0.6
    min_avg_score: float = 0.3
    cooldown_seconds: int = 3600
    max_consecutive_failures: int = 3
    window_size: int = 20


@dataclass
class ExecutionRecord:
    """One recorded execution of a skill."""

    timestamp: float
    success: bool
    score: float
    duration_ms: int


@dataclass
class SkillHealth:
    """Health snapshot for a single skill."""

    skill_name: str
    total_executions: int = 0
    window_executions: int = 0
    failure_rate: float = 0.0
    avg_score: float = 0.0
    avg_duration_ms: float = 0.0
    is_degraded: bool = False
    degraded_since: float | None = None
    cooldown_remaining_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Internal per-skill state (not exposed directly)
# ---------------------------------------------------------------------------


@dataclass
class _SkillState:
    total_executions: int = 0
    consecutive_failures: int = 0
    is_degraded: bool = False
    degraded_since: float | None = None
    window: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class SkillPerformanceTracker:
    """Thread-safe async tracker that auto-disables degraded skills."""

    def __init__(
        self,
        config: DegradationConfig | None = None,
        data_path: Path | None = None,
    ) -> None:
        self._config = config or DegradationConfig()
        if data_path is None:
            data_path = Path.home() / ".cognithor" / "data" / "skill_performance.json"
        self._data_path = data_path
        self._lock = asyncio.Lock()
        self._states: dict[str, _SkillState] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self._load()

    async def _load(self) -> None:
        try:
            if self._data_path.exists():
                raw = self._data_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                for name, blob in data.items():
                    st = _SkillState(
                        total_executions=blob.get("total_executions", 0),
                        consecutive_failures=blob.get("consecutive_failures", 0),
                        is_degraded=blob.get("is_degraded", False),
                        degraded_since=blob.get("degraded_since"),
                        window=[r for r in blob.get("window", [])],
                    )
                    self._states[name] = st
                log.debug("perf_tracker_loaded", skills=len(self._states))
        except Exception as exc:
            log.warning("perf_tracker_load_error", error=str(exc))
        self._loaded = True

    async def _save(self) -> None:
        try:
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {}
            for name, st in self._states.items():
                data[name] = {
                    "total_executions": st.total_executions,
                    "consecutive_failures": st.consecutive_failures,
                    "is_degraded": st.is_degraded,
                    "degraded_since": st.degraded_since,
                    "window": st.window,
                }
            self._data_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("perf_tracker_save_error", error=str(exc))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def record_execution(
        self,
        skill_name: str,
        success: bool,
        score: float = 0.0,
        duration_ms: int = 0,
    ) -> None:
        """Record a skill execution and evaluate degradation rules."""
        async with self._lock:
            await self._ensure_loaded()

            st = self._states.setdefault(skill_name, _SkillState())
            st.total_executions += 1

            # Append to sliding window
            record = {
                "timestamp": time.time(),
                "success": success,
                "score": score,
                "duration_ms": duration_ms,
            }
            st.window.append(record)

            # Trim window
            if len(st.window) > self._config.window_size:
                st.window = st.window[-self._config.window_size :]

            # Consecutive failures
            if success:
                st.consecutive_failures = 0
            else:
                st.consecutive_failures += 1

            # Evaluate degradation
            self._evaluate(skill_name, st)

            await self._save()

    def _evaluate(self, skill_name: str, st: _SkillState) -> None:
        """Check degradation rules (called while lock is held)."""
        # If already degraded, check cooldown
        if st.is_degraded:
            if st.degraded_since is not None:
                elapsed = time.time() - st.degraded_since
                if elapsed >= self._config.cooldown_seconds:
                    # Cooldown expired -- re-enable for retry
                    st.is_degraded = False
                    st.degraded_since = None
                    st.consecutive_failures = 0
                    log.info(
                        "skill_cooldown_expired",
                        skill=skill_name,
                        msg="Re-enabled after cooldown",
                    )
            return

        window = st.window
        win_len = len(window)

        # Rule 1: instant disable on N consecutive failures
        if st.consecutive_failures >= self._config.max_consecutive_failures:
            self._degrade(skill_name, st, reason="consecutive_failures")
            return

        # Don't judge until min_executions reached
        if win_len < self._config.min_executions:
            return

        # Rule 2: failure rate in window
        failures = sum(1 for r in window if not r["success"])
        failure_rate = failures / win_len
        if failure_rate > self._config.failure_rate_threshold:
            self._degrade(skill_name, st, reason="high_failure_rate")
            return

        # Rule 3: avg score too low
        scores = [r["score"] for r in window]
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg < self._config.min_avg_score:
            self._degrade(skill_name, st, reason="low_avg_score")
            return

    def _degrade(self, skill_name: str, st: _SkillState, *, reason: str) -> None:
        st.is_degraded = True
        st.degraded_since = time.time()
        log.warning(
            "skill_degraded",
            skill=skill_name,
            reason=reason,
            consecutive_failures=st.consecutive_failures,
            window_size=len(st.window),
        )

    async def is_degraded(self, skill_name: str) -> bool:
        """Return True if the skill is currently degraded."""
        async with self._lock:
            await self._ensure_loaded()
            st = self._states.get(skill_name)
            if st is None:
                return False
            # Check cooldown transparently
            if st.is_degraded and st.degraded_since is not None:
                elapsed = time.time() - st.degraded_since
                if elapsed >= self._config.cooldown_seconds:
                    st.is_degraded = False
                    st.degraded_since = None
                    st.consecutive_failures = 0
                    await self._save()
            return st.is_degraded

    async def get_skill_health(self, skill_name: str) -> SkillHealth:
        """Build a health snapshot for one skill."""
        async with self._lock:
            await self._ensure_loaded()
            return self._build_health(skill_name)

    async def get_all_health(self) -> dict[str, SkillHealth]:
        """Health snapshots for every tracked skill."""
        async with self._lock:
            await self._ensure_loaded()
            return {name: self._build_health(name) for name in self._states}

    async def get_degraded_skills(self) -> list[str]:
        """List skill names currently degraded (respecting cooldown)."""
        async with self._lock:
            await self._ensure_loaded()
            now = time.time()
            result: list[str] = []
            for name, st in self._states.items():
                if st.is_degraded:
                    if (
                        st.degraded_since is not None
                        and (now - st.degraded_since) >= self._config.cooldown_seconds
                    ):
                        # cooldown expired
                        st.is_degraded = False
                        st.degraded_since = None
                        st.consecutive_failures = 0
                    else:
                        result.append(name)
            return result

    async def reset_skill(self, skill_name: str) -> None:
        """Manually re-enable a degraded skill."""
        async with self._lock:
            await self._ensure_loaded()
            st = self._states.get(skill_name)
            if st is None:
                return
            st.is_degraded = False
            st.degraded_since = None
            st.consecutive_failures = 0
            st.window.clear()
            log.info("skill_manually_reset", skill=skill_name)
            await self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_health(self, skill_name: str) -> SkillHealth:
        """Build SkillHealth without acquiring the lock (caller holds it)."""
        st = self._states.get(skill_name)
        if st is None:
            return SkillHealth(skill_name=skill_name)

        window = st.window
        win_len = len(window)
        failures = sum(1 for r in window if not r["success"]) if win_len else 0
        failure_rate = failures / win_len if win_len else 0.0
        scores = [r["score"] for r in window]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        durations = [r["duration_ms"] for r in window]
        avg_dur = sum(durations) / len(durations) if durations else 0.0

        cooldown_remaining = 0.0
        if st.is_degraded and st.degraded_since is not None:
            elapsed = time.time() - st.degraded_since
            cooldown_remaining = max(0.0, self._config.cooldown_seconds - elapsed)

        return SkillHealth(
            skill_name=skill_name,
            total_executions=st.total_executions,
            window_executions=win_len,
            failure_rate=round(failure_rate, 4),
            avg_score=round(avg_score, 4),
            avg_duration_ms=round(avg_dur, 2),
            is_degraded=st.is_degraded,
            degraded_since=st.degraded_since,
            cooldown_remaining_seconds=round(cooldown_remaining, 1),
        )
