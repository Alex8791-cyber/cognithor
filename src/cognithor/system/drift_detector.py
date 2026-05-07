"""Layer 7 — Drift-Detector.

Two kinds of drift are tracked:
1. Hardware drift — capability-relevant fields changed since last apply
   (new GPU, driver upgrade unlocking NVFP4, RAM changed, …).
2. Performance drift — measured tok/s / first-token-latency is significantly
   below the manifest's `performance_estimates.*_p50`.

Hysteresis prevents banner-spam:
- Hardware drift: surfaced ONCE per profile_hash change, then 30-day cooldown.
- Performance drift: needs 3 consecutive 24h windows below 50% of expected
  before banner; cooldown 30 days after dismissal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cognithor.system.capabilities import Capabilities
from cognithor.system.manifest_models import Manifest
from cognithor.system.perf_tracker import PerfTracker, get_default_tracker
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DriftDetector",
    "DriftKind",
    "DriftReport",
    "DriftSeverity",
]


_HYSTERESIS_DEGRADED_WINDOWS = 3
_DEGRADATION_THRESHOLD = 0.5  # < 50% of expected p50 = degraded
_BANNER_COOLDOWN_DAYS = 30
_DRIFT_STATE_PATH_DEFAULT = Path.home() / ".cognithor" / ".drift_state.json"


# ── Types ──────────────────────────────────────────────────────────────────


DriftKind = str  # "hardware" | "performance" | "manifest_recall"
DriftSeverity = str  # "info" | "warn" | "error"


@dataclass(frozen=True)
class DriftReport:
    detected: bool
    kind: DriftKind = ""
    severity: DriftSeverity = "info"
    message: str = ""
    components: tuple[str, ...] = ()
    cooldown_active: bool = False
    cooldown_until_utc: str | None = None


@dataclass
class _DriftState:
    last_hardware_hash: str | None = None
    last_hardware_drift_seen_utc: str | None = None
    last_hardware_dismissed_utc: str | None = None
    consecutive_perf_degraded_windows: int = 0
    last_perf_drift_seen_utc: str | None = None
    last_perf_dismissed_utc: str | None = None
    perf_degradation_per_model: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> _DriftState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(
            last_hardware_hash=data.get("last_hardware_hash"),
            last_hardware_drift_seen_utc=data.get("last_hardware_drift_seen_utc"),
            last_hardware_dismissed_utc=data.get("last_hardware_dismissed_utc"),
            consecutive_perf_degraded_windows=int(data.get("consecutive_perf_degraded_windows", 0)),
            last_perf_drift_seen_utc=data.get("last_perf_drift_seen_utc"),
            last_perf_dismissed_utc=data.get("last_perf_dismissed_utc"),
            perf_degradation_per_model=dict(data.get("perf_degradation_per_model", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "last_hardware_hash": self.last_hardware_hash,
                    "last_hardware_drift_seen_utc": self.last_hardware_drift_seen_utc,
                    "last_hardware_dismissed_utc": self.last_hardware_dismissed_utc,
                    "consecutive_perf_degraded_windows": self.consecutive_perf_degraded_windows,
                    "last_perf_drift_seen_utc": self.last_perf_drift_seen_utc,
                    "last_perf_dismissed_utc": self.last_perf_dismissed_utc,
                    "perf_degradation_per_model": self.perf_degradation_per_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _days_since(utc_str: str | None) -> float:
    if not utc_str:
        return 10**9
    try:
        then = datetime.strptime(utc_str.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
        delta = datetime.now(UTC) - then
        return delta.total_seconds() / 86400
    except (ValueError, TypeError):
        return 10**9


# ── Detector ───────────────────────────────────────────────────────────────


class DriftDetector:
    """Stateful drift-checker with hysteresis + cooldowns."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        perf_tracker: PerfTracker | None = None,
        cooldown_days: int = _BANNER_COOLDOWN_DAYS,
        hysteresis_windows: int = _HYSTERESIS_DEGRADED_WINDOWS,
        degradation_threshold: float = _DEGRADATION_THRESHOLD,
    ) -> None:
        self._state_path = state_path or _DRIFT_STATE_PATH_DEFAULT
        self._tracker = perf_tracker or get_default_tracker()
        self._cooldown_days = cooldown_days
        self._hysteresis_windows = hysteresis_windows
        self._degradation_threshold = degradation_threshold
        self._state = _DriftState.load(self._state_path)

    # ── Hardware drift ────────────────────────────────────────────────

    def check_hardware_drift(
        self,
        current: Capabilities,
        last_known_hash: str | None,
    ) -> DriftReport:
        """Compare current profile_hash against last-applied. Surface once,
        cooldown for `cooldown_days`, save state."""
        if last_known_hash is None or last_known_hash == current.profile_hash:
            # No drift: clear any stale state
            if self._state.last_hardware_hash != current.profile_hash:
                self._state.last_hardware_hash = current.profile_hash
                self._state.save(self._state_path)
            return DriftReport(detected=False)

        # Drift detected
        in_cooldown = (
            self._state.last_hardware_dismissed_utc is not None
            and _days_since(self._state.last_hardware_dismissed_utc) < self._cooldown_days
            and self._state.last_hardware_hash == current.profile_hash
        )
        if in_cooldown:
            return DriftReport(
                detected=True,
                kind="hardware",
                severity="info",
                message=(
                    "Hardware-Änderung erkannt — Cooldown aktiv "
                    f"({self._cooldown_days} d nach letzter Auslösung)."
                ),
                components=("hardware_hash_mismatch",),
                cooldown_active=True,
                cooldown_until_utc=self._state.last_hardware_dismissed_utc,
            )

        # New drift event — record + return
        self._state.last_hardware_hash = current.profile_hash
        self._state.last_hardware_drift_seen_utc = _now_utc()
        self._state.save(self._state_path)
        return DriftReport(
            detected=True,
            kind="hardware",
            severity="warn",
            message=(
                "Hardware-Änderung erkannt seit letzter Konfiguration. "
                "`cognithor doctor --reconfigure` bietet aktualisierte Empfehlungen."
            ),
            components=("hardware_hash_mismatch",),
        )

    # ── Performance drift ─────────────────────────────────────────────

    def check_performance_drift(self, manifest: Manifest, current_tier_id: str) -> DriftReport:
        """Compare measured tok/s p95 vs manifest expected p50.

        Hysteresis: needs `hysteresis_windows` consecutive degraded windows
        before reporting. Resets on any non-degraded window."""
        tier = next((t for t in manifest.tiers if t.id == current_tier_id), None)
        if tier is None:
            return DriftReport(detected=False)
        expected = tier.performance_estimates.planner_tok_s_p50
        if expected <= 0:
            return DriftReport(detected=False)

        # Read measured perf for the planner model
        from cognithor.system.manifest_loader import ManifestLoader

        loader = ManifestLoader()
        try:
            full_manifest, _ = loader.load(prefer_online=False)
        except Exception:
            full_manifest = manifest
        planner_model = full_manifest.models.get(tier.model_set.planner)
        # The runtime backend-name (Ollama-tag or HF-id) is what's being measured
        if planner_model is not None:
            measured_id = planner_model.backend_ids.get(tier.backend) or planner_model.id
        else:
            measured_id = tier.model_set.planner

        measured_p95 = self._tracker.rolling_p95_tokens_per_s(measured_id)
        if measured_p95 is None:
            return DriftReport(detected=False)

        ratio = measured_p95 / expected if expected > 0 else 1.0
        degraded_now = ratio < self._degradation_threshold

        if degraded_now:
            self._state.consecutive_perf_degraded_windows += 1
            self._state.perf_degradation_per_model[measured_id] = (
                self._state.perf_degradation_per_model.get(measured_id, 0) + 1
            )
        else:
            self._state.consecutive_perf_degraded_windows = 0
            self._state.perf_degradation_per_model.pop(measured_id, None)
        self._state.save(self._state_path)

        if self._state.consecutive_perf_degraded_windows < self._hysteresis_windows:
            return DriftReport(detected=False)

        # Threshold reached — check cooldown
        in_cooldown = (
            self._state.last_perf_dismissed_utc is not None
            and _days_since(self._state.last_perf_dismissed_utc) < self._cooldown_days
        )
        if in_cooldown:
            return DriftReport(
                detected=True,
                kind="performance",
                severity="info",
                message="Performance-Drift detected — banner cooldown active.",
                components=(measured_id,),
                cooldown_active=True,
                cooldown_until_utc=self._state.last_perf_dismissed_utc,
            )

        self._state.last_perf_drift_seen_utc = _now_utc()
        self._state.save(self._state_path)
        return DriftReport(
            detected=True,
            kind="performance",
            severity="warn",
            message=(
                f"Performance unterhalb Erwartung: gemessen {measured_p95:.1f} tok/s p95 "
                f"vs Manifest-Erwartung {expected:.1f} tok/s p50 "
                f"(Verhältnis {ratio:.0%}). "
                "`cognithor doctor` zeigt Re-Empfehlungen."
            ),
            components=(measured_id,),
        )

    # ── Banner-dismissal ──────────────────────────────────────────────

    def dismiss_hardware_banner(self) -> None:
        self._state.last_hardware_dismissed_utc = _now_utc()
        self._state.save(self._state_path)

    def dismiss_performance_banner(self) -> None:
        self._state.last_perf_dismissed_utc = _now_utc()
        self._state.consecutive_perf_degraded_windows = 0
        self._state.save(self._state_path)
