"""GDPR Art. 33 — Automatic breach detection and notification.

Scans the AuditLogger for SECURITY events with CRITICAL or ERROR severity,
generates breach reports with 72-hour notification deadlines, and persists
scan state to avoid duplicate notifications.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.audit import AuditLogger

log = logging.getLogger("jarvis.audit.breach")


class BreachDetector:
    """Detects potential data breaches from audit log entries.

    Implements GDPR Art. 33 requirement: notification within 72 hours.

    Args:
        state_path: JSON file for persisting scan state.
        cooldown_hours: Minimum hours between reports (prevents spam).
    """

    def __init__(
        self,
        state_path: Path,
        cooldown_hours: int | float = 1,
    ) -> None:
        self._state_path = state_path
        self._cooldown = timedelta(hours=cooldown_hours)
        self._state = self._load_state()

    # ── Public API ──────────────────────────────────────────────

    def scan(self, audit_logger: AuditLogger) -> list[dict[str, Any]]:
        """Scan audit entries for security breaches.

        Args:
            audit_logger: The AuditLogger instance to scan.

        Returns:
            List of breach report dicts (empty if cooldown active or
            no breaches found).
        """
        now = datetime.now(UTC)

        # Cooldown check
        last_report = self._state.get("last_report_time")
        if last_report:
            try:
                last_dt = datetime.fromisoformat(last_report)
                if now - last_dt < self._cooldown:
                    return []
            except (ValueError, TypeError):
                pass

        # Scan entries
        last_report_time = self._state.get("last_report_time")
        breaches: list[dict[str, Any]] = []

        for entry in audit_logger._entries:
            # Check category — handle both enum and string
            cat = entry.category
            cat_val = cat.value if hasattr(cat, "value") else str(cat)
            if cat_val != "security":
                continue

            # Check severity — handle both enum and string
            sev = entry.severity
            sev_val = sev.value if hasattr(sev, "value") else str(sev)
            if sev_val not in ("critical", "error"):
                continue

            # Skip entries already reported
            if last_report_time:
                try:
                    entry_ts = datetime.fromisoformat(entry.timestamp)
                    last_dt = datetime.fromisoformat(last_report_time)
                    if entry_ts <= last_dt:
                        continue
                except (ValueError, TypeError):
                    pass

            # Build breach report
            notification_deadline = datetime.fromisoformat(entry.timestamp) + timedelta(hours=72)

            breaches.append(
                {
                    "severity": sev_val,
                    "description": entry.description,
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "agent": entry.agent_name,
                    "tool": entry.tool_name,
                    "gdpr_article": "Art. 33 DSGVO",
                    "notification_deadline": notification_deadline.isoformat(),
                }
            )

        # Update state if breaches found
        if breaches:
            self._state["last_report_time"] = now.isoformat()
            self._state["last_breach_count"] = len(breaches)
            self._save_state()
            log.warning(
                "Breach detected: %d security event(s) require notification",
                len(breaches),
            )

        return breaches

    # ── State persistence ───────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        try:
            if self._state_path.exists():
                return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.debug("breach_state_load_failed", exc_info=True)
        return {}

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(self._state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            log.debug("breach_state_save_failed", exc_info=True)
